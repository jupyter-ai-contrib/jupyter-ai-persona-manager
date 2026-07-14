import asyncio
import html
import os
import traceback
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import asdict
from logging import Logger
from time import time
from typing import TYPE_CHECKING, Any

from jupyterlab_chat.models import Message, NewMessage, User
from jupyterlab_chat.utils import find_mentions
from pydantic import BaseModel
from traitlets import MetaHasTraits
from traitlets.config import LoggingConfigurable

from .awareness_models import (
    CommandOption,
    ModelConfiguration,
    ModelSpec,
    PersonaAwarenessState,
    SettingConfiguration,
    Usage,
)
from .persona_awareness import PersonaAwareness

# prevents a circular import
# types imported under this block have to be surrounded in single quotes on use
if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from jupyterlab_chat.ychat import YChat
    from .mcp_server_models import McpSettings
    from .persona_manager import PersonaManager


class PersonaDefaults(BaseModel):
    """
    Data structure that represents the default settings of a persona. Each persona
    must define some basic default settings, like its name.

    Each of these settings can be overwritten through the settings UI.
    """

    ################################################
    # required fields
    ################################################
    name: str  # e.g. "Jupyternaut"
    description: str  # e.g. "..."
    avatar_path: str  # Absolute filesystem path to avatar image file (SVG, PNG, or JPG)
    system_prompt: str  # e.g. "You are a language model named..."

    ################################################
    # optional fields
    ################################################
    slash_commands: set[str] = set("*")  # change this to enable/disable slash commands
    model_uid: str | None = None  # e.g. "ollama:deepseek-coder-v2"
    # ^^^ set this to automatically default to a model after a fresh start, no config file


class ABCLoggingConfigurableMeta(ABCMeta, MetaHasTraits):
    """
    Metaclass required for `BasePersona` to inherit from both `ABC` and
    `LoggingConfigurable`. This pattern is also followed by `BaseFileIdManager`
    from `jupyter_server_fileid`.
    """


class BasePersona(ABC, LoggingConfigurable, metaclass=ABCLoggingConfigurableMeta):
    """
    Abstract base class that defines a persona when implemented.
    """

    ychat: "YChat"
    """
    Reference to the `YChat` that this persona instance is scoped to.
    Automatically set by `BasePersona`.
    """

    parent: "PersonaManager"  # type: ignore
    """
    Reference to the `PersonaManager` for this `YChat`, which manages this
    instance. Automatically set by the `LoggingConfigurable` parent class.
    """

    log: Logger  # type: ignore
    """
    The `logging.Logger` instance used by this class. Automatically set by the
    `LoggingConfigurable` parent class.
    """

    awareness: PersonaAwareness
    """
    A custom wrapper around `self.ychat.awareness: pycrdt.Awareness`. The
    default `Awareness` API does not support setting local state fields on
    multiple AI personas, so any awareness method calls should be done through
    this attribute (`self.awareness`) instead of `self.ychat.awareness`. See the
    documentation in `PersonaAwareness` for more information.

    Automatically set by `BasePersona`.
    """

    _awareness_state: PersonaAwarenessState
    """
    This persona's awareness state (model configuration, settings, usage, and
    slash commands). Broadcast over the awareness channel whenever it changes.
    Always mutate this through the `update_*` methods so a rebroadcast happens;
    read it through the `get_*` methods. Automatically initialized by
    `BasePersona`.
    """

    ################################################
    # constructor
    ################################################
    def __init__(
        self,
        *args,
        ychat: "YChat",
        **kwargs,
    ):
        # Forward other arguments to parent class
        super().__init__(*args, **kwargs)

        # Bind arguments to instance attributes
        self.ychat = ychat

        # Initialize custom awareness object for this persona
        self.awareness = PersonaAwareness(
            ychat=self.ychat, log=self.log, user=self.as_user()
        )

        # Initialize this persona's awareness state (model configuration,
        # settings, usage, and slash commands) and broadcast it. Subclasses fill
        # in the model/settings configuration once they know it (e.g. an ACP
        # persona does so on session create/load); a persona that never touches
        # any of this simply carries the defaults.
        self._awareness_state = PersonaAwarenessState(id=self.id)
        self._broadcast_awareness_state()

        # Register this persona as a user in the chat
        self.ychat.set_user(self.as_user())

    ################################################
    # abstract methods, required by subclasses.
    ################################################
    @property
    @abstractmethod
    def defaults(self) -> PersonaDefaults:
        """
        Returns a `PersonaDefaults` data model that represents the default
        settings of this persona.

        This is an abstract method that must be implemented by subclasses.
        """

    @abstractmethod
    async def process_message(self, message: Message) -> None:
        """
        Processes a new message. This method exclusively defines how new
        messages are handled by a persona, and should be considered the "main
        entry point" to this persona. Reading chat history and streaming a reply
        can be done through method calls to `self.ychat`. See
        `JupyternautPersona` for a reference implementation on how to do so.

        This is an abstract method that must be implemented by subclasses.
        """

    @abstractmethod
    async def update_model(self, model_id: str) -> None:
        """
        Switch this persona to the model identified by `model_id`.

        Called by `apply_model_spec()` when a message specifies a model that
        differs from the current one. Implementations should perform the switch
        and then update the broadcast model configuration (e.g. via
        `set_model_configuration()`) so the new current model reaches clients.

        This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError()

    @abstractmethod
    async def update_model_settings(self, settings: dict[str, str]) -> None:
        """
        Apply the given model settings (e.g. context size), keyed by setting ID.

        Called by `apply_model_spec()` when a message specifies model settings
        that differ from the current ones. Implementations should apply the
        settings and update `self.awareness` so the new values are broadcast.

        This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError()

    @abstractmethod
    async def update_settings(self, settings: dict[str, str]) -> None:
        """
        Apply the given general settings (e.g. mode, effort level), keyed by
        setting ID. This is separate from model settings.

        Called by `apply_settings_spec()` when a message specifies settings that
        differ from the current ones. Implementations should apply the settings
        and update `self.awareness` so the new values are broadcast.

        This is an abstract method that must be implemented by subclasses.
        """
        raise NotImplementedError()

    ################################################
    # base class methods, available to subclasses.
    ################################################
    @property
    def id(self) -> str:
        """
        Returns a static & unique ID for this persona. This sets the `username`
        field in the data model returned by `self.as_user()`.

        The ID is guaranteed to follow the format
        `jupyter-ai-personas::<package-name>::<persona-class-name>`. The prefix
        allows consumers to easily distinguish AI personas from human users.

        If a package provides multiple personas, their class names must be
        different to ensure that their IDs are unique.
        """
        package_name = self.__module__.split(".")[0]
        class_name = self.__class__.__name__
        return f"jupyter-ai-personas::{package_name}::{class_name}"

    @property
    def name(self) -> str:
        """
        Returns the name shown on messages from this persona in the chat. This
        sets the `name` and `display_name` fields in the data model returned by
        `self.as_user()`. Provided by `BasePersona`.

        NOTE/TODO: This currently just returns the value set in `self.defaults`.
        This is set here because we may require this field to be configurable
        for all personas in the future.
        """
        return self.defaults.name

    @property
    def avatar_path(self) -> str:
        """
        Returns the API URL route that serves the avatar for this persona.

        The avatar is served at `{base_url}api/ai/avatars/{id}` where the ID is the
        unique persona identifier. This ensures that each persona has a unique
        avatar URL without exposing filesystem paths.

        The base_url is obtained from the PersonaManager and ensures the URL works
        correctly in both JupyterLab standalone and JupyterHub environments.

        The actual avatar file path is specified in `defaults.avatar_path` as an
        absolute filesystem path to an image file (SVG, PNG, or JPG) within the
        persona's package or module.

        This sets the `avatar_url` field in the data model returned by
        `self.as_user()`. Provided by `BasePersona`.
        """
        # URL-encode the persona ID to handle special characters
        from urllib.parse import quote
        base_url = getattr(self.parent, 'base_url', '/')
        # Ensure base_url ends with '/' for proper path joining
        if not base_url.endswith('/'):
            base_url += '/'
        return f"{base_url}api/ai/avatars/{quote(self.id, safe='')}"

    @property
    def system_prompt(self) -> str:
        """
        Returns the system prompt used by this persona. Provided by `BasePersona`.

        NOTE/TODO: This currently just returns the value set in `self.defaults`.
        This is set here because we may require this field to be configurable
        for all personas in the future.
        """
        return self.defaults.system_prompt

    @property
    def event_loop(self) -> asyncio.AbstractEventLoop:
        """
        The asyncio event loop running this process.
        """
        return self.parent.event_loop

    def as_user(self) -> User:
        """
        Returns the `jupyterlab_chat.models:User` model that represents this
        persona in the chat. This model also includes all attributes from
        `jupyter_server.auth:JupyterUser`, the user model returned by the
        `IdentityProvider` in Jupyter Server.

        This method is provided by `BasePersona`.
        """
        return User(
            username=self.id,
            name=self.name,
            display_name=self.name,
            avatar_url=self.avatar_path,
            bot=True,
        )

    def as_user_dict(self) -> dict[str, Any]:
        """
        Returns `self.as_user()` as a Python dictionary. This method is provided
        by `BasePersona`.
        """
        user = self.as_user()
        return asdict(user)

    async def stream_message(
        self, reply_stream: "AsyncIterator"
    ) -> None:
        """
        Takes an async iterator, dubbed the 'reply stream', and streams it to a
        new message by this persona in the YChat. The async iterator may yield
        either strings or `litellm.ModelResponseStream` objects. Details:

        - Creates a new message upon receiving the first chunk from the reply
        stream, then continuously updates it until the stream is closed.

        - Automatically manages its awareness state to show writing status.

        - Triggers mention detection after streaming completes, allowing
        personas to mention each other in their responses.
        """
        stream_id: str | None = None
        try:
            self.awareness.set_local_state_field("isWriting", True)
            async for chunk in reply_stream:
                # Coerce LiteLLM stream chunk to a string delta
                if not isinstance(chunk, str):
                    chunk = chunk.choices[0].delta.content

                # LiteLLM streams always terminate with an empty chunk, so we
                # ignore and continue when this occurs.
                if not chunk:
                    continue

                if not stream_id:
                    stream_id = self.ychat.add_message(
                        NewMessage(body="", sender=self.id)
                    )
                    self.awareness.set_local_state_field("isWriting", stream_id)

                assert stream_id
                self.ychat.update_message(
                    Message(
                        id=stream_id,
                        body=chunk,
                        time=time(),
                        sender=self.id,
                        raw_time=False,
                    ),
                    append=True,
                    trigger_actions=[],  # Defer mention extraction during streaming
                )

            # Stream complete - trigger mention extraction and notifications
            if stream_id:
                msg = self.ychat.get_message(stream_id)
                if msg:
                    self.ychat.update_message(
                        msg,
                        trigger_actions=[find_mentions],  # Extract mentions and notify mentioned personas
                    )
        except Exception as e:
            self.log.error(
                f"Persona '{self.name}' encountered an exception printed below when attempting to stream output."
            )
            self.log.exception(e)
            raise
        finally:
            self.awareness.set_local_state_field("isWriting", False)

    def send_message(self, body: str) -> None:
        """
        Sends a new message to the chat from this persona.
        """
        self.ychat.add_message(NewMessage(body=body, sender=self.id))

    async def handle_uncaught_exception(self, exc: Exception) -> None:
        """
        Called by PersonaManager when process_message() raises an unhandled
        exception. Override this method to customize error reporting.

        The default implementation sends a message to the chat with the error
        type and message visible in the summary, and the full traceback hidden
        under a collapsible <details> element.
        """
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        error_type = type(exc).__name__
        error_msg = str(exc)
        if len(error_msg) > 120:
            error_msg = error_msg[:120] + "…"
        summary = f"{html.escape(error_type)}: {html.escape(error_msg)}"
        body = (
            f"An error occurred while processing your message.\n\n"
            f'<details class="jp-jai-error-details">\n'
            f"<summary>Error details ({summary})</summary>\n"
            f'<pre class="jp-jai-error-traceback">{html.escape(tb)}</pre>\n'
            f"</details>"
        )
        self.send_message(body)

    def get_chat_path(self, relative: bool = False) -> str:
        """
        Returns the absolute path of the chat file assigned to this persona.

        To get a path relative to the `ContentsManager` root directory, call
        this method with `relative=True`.
        """
        return self.parent.get_chat_path(relative=relative)

    def get_chat_dir(self) -> str:
        """
        Returns the absolute path to the parent directory of the chat file
        assigned to this persona.
        """
        return self.parent.get_chat_dir()

    def get_dotjupyter_dir(self) -> str | None:
        """
        Returns the path to the .jupyter directory for the current chat.
        """
        return self.parent.get_dotjupyter_dir()

    def get_workspace_dir(self) -> str:
        """
        Returns the path to the workspace directory for the current chat.
        """
        return self.parent.get_workspace_dir()

    def get_mcp_settings(self) -> "McpSettings | None":
        """
        Returns the MCP config for the current chat.
        """
        return self.parent.get_mcp_settings()

    ################################################
    # awareness state: reading & broadcasting session information
    ################################################
    def _broadcast_awareness_state(self) -> None:
        """
        Write this persona's awareness state to `self.awareness`, which triggers
        a rebroadcast over the Yjs awareness channel. Called after any change to
        `self._awareness_state`.
        """
        self.awareness.set_local_state_field(
            "persona", self._awareness_state.model_dump()
        )

    def get_model_configuration(self) -> ModelConfiguration:
        """Return the current model, model settings, and all options for both."""
        return self._awareness_state.model

    def get_setting_configurations(self) -> list[SettingConfiguration]:
        """Return the current value and all options for each general setting."""
        return self._awareness_state.settings

    def get_model(self) -> str | None:
        """Return the current model ID, or None if using the default."""
        return self._awareness_state.model.current

    def get_model_settings(self) -> dict[str, str | None]:
        """Return the current model settings, keyed by setting ID."""
        return {s.id: s.current for s in self._awareness_state.model.settings}

    def get_settings(self) -> dict[str, str | None]:
        """
        Return the current general settings, keyed by setting ID. This is
        separate from the model settings returned by `get_model_settings()`.
        """
        return {s.id: s.current for s in self._awareness_state.settings}

    def get_usage(self) -> Usage:
        """Return the usage currently reported by this persona."""
        return self._awareness_state.usage

    def get_slash_commands(self) -> list[CommandOption]:
        """Return the slash commands currently advertised by this persona."""
        return self._awareness_state.slash_commands

    ################################################
    # awareness state: setting model, settings, usage, and slash commands
    ################################################
    def set_model_configuration(self, model: ModelConfiguration) -> None:
        """
        Replace the whole model configuration (current model, model options, and
        model settings) and rebroadcast. Personas call this once they know their
        model configuration (e.g. an ACP persona on session create/load).
        """
        self._awareness_state.model = model
        self._broadcast_awareness_state()

    def set_setting_configurations(
        self, settings: list[SettingConfiguration]
    ) -> None:
        """
        Replace the general (non-model) setting configurations and rebroadcast.
        """
        self._awareness_state.settings = settings
        self._broadcast_awareness_state()

    def update_usage(self, usage: Usage, *, append: bool = False) -> None:
        """
        Merge `usage` into the reported usage and rebroadcast. Only the fields
        set on `usage` are touched, so a source that reports context and tokens
        in separate calls composes into one Usage.

        append=False (default): each provided field replaces the stored value.
        Use for sources that already report totals — ACP reports cumulative
        counts and a live context snapshot, so replace is correct, and it's what
        we prioritize.

        append=True: each provided field is added to the stored value, for
        sources that emit per-turn deltas. Snapshot fields (context_*) should
        not be sent this way — you don't sum window sizes.
        """
        current = self._awareness_state.usage
        # Only the fields explicitly set on `usage` are merged; unset fields
        # (still None because they were never provided) leave the stored value
        # untouched.
        provided = usage.model_dump(exclude_none=True)
        for field, value in provided.items():
            if append:
                existing = getattr(current, field)
                setattr(current, field, (existing or 0) + value)
            else:
                setattr(current, field, value)
        self._broadcast_awareness_state()

    def update_slash_commands(self, commands: list[CommandOption]) -> None:
        """Replace the advertised slash commands and rebroadcast."""
        self._awareness_state.slash_commands = commands
        self._broadcast_awareness_state()

    ################################################
    # applying a message's model & settings specification
    ################################################
    async def apply_model_spec(self, spec: ModelSpec) -> None:
        """
        Apply a user's specified model and model settings.

        A None model ID or a None setting value means "use the persona's current
        value", so it is skipped. An `update_*` method is only called when a
        specified value actually differs from the current one.
        """
        if spec.id is not None and spec.id != self.get_model():
            await self.update_model(spec.id)

        current_model_settings = self.get_model_settings()
        for key, value in spec.settings.items():
            if value is not None and value != current_model_settings.get(key):
                await self.update_model_settings(spec.settings)
                break

    async def apply_settings_spec(self, spec: dict[str, str | None]) -> None:
        """
        Apply a user's specified general settings.

        A None value for a setting means "use the persona's current value", so
        it is skipped. `update_settings` is only called when a specified value
        actually differs from the current one.
        """
        current_settings = self.get_settings()
        for key, value in spec.items():
            if value is not None and value != current_settings.get(key):
                await self.update_settings(spec)
                break

    async def apply_message_metadata(self, message: Message) -> None:
        """
        Apply the model and settings specification carried on a message's
        metadata before the message is processed. Called by the `PersonaManager`
        for every routed message, so a persona picks up per-message selections
        without each `process_message()` implementation having to do so itself.

        A message with no relevant metadata is a no-op.
        """
        metadata = message.metadata or {}

        model_meta = metadata.get("model")
        if model_meta is not None:
            spec = (
                model_meta
                if isinstance(model_meta, ModelSpec)
                else ModelSpec(**model_meta)
            )
            await self.apply_model_spec(spec)

        settings_meta = metadata.get("settings")
        if settings_meta:
            await self.apply_settings_spec(settings_meta)

    def process_attachments(self, message: Message) -> str | None:
        """
        Process file attachments in the message and return their content as a string.
        """

        if not message.attachments:
            return None

        context_parts = []

        for attachment_id in message.attachments:
            self.log.info(f"FILE: Processing attachment with ID: {attachment_id}")
            try:
                # Try to resolve attachment using multiple strategies
                file_path = self.resolve_attachment_to_path(attachment_id)

                if not file_path:
                    self.log.warning(
                        f"Could not resolve attachment ID: {attachment_id}"
                    )
                    continue

                # Read the file content
                with open(file_path, encoding="utf-8") as f:
                    file_content = f.read()

                # Get relative path for display
                rel_path = os.path.relpath(file_path, self.get_workspace_dir())

                # Add file content with header
                context_parts.append(f"File: {rel_path}\n```\n{file_content}\n```")

            except Exception as e:
                self.log.warning(f"Failed to read attachment {attachment_id}: {e}")
                context_parts.append(
                    f"Attachment: {attachment_id} (could not read file: {e})"
                )

        result = "\n\n".join(context_parts) if context_parts else None
        return result

    def resolve_attachment_to_path(self, attachment_id: str) -> str | None:
        """
        Resolve an attachment ID to its file path using multiple strategies.
        """

        try:
            attachment_data = self.ychat.get_attachments().get(attachment_id)

            if attachment_data and isinstance(attachment_data, dict):
                # If attachment has a 'value' field with filename
                if "value" in attachment_data:
                    filename = attachment_data["value"]

                    # Try relative to workspace directory
                    workspace_path = os.path.join(self.get_workspace_dir(), filename)
                    if os.path.exists(workspace_path):
                        return workspace_path

                    # Try as absolute path
                    if os.path.exists(filename):
                        return filename

            return None

        except Exception as e:
            self.log.error(f"Failed to resolve attachment {attachment_id}: {e}")
            return None

    async def shutdown(self) -> None:
        """
        Shuts the persona down. This method should:

        - Halt all background tasks started by this instance.
        - Remove the persona from the chat awareness

        This method will be called when `/refresh-personas` is run, and may be
        called when the server is shutting down or when a chat session is
        closed.

        Subclasses may need to override this method to add custom shutdown
        logic. The override should generally call `super().shutdown()` first
        before running custom shutdown logic.
        """
        # Stop awareness heartbeat task & remove self from chat awareness
        self.awareness.shutdown()


class GenerationInterrupted(asyncio.CancelledError):
    """Exception raised when streaming is cancelled by the user"""