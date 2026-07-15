import asyncio
import contextlib
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
    SettingConfiguration,
    Usage,
)
from .doc_markers import (
    mark_consumer_api,
    mark_optional,
    mark_recommended,
    mark_required,
    mark_subclass_api,
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
    This persona's awareness slot: a `PersonaAwareness` scoped to this persona's
    own Yjs client ID. It holds the persona's broadcast session state (model
    configuration, settings, usage, slash commands, writing status) as typed
    properties, so `self.awareness.model = ...` publishes over the awareness
    channel directly. Use this instead of `self.ychat.awareness`, whose default
    API cannot set state for more than one client ID. See `PersonaAwareness` and
    its base `ScopedAwareness` for details.

    Automatically set by `BasePersona`.
    """

    _processing_count: int
    """
    Number of messages this persona is currently processing. Incremented while a
    `process_message()` call is in flight and decremented when it finishes (see
    `track_processing`). A count because a persona can process several messages
    concurrently. Exposed read-only via the `processing` property.
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
        self._processing_count = 0

        # Initialize this persona's awareness slot. It starts with a default
        # (empty) session state already published; the persona fills in its
        # model/settings/usage once it knows them (e.g. an ACP persona on session
        # create/load), and a persona that never does simply carries the defaults.
        self.awareness = PersonaAwareness(
            ychat=self.ychat, log=self.log, user=self.as_user(), id=self.id
        )

        # Register this persona as a user in the chat
        self.ychat.set_user(self.as_user())

    ################################################
    # abstract methods, required by subclasses.
    ################################################
    @mark_required
    @property
    @abstractmethod
    def defaults(self) -> PersonaDefaults:
        """
        Returns a `PersonaDefaults` data model that represents the default
        settings of this persona.

        This is an abstract method that must be implemented by subclasses.
        """

    @mark_required
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

    @mark_recommended
    async def cancel_response(self) -> None:
        """
        Stops this persona's in-progress response, if any. Called when the user
        interrupts the persona from the chat UI.

        This is the counterpart to `process_message`: it should halt whatever
        that method set in motion (a model stream, an agent turn, pending tool
        calls) so the persona stops writing to the chat promptly. `stream_message`
        already clears the `is_writing` awareness flag when it unwinds, so a
        subclass typically just needs to interrupt its own backend here.

        Optional: the default implementation is a no-op, for personas that have
        nothing cancellable or complete their responses synchronously. A persona
        that streams or runs a long-lived turn (e.g. an ACP agent) overrides this
        to actually interrupt it.

        Only invoked for a persona that's currently `processing` — the cancel
        handler gates on that — so an override can assume a response is in
        flight. This matters because some backends treat a cancel with no active
        response as an error (e.g. ACP defines `session/cancel` only for an
        ongoing prompt turn).
        """

    @mark_consumer_api
    @property
    def processing(self) -> bool:
        """
        Whether this persona is currently processing at least one message — i.e.
        a `process_message()` call is in flight. Use this to avoid interrupting a
        persona that has no response to cancel (see `cancel_response`).
        """
        return self._processing_count > 0

    @mark_consumer_api
    @contextlib.contextmanager
    def track_processing(self):
        """
        Context manager that marks this persona as processing for its duration,
        so `processing` reflects an in-flight response. `PersonaManager` wraps
        each `process_message()` call in this; the count is restored even if the
        call raises. A count (not a bool) because a persona may process several
        messages concurrently.
        """
        self._processing_count += 1
        try:
            yield
        finally:
            self._processing_count -= 1

    ################################################
    # base class methods, available to subclasses.
    ################################################
    @mark_consumer_api
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

    @mark_consumer_api
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

    @mark_consumer_api
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

    @mark_subclass_api
    @property
    def system_prompt(self) -> str:
        """
        Returns the system prompt used by this persona. Provided by `BasePersona`.

        NOTE/TODO: This currently just returns the value set in `self.defaults`.
        This is set here because we may require this field to be configurable
        for all personas in the future.
        """
        return self.defaults.system_prompt

    @mark_subclass_api
    @property
    def event_loop(self) -> asyncio.AbstractEventLoop:
        """
        The asyncio event loop running this process.
        """
        return self.parent.event_loop

    @mark_consumer_api
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

    @mark_consumer_api
    def as_user_dict(self) -> dict[str, Any]:
        """
        Returns `self.as_user()` as a Python dictionary. This method is provided
        by `BasePersona`.
        """
        user = self.as_user()
        return asdict(user)

    @mark_subclass_api
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
            self.awareness.is_writing = True
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
                    self.awareness.is_writing = stream_id

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
            self.awareness.is_writing = False

    @mark_subclass_api
    def send_message(self, body: str) -> None:
        """
        Sends a new message to the chat from this persona.
        """
        self.ychat.add_message(NewMessage(body=body, sender=self.id))

    @mark_optional
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

    @mark_subclass_api
    def get_chat_path(self, relative: bool = False) -> str:
        """
        Returns the absolute path of the chat file assigned to this persona.

        To get a path relative to the `ContentsManager` root directory, call
        this method with `relative=True`.
        """
        return self.parent.get_chat_path(relative=relative)

    @mark_subclass_api
    def get_chat_dir(self) -> str:
        """
        Returns the absolute path to the parent directory of the chat file
        assigned to this persona.
        """
        return self.parent.get_chat_dir()

    @mark_subclass_api
    def get_dotjupyter_dir(self) -> str | None:
        """
        Returns the path to the .jupyter directory for the current chat.
        """
        return self.parent.get_dotjupyter_dir()

    @mark_subclass_api
    def get_workspace_dir(self) -> str:
        """
        Returns the path to the workspace directory for the current chat.
        """
        return self.parent.get_workspace_dir()

    @mark_subclass_api
    def get_mcp_settings(self) -> "McpSettings | None":
        """
        Returns the MCP config for the current chat.
        """
        return self.parent.get_mcp_settings()

    ################################################
    # reading session information
    ################################################
    # These `get_*` readers and the `report_*` setters below are thin views over
    # `self.awareness`, whose typed properties are backed directly by the Yjs
    # awareness slot — reading returns the currently-published value and setting
    # rebroadcasts. There is no separate in-memory copy to keep in sync.
    @mark_subclass_api
    def get_model_configuration(self) -> ModelConfiguration:
        """Return the current model, model settings, and all options for both."""
        return self.awareness.model

    @mark_subclass_api
    def get_setting_configurations(self) -> list[SettingConfiguration]:
        """Return the current value and all options for each general setting."""
        return self.awareness.settings

    @mark_subclass_api
    def get_model(self) -> str | None:
        """Return the current model ID, or None if using the default."""
        return self.awareness.model.current

    @mark_subclass_api
    def get_model_settings(self) -> dict[str, str | None]:
        """Return the current model settings, keyed by setting ID."""
        return {s.id: s.current for s in self.awareness.model.settings}

    @mark_subclass_api
    def get_settings(self) -> dict[str, str | None]:
        """
        Return the current general settings, keyed by setting ID. This is
        separate from the model settings returned by `get_model_settings()`.
        """
        return {s.id: s.current for s in self.awareness.settings}

    @mark_subclass_api
    def get_usage(self) -> Usage:
        """Return the usage currently reported by this persona."""
        return self.awareness.usage

    @mark_subclass_api
    def get_slash_commands(self) -> list[CommandOption]:
        """Return the slash commands currently advertised by this persona."""
        return self.awareness.slash_commands

    ################################################
    # reporting session information (called by the persona itself)
    ################################################
    # A persona calls these `report_*` methods to publish its own session state
    # over awareness. They are the counterpart to the `get_*` readers, and are
    # meant to be called by the persona (and its collaborators), not by
    # consumers of the persona. Assigning an `self.awareness` property publishes
    # it, so these just forward — except `report_usage`, which owns the merge.
    @mark_subclass_api
    def report_model_configuration(self, model: ModelConfiguration) -> None:
        """
        Publish the persona's model configuration (current model, model options,
        and model settings). A persona calls this once it knows its models (e.g.
        an ACP persona on session create/load).
        """
        self.awareness.model = model

    @mark_subclass_api
    def report_settings_configuration(
        self, settings: list[SettingConfiguration]
    ) -> None:
        """Publish the persona's general (non-model) settings configuration."""
        self.awareness.settings = settings

    @mark_subclass_api
    def report_usage(self, usage: Usage, *, append: bool = False) -> None:
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
        current = self.awareness.usage
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
        self.awareness.usage = current

    @mark_subclass_api
    def report_slash_commands(self, commands: list[CommandOption]) -> None:
        """Publish the advertised slash commands."""
        self.awareness.slash_commands = commands

    ################################################
    # applying model & settings (overridden by configurable personas)
    ################################################
    # These `update_*` methods apply a user's specification: they tell the
    # persona's backend to switch, and do NOT touch awareness (`BasePersona`
    # records the new current values and rebroadcasts — see `apply_*_spec`
    # below). A subclass overrides an `update_*` only if it also reports the
    # corresponding configuration (via `report_model_configuration` /
    # `report_settings_configuration`); a persona whose model or settings aren't
    # configurable leaves the default no-op in place. There is deliberately no
    # `update_*` for usage or slash commands — a user can't set those.
    @mark_optional
    async def update_model(self, model_id: str) -> None:
        """Switch this persona to the model identified by `model_id`."""

    @mark_optional
    async def update_model_settings(self, settings: dict[str, str | None]) -> None:
        """Apply the given model settings (e.g. context size), keyed by ID."""

    @mark_optional
    async def update_settings(self, settings: dict[str, str | None]) -> None:
        """
        Apply the given general settings (e.g. mode, effort level), keyed by ID.
        Separate from model settings.
        """

    ################################################
    # applying a message's model & settings specification
    ################################################
    @mark_consumer_api
    async def apply_model_spec(self, spec: ModelSpec) -> None:
        """
        Apply a user's specified model and model settings, then record the new
        current values on the awareness slot (which rebroadcasts).

        A None model ID or a None setting value means "use the persona's current
        value", so it is skipped. The persona's backend is only asked to switch
        (and the change only broadcast) when a specified value actually differs
        from the current one.
        """
        model_changed = spec.id is not None and spec.id != self.get_model()
        if model_changed:
            await self.update_model(spec.id)

        current = self.get_model_settings()
        changed = {
            key: value
            for key, value in spec.settings.items()
            if value is not None and value != current.get(key)
        }
        if changed:
            await self.update_model_settings(changed)

        if not (model_changed or changed):
            return

        # Record the new current values and republish (assigning the property
        # rebroadcasts). Read a fresh copy, mutate, write it back.
        model = self.awareness.model
        if model_changed:
            model.current = spec.id
        for setting in model.settings:
            if setting.id in changed:
                setting.current = changed[setting.id]
        self.awareness.model = model

    @mark_consumer_api
    async def apply_settings_spec(self, spec: dict[str, str | None]) -> None:
        """
        Apply a user's specified general settings, then record the new current
        values on the awareness slot (which rebroadcasts).

        A None value for a setting means "use the persona's current value", so
        it is skipped. The persona's backend is only asked to switch (and the
        change only broadcast) when a specified value actually differs from the
        current one.
        """
        current = self.get_settings()
        changed = {
            key: value
            for key, value in spec.items()
            if value is not None and value != current.get(key)
        }
        if not changed:
            return

        await self.update_settings(changed)
        settings = self.awareness.settings
        for setting in settings:
            if setting.id in changed:
                setting.current = changed[setting.id]
        self.awareness.settings = settings

    @mark_consumer_api
    async def apply_specs_in_message(self, message: Message) -> None:
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

    @mark_subclass_api
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

    @mark_subclass_api
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

    @mark_recommended
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