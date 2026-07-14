import asyncio
import random
from contextlib import contextmanager
from dataclasses import asdict
from logging import Logger
from typing import TYPE_CHECKING, Any

from jupyterlab_chat.models import User
from pycrdt import Awareness

from .awareness_models import (
    CommandOption,
    ModelConfiguration,
    PersonaOption,
    SettingConfiguration,
    Usage,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from jupyterlab_chat.ychat import YChat


PERSONA_MANAGER_AWARENESS_CLIENT_ID = 7133713371337
"""
The fixed, hardcoded Yjs client ID under which every `PersonaManager` publishes
its awareness state (the persona list). A chosen 53-bit constant so it survives
reconnects and the browser can locate the manager's state in the awareness map
without enumerating clients. A single readiness REST endpoint hands this ID to
the browser once the manager and its personas are registered.
"""


class ScopedAwareness:
    """
    Base class that writes awareness local state under a *custom client ID*,
    working around the fact that `pycrdt.Awareness` provides no native way to
    set state for more than one client ID from a single process.

    It accepts a `YChat` and behaves like a small `pycrdt.Awareness` facade
    scoped to one client ID: every read/write temporarily swaps
    `ydoc.awareness.client_id` to this instance's ID (via `as_custom_client()`),
    so multiple `ScopedAwareness` instances can each own a distinct slot in the
    same shared awareness map.

    - Pass an explicit `client_id` to reserve a fixed, well-known slot (e.g. the
      `PersonaManager`); omit it for a random per-instance ID (e.g. a persona).
    - Pass a `User` to register it in the slot on init.
    - A heartbeat task keeps the slot alive past the 30s awareness timeout.

    Subclasses add typed getter/setter properties for the specific fields they
    own, so consumers work with models instead of raw dict fields.
    """

    awareness: Awareness
    log: Logger
    user: User | None

    _original_client_id: int
    _custom_client_id: int
    _heartbeat_task: asyncio.Task | None

    def __init__(
        self,
        *,
        ychat: "YChat",
        log: Logger,
        user: User | None = None,
        client_id: int | None = None,
    ):
        # Bind instance attributes
        self.log = log
        self.user = user

        # Bind awareness object if available, initialize it otherwise
        if ychat.awareness:
            self.awareness = ychat.awareness
        else:
            self.awareness = Awareness(ydoc=ychat._ydoc)

        # Initialize a custom client ID & save the original client ID. A fixed
        # `client_id` reserves a well-known slot; otherwise generate a random one.
        self._original_client_id = self.awareness.client_id
        self._custom_client_id = (
            client_id if client_id is not None else random.getrandbits(32)
        )

        # Initialize local awareness state using the custom client ID
        self.set_local_state({})
        if self.user:
            self._register_user()

        # Start the awareness heartbeat task. It needs a running event loop,
        # which is always the case in production (the server runs inside one).
        # When constructed without a loop (e.g. a synchronous unit test), skip
        # the heartbeat rather than fail — the awareness state is still fully
        # usable, it just isn't kept alive past the outdated timeout.
        self._heartbeat_task = None
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self.log.debug(
                "No running event loop; awareness heartbeat not started."
            )
        else:
            self._heartbeat_task = asyncio.create_task(self._start_heartbeat())

    @property
    def client_id(self) -> int:
        """
        The Yjs client ID this instance writes its awareness state under. This
        is the fixed ID passed to the constructor, or the random one generated
        when none was passed.
        """
        return self._custom_client_id

    @contextmanager
    def as_custom_client(self) -> "Iterator[None]":
        """
        Context manager that automatically:

        - Sets the awareness client ID to the custom client ID created by this
        instance upon entering.

        - Resets the awareness client ID back to the original client ID upon
        exiting.
        """
        self.awareness.client_id = self._custom_client_id
        try:
            yield
        finally:
            self.awareness.client_id = self._original_client_id

    @property
    def outdated_timeout(self) -> int:
        """
        Returns the outdated timeout of the document awareness, in milliseconds.
        The timeout value should be 30000 milliseconds (30 seconds), according to the
        default value in `y-protocols.awareness` and `pycrdt.Awareness`.
        - https://github.com/yjs/y-protocols/blob/2d8cd5c06b3925fbf9b5215dc341f8096a0a8d5c/awareness.js#L13
        - https://github.com/y-crdt/pycrdt/blob/e269a3e63ad7986a3349e2d2bc7bd5f0dfca9c79/python/pycrdt/_awareness.py#L23
        """
        return self.awareness._outdated_timeout

    def _register_user(self):
        if not self.user:
            return

        with self.as_custom_client():
            self.awareness.set_local_state_field("user", asdict(self.user))

    def get_local_state(self) -> dict[str, Any] | None:
        """
        Returns the local state of the awareness instance.
        """
        with self.as_custom_client():
            return self.awareness.get_local_state()

    def set_local_state(self, state: dict[str, Any] | None) -> None:
        """
        Sets the local state of this instance in the awareness map, indexed by
        this instance's custom client ID.

        Passing `state=None` deletes the local state indexed by this instance's
        custom client ID.
        """
        with self.as_custom_client():
            self.awareness.set_local_state(state)

    def set_local_state_field(self, field: str, value: Any) -> None:
        """
        Sets a specific field in the local state of the awareness instance.
        """
        with self.as_custom_client():
            self.awareness.set_local_state_field(field, value)

    def get_local_state_field(self, field: str, default: Any = None) -> Any:
        """
        Returns a specific field of this instance's local state, or `default` if
        the field (or the local state) is not set. The read counterpart to
        `set_local_state_field`, which `pycrdt.Awareness` itself does not provide.
        """
        return (self.get_local_state() or {}).get(field, default)

    async def _start_heartbeat(self):
        """
        Background task that updates this instance's local state every
        `0.8 * self.outdated_timeout` milliseconds. `pycrdt` and `yjs`
        automatically disconnect clients if they do not make updates in
        a long time (default: 30000 ms). This task keeps personas alive
        after 30 seconds of no usage in each chat session.
        """
        while True:
            await asyncio.sleep(0.8 * self.outdated_timeout / 1000)
            local_state = self.get_local_state() or {}
            self.set_local_state(local_state)

    def shutdown(self) -> None:
        """
        Stops this instance's background tasks and removes this instance's
        custom client ID from the awareness map.
        """
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
        self.set_local_state(None)


class PersonaAwareness(ScopedAwareness):
    """
    A persona's awareness slot: the session information a persona broadcasts to
    every client (its model configuration, general settings, usage, slash
    commands, and whether it is currently writing).

    Each field is a typed property backed directly by the awareness slot — the
    getter deserializes it from the slot, the setter serializes it in and
    triggers a rebroadcast. The slot itself is the persona's awareness state, so
    there is no separate in-memory copy to keep in sync.
    """

    def __init__(self, *, ychat: "YChat", log: Logger, user: User, id: str):
        super().__init__(ychat=ychat, log=log, user=user)
        # Publish the default state so the slot is a complete persona state from
        # the start. Subclasses/callers fill in real values via the properties
        # once they know them (e.g. an ACP persona on session create/load).
        self.set_local_state_field("id", id)
        self.model = ModelConfiguration()
        self.settings = []
        self.usage = Usage()
        self.slash_commands = []
        self.is_writing = False

    @property
    def id(self) -> str:
        """The persona ID (stable for the persona's lifetime)."""
        return self.get_local_state_field("id", "")

    @property
    def model(self) -> ModelConfiguration:
        """The persona's model configuration (current model, options, model settings)."""
        data = self.get_local_state_field("model")
        return ModelConfiguration(**data) if data else ModelConfiguration()

    @model.setter
    def model(self, model: ModelConfiguration) -> None:
        self.set_local_state_field("model", model.model_dump())

    @property
    def settings(self) -> list[SettingConfiguration]:
        """The persona's general (non-model) setting configurations."""
        return [SettingConfiguration(**s) for s in self.get_local_state_field("settings", [])]

    @settings.setter
    def settings(self, settings: list[SettingConfiguration]) -> None:
        self.set_local_state_field("settings", [s.model_dump() for s in settings])

    @property
    def usage(self) -> Usage:
        """The token and cost usage the persona reports for the session."""
        data = self.get_local_state_field("usage")
        return Usage(**data) if data else Usage()

    @usage.setter
    def usage(self, usage: Usage) -> None:
        self.set_local_state_field("usage", usage.model_dump())

    @property
    def slash_commands(self) -> list[CommandOption]:
        """The slash commands the persona advertises."""
        return [CommandOption(**c) for c in self.get_local_state_field("slash_commands", [])]

    @slash_commands.setter
    def slash_commands(self, commands: list[CommandOption]) -> None:
        self.set_local_state_field(
            "slash_commands", [c.model_dump() for c in commands]
        )

    @property
    def is_writing(self) -> bool | str:
        """
        Whether the persona is currently writing a reply: `False` when idle, or
        the ID of the message being written while streaming (jupyter-chat reads
        this to render the typing indicator and enable the stop button).

        Stored under the `isWriting` awareness key; assigning this property and
        calling `set_local_state_field("isWriting", ...)` are equivalent.
        """
        return self.get_local_state_field("isWriting", False)

    @is_writing.setter
    def is_writing(self, value: bool | str) -> None:
        self.set_local_state_field("isWriting", value)


class PersonaManagerAwareness(ScopedAwareness):
    """
    The `PersonaManager`'s awareness slot: the list of personas in the chat.

    Registered under the fixed `PERSONA_MANAGER_AWARENESS_CLIENT_ID` so the
    browser can find it across reconnects. The `personas` property is backed
    directly by the awareness slot.
    """

    def __init__(self, *, ychat: "YChat", log: Logger):
        super().__init__(
            ychat=ychat,
            log=log,
            user=None,
            client_id=PERSONA_MANAGER_AWARENESS_CLIENT_ID,
        )
        self.personas = []

    @property
    def personas(self) -> list[PersonaOption]:
        """The personas available in this chat."""
        return [PersonaOption(**p) for p in self.get_local_state_field("personas", [])]

    @personas.setter
    def personas(self, personas: list[PersonaOption]) -> None:
        self.set_local_state_field("personas", [p.model_dump() for p in personas])
