"""Jupyter Events-based publishing of persona session information.

This replaces the previous Yjs-awareness mechanism (``persona_awareness.py``).
Instead of writing awareness slots, the persona manager and each persona emit
Jupyter Events:

- ``jupyter_ai_persona_manager/personas/v1`` -- the list of personas in a chat
  (published by the manager).
- ``jupyter_ai_persona_manager/persona_state/v1`` -- a single persona's session
  state: model configuration, general settings, usage, and slash commands
  (published by each persona on change).

Events are fire-and-forget, so a client that connects after an event was emitted
would miss it. Catch-up is handled by re-emitting the current state when a client
connects to the chat (see ``PersonaManager`` and the ``client_connected`` action
on jupyterlab_chat's ``room/v1`` event bus). The current values are kept in memory
here so they can be re-emitted at any time.

Each event carries the chat's stable id (``chat.get_id()``) so the frontend can
route it to the right chat via its chat model/context ``id``.
"""
from __future__ import annotations

from logging import Logger
from typing import TYPE_CHECKING, Any, Optional

from .awareness_models import (
    CommandOption,
    ModelConfiguration,
    PersonaOption,
    SettingConfiguration,
    Usage,
)

if TYPE_CHECKING:
    from jupyter_events import EventLogger


PERSONAS_EVENT_SCHEMA_ID = (
    "https://schema.jupyter.org/jupyter_ai_persona_manager/personas/v1"
)
PERSONA_STATE_EVENT_SCHEMA_ID = (
    "https://schema.jupyter.org/jupyter_ai_persona_manager/persona_state/v1"
)
PERSONA_SELECTED_EVENT_SCHEMA_ID = (
    "https://schema.jupyter.org/jupyter_ai_persona_manager/persona_selected/v1"
)

PERSONAS_EVENT_SCHEMA = {
    "$id": PERSONAS_EVENT_SCHEMA_ID,
    "version": "1",
    "title": "Persona list",
    "personal-data": True,
    "description": "The list of personas available in a chat.",
    "type": "object",
    "required": ["chat_id", "personas"],
    "properties": {
        "chat_id": {"type": "string", "description": "The chat's stable id (used by the frontend to scope events to a chat)."},
        "personas": {
            "type": "array",
            "items": {"type": "object"},
            "description": "Serialized PersonaOption objects.",
        },
    },
    "additionalProperties": False,
}

PERSONA_STATE_EVENT_SCHEMA = {
    "$id": PERSONA_STATE_EVENT_SCHEMA_ID,
    "version": "1",
    "title": "Persona session state",
    "personal-data": True,
    "description": (
        "A single persona's model, settings, usage, slash commands, and "
        "whether it is currently processing a message."
    ),
    "type": "object",
    "required": ["chat_id", "persona_id"],
    "properties": {
        "chat_id": {"type": "string", "description": "The chat's stable id (used by the frontend to scope events to a chat)."},
        "persona_id": {"type": "string", "description": "The persona's stable id."},
        "model": {"type": "object"},
        "settings": {"type": "array", "items": {"type": "object"}},
        "usage": {"type": "object"},
        "slash_commands": {"type": "array", "items": {"type": "object"}},
        "processing": {
            "type": "boolean",
            "description": (
                "Whether the persona is currently processing a message. The "
                "frontend uses this to enable the stop button."
            ),
        },
    },
    "additionalProperties": False,
}

PERSONA_SELECTED_EVENT_SCHEMA = {
    "$id": PERSONA_SELECTED_EVENT_SCHEMA_ID,
    "version": "1",
    "title": "Persona selected",
    "personal-data": True,
    "description": (
        "Emitted by a client when a persona is selected in a chat, so the "
        "server can prepare it eagerly."
    ),
    "type": "object",
    "required": ["chat_id", "persona_id"],
    "properties": {
        "chat_id": {"type": "string", "description": "The chat's stable id the selection happened in."},
        "persona_id": {"type": "string", "description": "The stable id of the selected persona."},
    },
    "additionalProperties": False,
}


def register_persona_event_schemas(event_logger: "EventLogger") -> None:
    """Register the persona event schemas on ``event_logger`` (idempotent)."""
    for schema in (
        PERSONAS_EVENT_SCHEMA,
        PERSONA_STATE_EVENT_SCHEMA,
        PERSONA_SELECTED_EVENT_SCHEMA,
    ):
        try:
            event_logger.register_event_schema(schema)
        except Exception:  # pragma: no cover - already registered / defensive
            pass


class PersonaManagerSessionState:
    """The persona-manager's session state that is published to clients via the
    events API: the chat's persona list.

    Replaces ``PersonaManagerAwareness``. Holds the last-published list in memory
    so it can be re-emitted for catch-up when a new client connects.
    """

    def __init__(
        self,
        *,
        event_logger: Optional["EventLogger"],
        chat_id: str,
        log: Logger,
    ):
        self._event_logger = event_logger
        self._chat_id = chat_id
        self._log = log
        self._personas: list[PersonaOption] = []

    @property
    def personas(self) -> list[PersonaOption]:
        return list(self._personas)

    @personas.setter
    def personas(self, personas: list[PersonaOption]) -> None:
        self._personas = list(personas)
        self.publish()

    def publish(self) -> None:
        """(Re-)emit the current persona list. Used both on change and for
        catch-up when a client connects."""
        if self._event_logger is None:
            return
        try:
            self._event_logger.emit(
                schema_id=PERSONAS_EVENT_SCHEMA_ID,
                data={
                    "chat_id": self._chat_id,
                    "personas": [p.model_dump() for p in self._personas],
                },
            )
        except Exception:  # pragma: no cover - defensive
            self._log.exception("Failed to emit persona list event")


class PersonaSessionState:
    """A persona's session state that is published to clients via the events API:
    its model configuration, general settings, usage, and slash commands.

    Replaces ``PersonaAwareness``. The typed properties keep the last value in
    memory and emit a ``persona_state`` event carrying only the changed
    attribute, so consumers merge live updates onto their existing state;
    :meth:`publish` emits the full state for catch-up.
    """

    def __init__(
        self,
        *,
        event_logger: Optional["EventLogger"],
        persona_id: str,
        chat_id: str,
        log: Logger,
    ):
        self._event_logger = event_logger
        self._chat_id = chat_id
        self._persona_id = persona_id
        self._log = log
        self._model = ModelConfiguration()
        self._settings: list[SettingConfiguration] = []
        self._usage = Usage()
        self._slash_commands: list[CommandOption] = []
        self._processing = False

    @property
    def id(self) -> str:
        return self._persona_id

    @property
    def model(self) -> ModelConfiguration:
        return self._model

    @model.setter
    def model(self, model: ModelConfiguration) -> None:
        self._model = model
        self._emit({"model": model.model_dump()})

    @property
    def settings(self) -> list[SettingConfiguration]:
        return self._settings

    @settings.setter
    def settings(self, settings: list[SettingConfiguration]) -> None:
        self._settings = settings
        self._emit({"settings": [s.model_dump() for s in settings]})

    @property
    def usage(self) -> Usage:
        return self._usage

    @usage.setter
    def usage(self, usage: Usage) -> None:
        self._usage = usage
        self._emit({"usage": usage.model_dump()})

    @property
    def slash_commands(self) -> list[CommandOption]:
        return self._slash_commands

    @slash_commands.setter
    def slash_commands(self, commands: list[CommandOption]) -> None:
        self._slash_commands = commands
        self._emit({"slash_commands": [c.model_dump() for c in commands]})

    @property
    def processing(self) -> bool:
        return self._processing

    @processing.setter
    def processing(self, processing: bool) -> None:
        # Only publish on an actual transition, so repeated no-op assignments
        # (e.g. from serialized message processing) don't emit redundant events.
        if self._processing == processing:
            return
        self._processing = processing
        self._emit({"processing": processing})

    def to_data(self) -> dict[str, Any]:
        """The persona's full current state, including the routing ids. Used by
        :meth:`publish` for the catch-up snapshot."""
        return {
            "chat_id": self._chat_id,
            "persona_id": self._persona_id,
            "model": self._model.model_dump(),
            "settings": [s.model_dump() for s in self._settings],
            "usage": self._usage.model_dump(),
            "slash_commands": [c.model_dump() for c in self._slash_commands],
            "processing": self._processing,
        }

    def _emit(self, fields: dict[str, Any]) -> None:
        """Emit a ``persona_state`` event carrying ``fields`` plus the routing
        ids (``chat_id``/``persona_id``).

        Field setters emit only the attribute that changed; the frontend merges
        each event into the persona's existing state, treating absent keys as
        unchanged. This avoids re-sending the whole snapshot (model, settings,
        slash commands) on every usage tick. :meth:`publish` sends the full
        snapshot for catch-up when a client connects.
        """
        if self._event_logger is None:
            return
        try:
            self._event_logger.emit(
                schema_id=PERSONA_STATE_EVENT_SCHEMA_ID,
                data={
                    "chat_id": self._chat_id,
                    "persona_id": self._persona_id,
                    **fields,
                },
            )
        except Exception:  # pragma: no cover - defensive
            self._log.exception("Failed to emit persona state event")

    def publish(self) -> None:
        """Emit the persona's full current state. Used for catch-up when a
        client connects; individual field changes emit only the changed field
        (see :meth:`_emit`)."""
        data = self.to_data()
        # `_emit` re-adds the routing ids, so drop them from the field set here.
        data.pop("chat_id", None)
        data.pop("persona_id", None)
        self._emit(data)

    def shutdown(self) -> None:
        """Symmetry with the old awareness slot's shutdown. Events are
        fire-and-forget, so there is no retained slot to clear; this is a no-op
        kept so callers need not special-case it."""
        return None
