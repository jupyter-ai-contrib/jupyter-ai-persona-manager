"""Jupyter Events-based publishing of persona session information.

This replaces the previous Yjs-awareness mechanism (``persona_awareness.py``).
Instead of writing awareness slots, the persona manager and each persona emit
Jupyter Events:

- ``jupyter_ai_persona_manager/personas/v1`` -- the list of personas in a chat
  (published by the manager).
- ``jupyter_ai_persona_manager/persona_state/v1`` -- a single persona's session
  state: model configuration, general settings, usage, and slash commands
  (published by each persona on change).

A ``persona_state`` event carries the chat id, the persona id, and *only the
attribute(s) that changed* -- setting ``usage`` emits an event with just
``usage``, leaving ``model``/``settings``/``slash_commands`` absent so they are
not needlessly re-broadcast (usage updates on every streamed chunk). Consumers
merge each event into their view, replacing only the attributes present. A bare
:meth:`PersonaSessionState.publish` (used for catch-up) is the one exception: it
re-emits the full state at once.

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
    "description": "A single persona's model, settings, usage, and slash commands.",
    "type": "object",
    "required": ["chat_id", "persona_id"],
    "properties": {
        "chat_id": {"type": "string", "description": "The chat's stable id (used by the frontend to scope events to a chat)."},
        "persona_id": {"type": "string", "description": "The persona's stable id."},
        "model": {"type": "object"},
        "settings": {"type": "array", "items": {"type": "object"}},
        "usage": {"type": "object"},
        "slash_commands": {"type": "array", "items": {"type": "object"}},
    },
    "additionalProperties": False,
}


def register_persona_event_schemas(event_logger: "EventLogger") -> None:
    """Register the persona event schemas on ``event_logger`` (idempotent)."""
    for schema in (PERSONAS_EVENT_SCHEMA, PERSONA_STATE_EVENT_SCHEMA):
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
    memory and emit a ``persona_state`` event on every change, so consumers see
    live updates; :meth:`publish` re-emits the full state for catch-up.
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

    @property
    def id(self) -> str:
        return self._persona_id

    @property
    def model(self) -> ModelConfiguration:
        return self._model

    @model.setter
    def model(self, model: ModelConfiguration) -> None:
        self._model = model
        self._emit(model=model.model_dump())

    @property
    def settings(self) -> list[SettingConfiguration]:
        return self._settings

    @settings.setter
    def settings(self, settings: list[SettingConfiguration]) -> None:
        self._settings = settings
        self._emit(settings=[s.model_dump() for s in settings])

    @property
    def usage(self) -> Usage:
        return self._usage

    @usage.setter
    def usage(self, usage: Usage) -> None:
        self._usage = usage
        self._emit(usage=usage.model_dump())

    @property
    def slash_commands(self) -> list[CommandOption]:
        return self._slash_commands

    @slash_commands.setter
    def slash_commands(self, commands: list[CommandOption]) -> None:
        self._slash_commands = commands
        self._emit(slash_commands=[c.model_dump() for c in commands])

    def to_data(self) -> dict[str, Any]:
        return {
            "chat_id": self._chat_id,
            "persona_id": self._persona_id,
            "model": self._model.model_dump(),
            "settings": [s.model_dump() for s in self._settings],
            "usage": self._usage.model_dump(),
            "slash_commands": [c.model_dump() for c in self._slash_commands],
        }

    def _emit(self, **attributes: Any) -> None:
        """Emit a ``persona_state`` event carrying only the given attribute(s),
        alongside the always-present ``chat_id`` and ``persona_id``.

        Each attribute setter publishes only what changed, so updating one field
        (e.g. usage, which changes on every streamed chunk) does not re-broadcast
        the others. Consumers merge each event onto their existing view.
        """
        if self._event_logger is None:
            return
        try:
            self._event_logger.emit(
                schema_id=PERSONA_STATE_EVENT_SCHEMA_ID,
                data={
                    "chat_id": self._chat_id,
                    "persona_id": self._persona_id,
                    **attributes,
                },
            )
        except Exception:  # pragma: no cover - defensive
            self._log.exception("Failed to emit persona state event")

    def publish(self) -> None:
        """(Re-)emit the persona's full current state, all attributes at once.
        Used for catch-up when a client connects."""
        if self._event_logger is None:
            return
        try:
            self._event_logger.emit(
                schema_id=PERSONA_STATE_EVENT_SCHEMA_ID, data=self.to_data()
            )
        except Exception:  # pragma: no cover - defensive
            self._log.exception("Failed to emit persona state event")

    def shutdown(self) -> None:
        """Symmetry with the old awareness slot's shutdown. Events are
        fire-and-forget, so there is no retained slot to clear; this is a no-op
        kept so callers need not special-case it."""
        return None
