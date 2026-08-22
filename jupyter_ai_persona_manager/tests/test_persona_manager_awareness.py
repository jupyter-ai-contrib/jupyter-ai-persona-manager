"""
Tests for the PersonaManager persona-list publishing (now over Jupyter Events)
and for spec application happening before processing in the message-dispatch
path.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from jupyter_events import EventLogger
from jupyterlab_chat.models import Message

from jupyter_ai_persona_manager.persona_events import (
    PERSONAS_EVENT_SCHEMA_ID,
    PersonaManagerSessionState,
    register_persona_event_schemas,
)
from jupyter_ai_persona_manager.persona_manager import (
    PersonaManager,
    _safe_process,
)


def _mock_persona(id: str, name: str, avatar_url: str = "/a"):
    persona = MagicMock()
    persona.id = id
    persona.name = name
    user = MagicMock()
    user.avatar_url = avatar_url
    persona.as_user.return_value = user
    return persona


def _manager(personas, event_logger):
    """A PersonaManager wired to an events-based PersonaManagerSessionState."""
    pm = PersonaManager.__new__(PersonaManager)
    pm._personas = personas
    pm.log = logging.getLogger("test-pm-events")
    # `chat_path` is a live property backed by the chat model; the manager reads
    # the chat id (for event scoping) and path (for room-event matching) from it.
    pm.chat = MagicMock()
    pm.chat.get_id.return_value = "chat-1"
    pm.chat.get_path.return_value = "file-id.chat"
    pm.state = PersonaManagerSessionState(
        event_logger=event_logger, chat_id=pm.chat.get_id(), log=pm.log
    )
    return pm


class TestPersonaListEvents:
    def test_publishes_every_persona(self):
        async def run():
            captured: list = []
            logger = EventLogger()
            register_persona_event_schemas(logger)

            async def listener(logger, schema_id, data):
                if schema_id == PERSONAS_EVENT_SCHEMA_ID:
                    captured.append(data)

            logger.add_listener(schema_id=PERSONAS_EVENT_SCHEMA_ID, listener=listener)

            pm = _manager(
                {
                    "p1": _mock_persona("p1", "One", "/one"),
                    "p2": _mock_persona("p2", "Two", "/two"),
                },
                logger,
            )
            pm._publish_persona_list()
            await asyncio.sleep(0.1)

            assert captured, "no personas event emitted"
            data = captured[-1]
            assert data["chat_id"] == "chat-1"
            by_id = {p["id"]: p for p in data["personas"]}
            assert by_id["p1"]["name"] == "One"
            assert by_id["p1"]["avatar_url"] == "/one"
            assert set(by_id) == {"p1", "p2"}

        asyncio.run(run())

    def test_empty_when_no_personas(self):
        async def run():
            captured: list = []
            logger = EventLogger()
            register_persona_event_schemas(logger)

            async def listener(logger, schema_id, data):
                captured.append(data)

            logger.add_listener(schema_id=PERSONAS_EVENT_SCHEMA_ID, listener=listener)

            pm = _manager({}, logger)
            pm._publish_persona_list()
            await asyncio.sleep(0.1)

            assert captured[-1]["personas"] == []

        asyncio.run(run())

    def test_catchup_reemits_on_client_connected(self):
        async def run():
            logger = EventLogger()
            register_persona_event_schemas(logger)
            personas = {"p1": _mock_persona("p1", "One")}
            pm = _manager(personas, logger)

            # A client connecting to this chat triggers a full re-publish.
            await pm._on_chat_event(
                None,
                "https://schema.jupyter.org/jupyterlab_chat/room/v1",
                {"action": "client_connected", "path": pm.chat_path},
            )
            personas["p1"].state.publish.assert_called()

            # An event for a different chat is ignored.
            personas["p1"].state.publish.reset_mock()
            await pm._on_chat_event(
                None,
                "https://schema.jupyter.org/jupyterlab_chat/room/v1",
                {"action": "client_connected", "path": "other.chat"},
            )
            personas["p1"].state.publish.assert_not_called()

        asyncio.run(run())


class TestSafeProcessAppliesSpecsFirst:
    """apply_specs_in_message must run before process_message."""

    @pytest.mark.asyncio
    async def test_specs_applied_before_processing(self):
        order = []
        persona = MagicMock()
        persona.name = "P"
        persona.log = MagicMock()
        persona.apply_specs_in_message = AsyncMock(
            side_effect=lambda m: order.append("apply")
        )
        persona.process_message = AsyncMock(
            side_effect=lambda m: order.append("process")
        )
        persona.handle_uncaught_exception = AsyncMock()

        await _safe_process(persona, MagicMock(spec=Message))

        assert order == ["apply", "process"]

    @pytest.mark.asyncio
    async def test_processing_error_routed_to_handler(self):
        persona = MagicMock()
        persona.name = "P"
        persona.log = MagicMock()
        persona.apply_specs_in_message = AsyncMock()
        exc = RuntimeError("boom")
        persona.process_message = AsyncMock(side_effect=exc)
        persona.handle_uncaught_exception = AsyncMock()

        await _safe_process(persona, MagicMock(spec=Message))

        persona.handle_uncaught_exception.assert_awaited_once_with(exc)

    @pytest.mark.asyncio
    async def test_spec_error_routed_to_handler(self):
        # A failure while applying specs is also delivered to the user rather
        # than crashing the dispatch task.
        persona = MagicMock()
        persona.name = "P"
        persona.log = MagicMock()
        exc = RuntimeError("bad spec")
        persona.apply_specs_in_message = AsyncMock(side_effect=exc)
        persona.process_message = AsyncMock()
        persona.handle_uncaught_exception = AsyncMock()

        await _safe_process(persona, MagicMock(spec=Message))

        persona.process_message.assert_not_awaited()
        persona.handle_uncaught_exception.assert_awaited_once_with(exc)
