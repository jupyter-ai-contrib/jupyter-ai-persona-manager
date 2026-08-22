"""Unit tests for the events-based persona session state (persona_events.py)."""

import asyncio
import logging

from jupyter_events import EventLogger

from jupyter_ai_persona_manager.awareness_models import (
    ModelConfiguration,
    Usage,
)
from jupyter_ai_persona_manager.persona_events import (
    PERSONA_STATE_EVENT_SCHEMA_ID,
    PersonaSessionState,
    register_persona_event_schemas,
)


def _logger_and_capture():
    logger = EventLogger()
    register_persona_event_schemas(logger)
    captured: list = []

    async def listener(logger, schema_id, data):
        if schema_id == PERSONA_STATE_EVENT_SCHEMA_ID:
            captured.append(data)

    logger.add_listener(schema_id=PERSONA_STATE_EVENT_SCHEMA_ID, listener=listener)
    return logger, captured


def test_setting_a_field_emits_only_that_attribute():
    async def run():
        logger, captured = _logger_and_capture()
        state = PersonaSessionState(
            event_logger=logger,
            chat_id="chat.chat",
            persona_id="jupyternaut",
            log=logging.getLogger("t"),
        )
        state.model = ModelConfiguration(current="gpt-9")
        state.usage = Usage(input_tokens=42)
        await asyncio.sleep(0.1)

        assert len(captured) == 2
        # chat_id and persona_id ride every event.
        for event in captured:
            assert event["chat_id"] == "chat.chat"
            assert event["persona_id"] == "jupyternaut"

        # The model event carries only `model`; the usage event only `usage`.
        model_event, usage_event = captured
        assert model_event["model"]["current"] == "gpt-9"
        assert "usage" not in model_event
        assert "settings" not in model_event
        assert "slash_commands" not in model_event

        assert usage_event["usage"]["input_tokens"] == 42
        assert "model" not in usage_event

    asyncio.run(run())


def test_publish_reemits_full_state_for_catchup():
    async def run():
        logger, captured = _logger_and_capture()
        state = PersonaSessionState(
            event_logger=logger,
            chat_id="chat.chat",
            persona_id="p1",
            log=logging.getLogger("t"),
        )
        state.model = ModelConfiguration(current="m1")
        state.usage = Usage(input_tokens=7)
        await asyncio.sleep(0.1)
        captured.clear()

        # A bare publish() (used for catch-up on client_connected) re-emits the
        # full current state -- all attributes at once.
        state.publish()
        await asyncio.sleep(0.1)

        assert len(captured) == 1
        full = captured[0]
        assert full["model"]["current"] == "m1"
        assert full["usage"]["input_tokens"] == 7
        assert "settings" in full
        assert "slash_commands" in full

    asyncio.run(run())


def test_no_event_logger_is_noop():
    state = PersonaSessionState(
        event_logger=None,
        chat_id="chat.chat",
        persona_id="p1",
        log=logging.getLogger("t"),
    )
    # Should not raise without an event logger.
    state.model = ModelConfiguration(current="m1")
    state.publish()
    assert state.model.current == "m1"
