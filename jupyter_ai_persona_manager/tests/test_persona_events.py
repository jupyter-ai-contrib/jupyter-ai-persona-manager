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


def test_setting_fields_emits_state_events():
    async def run():
        logger, captured = _logger_and_capture()
        state = PersonaSessionState(
            event_logger=logger,
            room_id="room:chat:x",
            persona_id="jupyternaut",
            log=logging.getLogger("t"),
        )
        state.model = ModelConfiguration(current="gpt-9")
        state.usage = Usage(input_tokens=42)
        await asyncio.sleep(0.1)

        assert len(captured) == 2
        last = captured[-1]
        assert last["room_id"] == "room:chat:x"
        assert last["persona_id"] == "jupyternaut"
        assert last["model"]["current"] == "gpt-9"
        assert last["usage"]["input_tokens"] == 42

    asyncio.run(run())


def test_publish_reemits_current_state_for_catchup():
    async def run():
        logger, captured = _logger_and_capture()
        state = PersonaSessionState(
            event_logger=logger,
            room_id="room:chat:x",
            persona_id="p1",
            log=logging.getLogger("t"),
        )
        state.model = ModelConfiguration(current="m1")
        await asyncio.sleep(0.1)
        captured.clear()

        # A bare publish() (used for catch-up on client_connected) re-emits the
        # full current state without any change.
        state.publish()
        await asyncio.sleep(0.1)

        assert len(captured) == 1
        assert captured[0]["model"]["current"] == "m1"

    asyncio.run(run())


def test_no_event_logger_is_noop():
    state = PersonaSessionState(
        event_logger=None,
        room_id="room:chat:x",
        persona_id="p1",
        log=logging.getLogger("t"),
    )
    # Should not raise without an event logger.
    state.model = ModelConfiguration(current="m1")
    state.publish()
    assert state.model.current == "m1"
