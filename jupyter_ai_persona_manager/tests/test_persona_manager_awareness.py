"""
Tests for the PersonaManager awareness broadcast (the persona list published
under the fixed client ID) and for metadata application happening before
processing in the message-dispatch path.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from jupyterlab_chat.models import Message

from jupyter_ai_persona_manager.persona_manager import (
    PERSONA_MANAGER_AWARENESS_CLIENT_ID,
    PersonaManager,
    _safe_process,
)


def _mock_persona(id: str, name: str, client_id: int, avatar_url: str = "/a"):
    persona = MagicMock()
    persona.id = id
    persona.name = name
    persona.awareness.client_id = client_id
    user = MagicMock()
    user.avatar_url = avatar_url
    persona.as_user.return_value = user
    return persona


def _manager(personas):
    """A PersonaManager with only the state the awareness broadcast reads."""
    pm = PersonaManager.__new__(PersonaManager)
    pm._personas = personas
    pm.log = logging.getLogger("test-pm-awareness")
    pm._awareness = MagicMock()
    return pm


class TestBuildAwarenessState:
    def test_lists_every_persona_with_yjs_client_id(self):
        pm = _manager(
            {
                "p1": _mock_persona("p1", "One", 111, "/one"),
                "p2": _mock_persona("p2", "Two", 222, "/two"),
            }
        )
        state = pm.build_awareness_state()
        by_id = {p.id: p for p in state.personas}
        assert by_id["p1"].name == "One"
        assert by_id["p1"].yjs_client_id == 111
        assert by_id["p1"].avatar_url == "/one"
        assert by_id["p2"].yjs_client_id == 222

    def test_empty_when_no_personas(self):
        pm = _manager({})
        assert pm.build_awareness_state().personas == []


class TestBroadcastAwarenessState:
    def test_writes_persona_list_to_awareness_client(self):
        pm = _manager({"p1": _mock_persona("p1", "One", 111)})
        pm.broadcast_awareness_state()
        pm._awareness.set_local_state_field.assert_called_once()
        field, payload = pm._awareness.set_local_state_field.call_args[0]
        assert field == "personas"
        assert payload[0]["id"] == "p1"
        assert payload[0]["yjs_client_id"] == 111

    def test_fixed_client_id_constant_is_53_bit(self):
        # A 53-bit integer is representable exactly as a JS number.
        assert 0 < PERSONA_MANAGER_AWARENESS_CLIENT_ID < 2**53


class TestSafeProcessAppliesMetadataFirst:
    """apply_message_metadata must run before process_message."""

    async def test_metadata_applied_before_processing(self):
        order = []
        persona = MagicMock()
        persona.name = "P"
        persona.log = MagicMock()
        persona.apply_message_metadata = AsyncMock(
            side_effect=lambda m: order.append("apply")
        )
        persona.process_message = AsyncMock(
            side_effect=lambda m: order.append("process")
        )
        persona.handle_uncaught_exception = AsyncMock()

        await _safe_process(persona, MagicMock(spec=Message))

        assert order == ["apply", "process"]

    async def test_processing_error_routed_to_handler(self):
        persona = MagicMock()
        persona.name = "P"
        persona.log = MagicMock()
        persona.apply_message_metadata = AsyncMock()
        exc = RuntimeError("boom")
        persona.process_message = AsyncMock(side_effect=exc)
        persona.handle_uncaught_exception = AsyncMock()

        await _safe_process(persona, MagicMock(spec=Message))

        persona.handle_uncaught_exception.assert_awaited_once_with(exc)

    async def test_metadata_error_routed_to_handler(self):
        # A failure while applying metadata is also delivered to the user rather
        # than crashing the dispatch task.
        persona = MagicMock()
        persona.name = "P"
        persona.log = MagicMock()
        exc = RuntimeError("bad spec")
        persona.apply_message_metadata = AsyncMock(side_effect=exc)
        persona.process_message = AsyncMock()
        persona.handle_uncaught_exception = AsyncMock()

        await _safe_process(persona, MagicMock(spec=Message))

        persona.process_message.assert_not_awaited()
        persona.handle_uncaught_exception.assert_awaited_once_with(exc)
