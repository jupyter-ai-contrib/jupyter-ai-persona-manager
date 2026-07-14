"""
Tests for the PersonaManager awareness state (the persona list published under
the fixed client ID) and for spec application happening before processing in the
message-dispatch path.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

from jupyterlab_chat.models import Message
from jupyterlab_chat.ychat import YChat
from pycrdt import Awareness

from jupyter_ai_persona_manager.persona_awareness import (
    PERSONA_MANAGER_AWARENESS_CLIENT_ID,
    PersonaManagerAwareness,
)
from jupyter_ai_persona_manager.persona_manager import (
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
    """A PersonaManager wired to a real PersonaManagerAwareness over an in-memory
    YChat (no event loop, so no heartbeat), plus the state the awareness methods
    read."""
    ychat = YChat()
    ychat.awareness = Awareness(ydoc=ychat._ydoc)
    pm = PersonaManager.__new__(PersonaManager)
    pm._personas = personas
    pm.log = logging.getLogger("test-pm-awareness")
    pm._awareness = PersonaManagerAwareness(ychat=ychat, log=pm.log)
    return pm


class TestAwarenessState:
    def test_publishes_every_persona_with_yjs_client_id(self):
        pm = _manager(
            {
                "p1": _mock_persona("p1", "One", 111, "/one"),
                "p2": _mock_persona("p2", "Two", 222, "/two"),
            }
        )
        pm._publish_persona_list()

        # Read the published personas back off the awareness slot.
        by_id = {p.id: p for p in pm._awareness.personas}
        assert by_id["p1"].name == "One"
        assert by_id["p1"].yjs_client_id == 111
        assert by_id["p1"].avatar_url == "/one"
        assert by_id["p2"].yjs_client_id == 222

    def test_empty_when_no_personas(self):
        pm = _manager({})
        pm._publish_persona_list()
        assert pm._awareness.personas == []

    def test_republishes_after_personas_change(self):
        pm = _manager({"p1": _mock_persona("p1", "One", 111)})
        pm._publish_persona_list()
        assert [p.id for p in pm._awareness.personas] == ["p1"]

        pm._personas = {"p2": _mock_persona("p2", "Two", 222)}
        pm._publish_persona_list()
        assert [p.id for p in pm._awareness.personas] == ["p2"]

    def test_fixed_client_id_constant_is_53_bit(self):
        # A 53-bit integer is representable exactly as a JS number.
        assert 0 < PERSONA_MANAGER_AWARENESS_CLIENT_ID < 2**53


class TestSafeProcessAppliesSpecsFirst:
    """apply_specs_in_message must run before process_message."""

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
