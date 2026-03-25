"""
Tests for message edit and delete lifecycle hooks.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock, MagicMock, patch
from jupyterlab_chat.models import Message
from traitlets.config import LoggingConfigurable
from jupyter_ai_persona_manager.base_persona import BasePersona, PersonaDefaults


class ConcretePersona(BasePersona):
    """Minimal concrete persona for testing."""

    @property
    def defaults(self):
        return PersonaDefaults(
            name="Test",
            description="Test persona",
            avatar_path="/test.svg",
            system_prompt="You are a test persona.",
        )

    async def process_message(self, message: Message):
        pass


class StubParent(LoggingConfigurable):
    """Minimal parent satisfying the Configurable trait requirement."""
    base_url = "/"


@pytest.fixture
def mock_ychat():
    ychat = MagicMock()
    ychat.awareness = MagicMock()
    ychat.set_user = Mock()
    return ychat


@pytest.fixture
def stub_parent():
    return StubParent()


def _make_persona(cls, parent, ychat):
    """Create a persona with PersonaAwareness heartbeat patched out."""
    with patch(
        "jupyter_ai_persona_manager.base_persona.PersonaAwareness"
    ) as MockAwareness:
        MockAwareness.return_value = MagicMock()
        return cls(parent=parent, ychat=ychat)


@pytest.fixture
def persona(mock_ychat, stub_parent):
    return _make_persona(ConcretePersona, stub_parent, mock_ychat)


class TestOnMessageEdited:
    """Test BasePersona.on_message_edited lifecycle hook."""

    @pytest.mark.asyncio
    async def test_default_calls_process_message(self, persona):
        """Default on_message_edited delegates to process_message."""
        persona.process_message = AsyncMock()
        msg = Message(id="1", body="edited text", sender="user", time=123)

        await persona.on_message_edited(msg)

        persona.process_message.assert_awaited_once_with(msg)

    @pytest.mark.asyncio
    async def test_override_custom_behavior(self, mock_ychat, stub_parent):
        """Subclasses can override on_message_edited with custom logic."""
        edit_log = []

        class CustomPersona(ConcretePersona):
            async def on_message_edited(self, message):
                edit_log.append(message.id)

        persona = _make_persona(CustomPersona, stub_parent, mock_ychat)
        msg = Message(id="msg-42", body="new text", sender="user", time=123)

        await persona.on_message_edited(msg)

        assert edit_log == ["msg-42"]


class TestOnMessageDeleted:
    """Test BasePersona.on_message_deleted lifecycle hook."""

    @pytest.mark.asyncio
    async def test_default_is_noop(self, persona):
        """Default on_message_deleted does nothing."""
        msg = Message(id="1", body="text", sender="user", time=123, deleted=True)

        # Should not raise
        await persona.on_message_deleted(msg)

    @pytest.mark.asyncio
    async def test_override_custom_behavior(self, mock_ychat, stub_parent):
        """Subclasses can override on_message_deleted with custom logic."""
        delete_log = []

        class CustomPersona(ConcretePersona):
            async def on_message_deleted(self, message):
                delete_log.append(message.id)

        persona = _make_persona(CustomPersona, stub_parent, mock_ychat)
        msg = Message(id="msg-99", body="text", sender="user", time=123, deleted=True)

        await persona.on_message_deleted(msg)

        assert delete_log == ["msg-99"]
