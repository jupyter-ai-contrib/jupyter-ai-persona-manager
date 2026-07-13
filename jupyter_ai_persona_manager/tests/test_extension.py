"""
Tests for the PersonaManagerExtension class.
"""

import pytest
from unittest.mock import AsyncMock, Mock, PropertyMock, patch
from jupyter_ai_persona_manager.extension import PersonaManagerExtension


@pytest.fixture
def extension(mock_server_app):
    """Create a PersonaManagerExtension instance for testing."""
    ext = PersonaManagerExtension()
    ext.serverapp = mock_server_app
    ext.log = mock_server_app.log
    ext._stopping_rooms = {}
    ext._stop_lock = __import__('asyncio').Lock()
    return ext


@pytest.mark.asyncio
async def test_stop_extension_with_no_persona_managers(extension, mock_server_app):
    """Test that stop_extension works when no persona managers exist."""
    # Setup: ensure persona-managers dict exists but is empty
    mock_server_app.web_app.settings['jupyter-ai']['persona-managers'] = {}

    # Should not raise an exception
    await extension.stop_extension()


@pytest.mark.asyncio
async def test_stop_extension_with_persona_managers(extension, mock_server_app):
    """Test that stop_extension properly cleans up persona managers."""
    # Setup: create mock persona managers
    mock_pm1 = Mock()
    mock_pm1.shutdown_personas = AsyncMock()
    mock_pm2 = Mock()
    mock_pm2.shutdown_personas = AsyncMock()

    mock_server_app.web_app.settings['jupyter-ai']['persona-managers'] = {
        'room1': mock_pm1,
        'room2': mock_pm2
    }

    # Act
    await extension.stop_extension()

    # Assert: shutdown_personas was called for each manager
    mock_pm1.shutdown_personas.assert_called_once()
    mock_pm2.shutdown_personas.assert_called_once()

    # Assert: the dictionary was cleared
    assert len(mock_server_app.web_app.settings['jupyter-ai']['persona-managers']) == 0


@pytest.mark.asyncio
async def test_stop_extension_with_failing_shutdown(extension, mock_server_app):
    """Test that stop_extension handles exceptions during shutdown gracefully."""
    # Setup: create a persona manager that fails on shutdown
    mock_pm = Mock()
    mock_pm.shutdown_personas = AsyncMock(side_effect=Exception("Shutdown failed"))

    mock_server_app.web_app.settings['jupyter-ai']['persona-managers'] = {
        'room1': mock_pm
    }

    # Should not raise an exception, but should log the error
    await extension.stop_extension()

    # Assert: error was logged
    assert extension.log.exception.called


@pytest.mark.asyncio
async def test_stop_extension_without_jupyter_ai_settings(extension, mock_server_app):
    """Test that stop_extension handles missing jupyter-ai settings gracefully."""
    # Setup: remove jupyter-ai from settings
    mock_server_app.web_app.settings.pop('jupyter-ai', None)

    # Should not raise an exception
    await extension.stop_extension()


from traitlets.config import Config

from jupyter_ai_persona_manager.persona_manager import PersonaManager


class TestDefaultPersonaId:
    """The default persona ID resolved for the frontend PageConfig."""

    def test_uses_the_trait_default_when_not_configured(self, extension):
        assert (
            extension._default_persona_id()
            == PersonaManager.default_persona_id.default_value
        )

    def test_uses_a_user_configured_override(self, extension):
        extension.config = Config(
            {"PersonaManager": {"default_persona_id": "pkg::Custom"}}
        )
        assert extension._default_persona_id() == "pkg::Custom"

    def test_returns_empty_string_when_default_is_disabled(self, extension):
        # `default_persona_id` allows None to disable the default persona.
        extension.config = Config({"PersonaManager": {"default_persona_id": None}})
        assert extension._default_persona_id() == ""


def test_initialize_settings_advertises_default_persona(extension, mock_server_app):
    """initialize_settings writes the default persona ID into page_config_data
    so the frontend can read it at startup."""
    # `self.settings` mirrors the web app settings for an ExtensionApp; provide
    # it so the method can seed the persona-managers dict.
    extension.settings = mock_server_app.web_app.settings
    # `initialize_settings` kicks off a router-integration task on the event
    # loop; stub the loop (a read-only property) and the coroutine it schedules
    # so the test needs no running loop.
    with patch.object(
        type(extension), "event_loop", new_callable=PropertyMock
    ) as event_loop, patch.object(extension, "_setup_router_integration"):
        event_loop.return_value = Mock()
        extension.initialize_settings()

    page_config = mock_server_app.web_app.settings["page_config_data"]
    assert (
        page_config["jupyter_ai_default_persona"]
        == PersonaManager.default_persona_id.default_value
    )
