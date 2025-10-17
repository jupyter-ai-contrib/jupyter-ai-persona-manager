"""
Tests for the PersonaManagerExtension class.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from jupyter_ai_persona_manager.extension import PersonaManagerExtension


@pytest.fixture
def extension(mock_server_app):
    """Create a PersonaManagerExtension instance for testing."""
    ext = PersonaManagerExtension()
    ext.serverapp = mock_server_app
    ext.log = mock_server_app.log
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
    assert extension.log.error.called


@pytest.mark.asyncio
async def test_stop_extension_without_jupyter_ai_settings(extension, mock_server_app):
    """Test that stop_extension handles missing jupyter-ai settings gracefully."""
    # Setup: remove jupyter-ai from settings
    mock_server_app.web_app.settings.pop('jupyter-ai', None)

    # Should not raise an exception
    await extension.stop_extension()
