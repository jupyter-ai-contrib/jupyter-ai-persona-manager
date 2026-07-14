import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import quote

import pytest

from jupyter_ai_persona_manager.handlers import build_avatar_cache
from jupyter_ai_persona_manager.persona_manager import (
    PERSONA_MANAGER_AWARENESS_CLIENT_ID,
)


async def test_avatar_handler_serves_file(jp_fetch, jp_serverapp, tmp_path):
    """Test that the avatar handler can serve avatar files."""

    # Create avatar file
    avatar_file = tmp_path / "test.svg"
    avatar_file.write_text('<svg><circle r="10"/></svg>')

    # Create mock persona with avatar
    mock_persona = Mock()
    mock_persona.defaults.avatar_path = str(avatar_file)
    mock_persona.name = "TestPersona"
    mock_persona.id = "jupyter-ai-personas::test::TestPersona"

    # Create mock persona manager
    mock_pm = Mock()
    mock_pm.personas = {"test-persona": mock_persona}

    # Add to settings
    if 'jupyter-ai' not in jp_serverapp.web_app.settings:
        jp_serverapp.web_app.settings['jupyter-ai'] = {}
    jp_serverapp.web_app.settings['jupyter-ai']['persona-managers'] = {
        'room1': mock_pm
    }

    # Build the avatar cache
    build_avatar_cache(jp_serverapp.web_app.settings['jupyter-ai']['persona-managers'])

    # Fetch the avatar using URL-encoded persona ID
    encoded_id = quote(mock_persona.id, safe='')
    response = await jp_fetch("api", "ai", "avatars", encoded_id)

    # Verify response
    assert response.code == 200
    assert b'<svg><circle r="10"/></svg>' in response.body
    assert 'image/svg+xml' in response.headers.get('Content-Type', '')


async def test_avatar_handler_404_for_missing_file(jp_fetch, jp_serverapp):
    """Test that the avatar handler returns 404 for missing files."""

    # Create mock persona manager with no matching avatar
    mock_pm = Mock()
    mock_pm.personas = {}

    # Add to settings
    if 'jupyter-ai' not in jp_serverapp.web_app.settings:
        jp_serverapp.web_app.settings['jupyter-ai'] = {}
    jp_serverapp.web_app.settings['jupyter-ai']['persona-managers'] = {
        'room1': mock_pm
    }

    # Build the avatar cache (will be empty)
    build_avatar_cache(jp_serverapp.web_app.settings['jupyter-ai']['persona-managers'])

    # Try to fetch a non-existent avatar
    with pytest.raises(Exception) as exc_info:
        await jp_fetch("api", "ai", "avatars", "nonexistent-id")

    # Verify 404 response
    assert '404' in str(exc_info.value) or 'Not Found' in str(exc_info.value)


async def test_avatar_handler_serves_png(jp_fetch, jp_serverapp, tmp_path):
    """Test that the avatar handler can serve PNG files."""

    # Create PNG file
    avatar_file = tmp_path / "test.png"
    avatar_file.write_bytes(b'\x89PNG\r\n\x1a\n')

    # Create mock persona with avatar
    mock_persona = Mock()
    mock_persona.defaults.avatar_path = str(avatar_file)
    mock_persona.name = "TestPersona"
    mock_persona.id = "jupyter-ai-personas::test::AnotherPersona"

    # Create mock persona manager
    mock_pm = Mock()
    mock_pm.personas = {"test-persona": mock_persona}

    # Add to settings
    if 'jupyter-ai' not in jp_serverapp.web_app.settings:
        jp_serverapp.web_app.settings['jupyter-ai'] = {}
    jp_serverapp.web_app.settings['jupyter-ai']['persona-managers'] = {
        'room1': mock_pm
    }

    # Build the avatar cache
    build_avatar_cache(jp_serverapp.web_app.settings['jupyter-ai']['persona-managers'])

    # Fetch the avatar using URL-encoded persona ID
    encoded_id = quote(mock_persona.id, safe='')
    response = await jp_fetch("api", "ai", "avatars", encoded_id)

    # Verify response
    assert response.code == 200
    assert response.body.startswith(b'\x89PNG')
    assert 'image/png' in response.headers.get('Content-Type', '')


def _install_file_id_manager(jp_serverapp, chat_path, file_id):
    """Register a mock file-id manager mapping chat_path -> file_id."""
    fim = Mock()
    fim.get_id.return_value = file_id
    jp_serverapp.web_app.settings["file_id_manager"] = fim
    return fim


async def test_readiness_returns_fixed_client_id_when_manager_registered(
    jp_fetch, jp_serverapp
):
    """The readiness endpoint returns the manager's fixed client ID once the
    PersonaManager for the chat is registered."""
    _install_file_id_manager(jp_serverapp, "chat.ipynb", "file-1")
    room_id = "text:chat:file-1"
    jp_serverapp.web_app.settings.setdefault("jupyter-ai", {})
    jp_serverapp.web_app.settings["jupyter-ai"]["persona-managers"] = {
        room_id: Mock()
    }

    response = await jp_fetch(
        "api", "ai", "persona_manager_awareness", params={"chat_path": "chat.ipynb"}
    )

    assert response.code == 200
    body = json.loads(response.body)
    assert body["persona_manager_client_id"] == PERSONA_MANAGER_AWARENESS_CLIENT_ID


async def test_readiness_503_when_manager_not_registered(jp_fetch, jp_serverapp):
    """Before the PersonaManager is registered, the endpoint reports not-ready
    (503) so the client retries."""
    _install_file_id_manager(jp_serverapp, "chat.ipynb", "file-1")
    jp_serverapp.web_app.settings.setdefault("jupyter-ai", {})
    jp_serverapp.web_app.settings["jupyter-ai"]["persona-managers"] = {}

    with pytest.raises(Exception) as exc_info:
        await jp_fetch(
            "api",
            "ai",
            "persona_manager_awareness",
            params={"chat_path": "chat.ipynb"},
        )

    assert "503" in str(exc_info.value)


async def test_readiness_400_without_chat_path(jp_fetch, jp_serverapp):
    """chat_path is required."""
    _install_file_id_manager(jp_serverapp, "chat.ipynb", "file-1")

    with pytest.raises(Exception) as exc_info:
        await jp_fetch("api", "ai", "persona_manager_awareness")

    assert "400" in str(exc_info.value)


