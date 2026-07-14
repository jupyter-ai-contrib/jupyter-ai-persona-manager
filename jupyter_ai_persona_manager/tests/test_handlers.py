import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import quote

import pytest

from jupyter_ai_persona_manager.handlers import build_avatar_cache


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


