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




# ---------------------------------------------------------------------------
# CancelHandler
# ---------------------------------------------------------------------------


def _install_cancel_fixtures(jp_serverapp, chat_id, personas):
    """Register a persona manager under its chat id for a cancel request.
    Managers are keyed by the chat's stable id (``chat.get_id()``), so the
    handler resolves them by a direct lookup."""
    from unittest.mock import Mock

    mock_pm = Mock()
    mock_pm.personas = personas
    mock_pm.chat.get_id.return_value = chat_id
    settings = jp_serverapp.web_app.settings.setdefault("jupyter-ai", {})
    settings["persona-managers"] = {chat_id: mock_pm}


async def test_cancel_handler_calls_cancel_response(jp_fetch, jp_serverapp):
    """A POST cancels each processing persona in the chat via cancel_response()."""
    from unittest.mock import AsyncMock, Mock

    persona = Mock()
    persona.id = "jupyter-ai-personas::test::TestPersona"
    persona.processing = True
    persona.cancel_response = AsyncMock()

    _install_cancel_fixtures(jp_serverapp, "chat-abc", {"p": persona})

    response = await jp_fetch(
        "api", "ai", "personas", "cancel",
        method="POST", body="",
        params={"chat_id": "chat-abc"},
    )

    assert response.code == 200
    body = json.loads(response.body)
    assert body["status"] == "cancelled"
    assert persona.id in body["cancelled"]
    persona.cancel_response.assert_awaited_once()


async def test_cancel_handler_skips_idle_personas(jp_fetch, jp_serverapp):
    """A persona that isn't processing is left alone — no cancel_response call."""
    from unittest.mock import AsyncMock, Mock

    idle = Mock()
    idle.id = "jupyter-ai-personas::test::IdlePersona"
    idle.processing = False
    idle.cancel_response = AsyncMock()

    busy = Mock()
    busy.id = "jupyter-ai-personas::test::BusyPersona"
    busy.processing = True
    busy.cancel_response = AsyncMock()

    _install_cancel_fixtures(
        jp_serverapp,
        "chat-abc",
        {"idle": idle, "busy": busy},
    )

    response = await jp_fetch(
        "api", "ai", "personas", "cancel",
        method="POST", body="",
        params={"chat_id": "chat-abc"},
    )

    assert response.code == 200
    body = json.loads(response.body)
    assert body["cancelled"] == [busy.id]
    idle.cancel_response.assert_not_awaited()
    busy.cancel_response.assert_awaited_once()


async def test_cancel_handler_requires_chat_id(jp_fetch):
    """Missing chat_id is a 400."""
    from tornado.httpclient import HTTPClientError

    with pytest.raises(HTTPClientError) as exc:
        await jp_fetch("api", "ai", "personas", "cancel", method="POST", body="")
    assert exc.value.code == 400


async def test_cancel_handler_404_for_uninitialized_chat(jp_fetch, jp_serverapp):
    """A chat with no persona manager is a 404."""
    from tornado.httpclient import HTTPClientError

    jp_serverapp.web_app.settings.setdefault("jupyter-ai", {})["persona-managers"] = {}

    with pytest.raises(HTTPClientError) as exc:
        await jp_fetch(
            "api", "ai", "personas", "cancel",
            method="POST", body="",
            params={"chat_id": "chat-abc"},
        )
    assert exc.value.code == 404


async def test_cancel_handler_resolves_manager_by_id(jp_fetch, jp_serverapp):
    """Persona managers are keyed by the chat's stable id (``chat.get_id()``).
    The handler resolves the manager for the requested chat_id by a direct
    lookup and leaves other chats' managers untouched."""
    from unittest.mock import AsyncMock, Mock

    target = Mock()
    target.id = "jupyter-ai-personas::test::TargetPersona"
    target.processing = True
    target.cancel_response = AsyncMock()

    other = Mock()
    other.id = "jupyter-ai-personas::test::OtherPersona"
    other.processing = True
    other.cancel_response = AsyncMock()

    target_pm = Mock(personas={"p": target})
    target_pm.chat.get_id.return_value = "chat-abc"
    other_pm = Mock(personas={"p": other})
    other_pm.chat.get_id.return_value = "chat-xyz"
    settings = jp_serverapp.web_app.settings.setdefault("jupyter-ai", {})
    settings["persona-managers"] = {"chat-abc": target_pm, "chat-xyz": other_pm}

    response = await jp_fetch(
        "api", "ai", "personas", "cancel",
        method="POST", body="",
        params={"chat_id": "chat-abc"},
    )

    assert response.code == 200
    body = json.loads(response.body)
    assert target.id in body["cancelled"]
    target.cancel_response.assert_awaited_once()
    # A different chat's persona must not be cancelled.
    other.cancel_response.assert_not_awaited()
