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


def _install_cancel_fixtures(jp_serverapp, chat_path, room_id, personas):
    """Wire a mock file_id_manager (chat_path -> file id) and a persona manager
    (room_id -> personas) into the server settings for a cancel request."""
    from unittest.mock import Mock

    # file id manager: chat_path -> file id, so room_id = text:chat:<file id>.
    file_id = room_id.split(":")[-1]
    mock_fim = Mock()
    mock_fim.get_id.return_value = file_id
    jp_serverapp.web_app.settings["file_id_manager"] = mock_fim

    mock_pm = Mock()
    mock_pm.personas = personas
    settings = jp_serverapp.web_app.settings.setdefault("jupyter-ai", {})
    settings["persona-managers"] = {room_id: mock_pm}


async def test_cancel_handler_calls_cancel_response(jp_fetch, jp_serverapp):
    """A POST cancels each processing persona in the chat via cancel_response()."""
    from unittest.mock import AsyncMock, Mock

    persona = Mock()
    persona.id = "jupyter-ai-personas::test::TestPersona"
    persona.processing = True
    persona.cancel_response = AsyncMock()

    _install_cancel_fixtures(
        jp_serverapp, "notebooks/chat.chat", "text:chat:file-1", {"p": persona}
    )

    response = await jp_fetch(
        "api", "ai", "personas", "cancel",
        method="POST", body="",
        params={"chat_path": "notebooks/chat.chat"},
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
        "notebooks/chat.chat",
        "text:chat:file-1",
        {"idle": idle, "busy": busy},
    )

    response = await jp_fetch(
        "api", "ai", "personas", "cancel",
        method="POST", body="",
        params={"chat_path": "notebooks/chat.chat"},
    )

    assert response.code == 200
    body = json.loads(response.body)
    assert body["cancelled"] == [busy.id]
    idle.cancel_response.assert_not_awaited()
    busy.cancel_response.assert_awaited_once()


async def test_cancel_handler_requires_chat_path(jp_fetch):
    """Missing chat_path is a 400."""
    from tornado.httpclient import HTTPClientError

    with pytest.raises(HTTPClientError) as exc:
        await jp_fetch("api", "ai", "personas", "cancel", method="POST", body="")
    assert exc.value.code == 400


async def test_cancel_handler_404_for_uninitialized_chat(jp_fetch, jp_serverapp):
    """A chat with no persona manager is a 404."""
    from tornado.httpclient import HTTPClientError
    from unittest.mock import Mock

    mock_fim = Mock()
    mock_fim.get_id.return_value = "file-unknown"
    jp_serverapp.web_app.settings["file_id_manager"] = mock_fim
    jp_serverapp.web_app.settings.setdefault("jupyter-ai", {})["persona-managers"] = {}

    with pytest.raises(HTTPClientError) as exc:
        await jp_fetch(
            "api", "ai", "personas", "cancel",
            method="POST", body="",
            params={"chat_path": "notebooks/chat.chat"},
        )
    assert exc.value.code == 404


async def test_cancel_handler_resolves_manager_by_path_rtc_free(jp_fetch, jp_serverapp):
    """In RTC-free mode the router registers the PersonaManager under the chat
    path, not `text:chat:{file_id}`. The handler must resolve it by path even
    when the RTC room_id is not present in the registry."""
    from unittest.mock import AsyncMock, Mock

    persona = Mock()
    persona.id = "jupyter-ai-personas::test::TestPersona"
    persona.processing = True
    persona.cancel_response = AsyncMock()

    chat_path = "notebooks/chat.chat"
    # file_id_manager resolves some id, but the RTC room_id it yields is NOT
    # registered — only the path-keyed manager is (as the router does RTC-free).
    mock_fim = Mock()
    mock_fim.get_id.return_value = "unregistered-file-id"
    jp_serverapp.web_app.settings["file_id_manager"] = mock_fim
    settings = jp_serverapp.web_app.settings.setdefault("jupyter-ai", {})
    settings["persona-managers"] = {chat_path: Mock(personas={"p": persona})}

    response = await jp_fetch(
        "api", "ai", "personas", "cancel",
        method="POST", body="",
        params={"chat_path": chat_path},
    )

    assert response.code == 200
    body = json.loads(response.body)
    assert persona.id in body["cancelled"]
    persona.cancel_response.assert_awaited_once()


# ---------------------------------------------------------------------------
# MessageHandler  (issue #125: keep the temporary-chat REST endpoint working)
# ---------------------------------------------------------------------------

from jupyterlab_chat.models import Message as ChatMessage
from jupyterlab_chat.models import NewMessage

from jupyter_ai_persona_manager.base_persona import BasePersona, PersonaDefaults
from jupyter_ai_persona_manager.persona_manager import PersonaManager


class _EchoPersona(BasePersona):
    """Minimal persona that echoes the incoming message back into the chat."""

    @property
    def defaults(self):
        return PersonaDefaults(
            name="EchoPersona",
            description="echoes messages",
            avatar_path="/tmp/echo.svg",
            system_prompt="echo",
        )

    async def process_message(self, message: ChatMessage) -> None:
        self.chat.add_message(
            NewMessage(body="echo: " + message.body, sender=self.id)
        )

    async def shutdown(self) -> None:
        # Keep teardown trivial; the base awareness cleanup isn't needed here.
        pass


@pytest.fixture
def jp_server_config(jp_server_config):
    """Enable jupyter_server_fileid so `file_id_manager` is present in settings,
    mirroring a real deployment (it is a declared dependency of this package).
    Overrides the root-conftest fixture for this module."""
    cfg = dict(jp_server_config)
    server = dict(cfg.get("ServerApp", {}))
    exts = dict(server.get("jpserver_extensions", {}))
    exts["jupyter_server_fileid"] = True
    server["jpserver_extensions"] = exts
    cfg["ServerApp"] = server
    return cfg


@pytest.fixture
def inject_echo_persona(monkeypatch):
    """Make the temporary PersonaManager created by MessageHandler load the echo
    persona (personas normally come from installed entry points)."""
    monkeypatch.setattr(
        PersonaManager,
        "_ep_persona_classes",
        [{"module": "echo", "persona_class": _EchoPersona, "traceback": None}],
    )


class TestMessageHandler:
    """The `/api/ai/message/<persona>` endpoint spins up a throwaway chat,
    routes a single message to the named persona, and returns its output."""

    async def test_returns_persona_response(
        self, jp_fetch, jp_serverapp, inject_echo_persona
    ):
        """Happy path: the endpoint returns the persona's chat output (#125)."""
        body = json.dumps({"message": "hello"})
        response = await jp_fetch(
            "api", "ai", "message", "EchoPersona", method="POST", body=body
        )
        assert response.code == 200
        data = json.loads(response.body)
        assert "echo: hello" in data["response"]

    async def test_404_for_unknown_persona(
        self, jp_fetch, jp_serverapp, inject_echo_persona
    ):
        """An unknown persona name is a clean 404, not a 500."""
        from tornado.httpclient import HTTPClientError

        body = json.dumps({"message": "hi"})
        with pytest.raises(HTTPClientError) as exc:
            await jp_fetch(
                "api", "ai", "message", "NoSuchPersona", method="POST", body=body
            )
        assert exc.value.code == 404

    async def test_400_for_missing_message_field(
        self, jp_fetch, jp_serverapp, inject_echo_persona
    ):
        """A body with no `message` field is a 400."""
        from tornado.httpclient import HTTPClientError

        with pytest.raises(HTTPClientError) as exc:
            await jp_fetch(
                "api", "ai", "message", "EchoPersona",
                method="POST", body=json.dumps({}),
            )
        assert exc.value.code == 400

    async def test_500_when_fileid_manager_missing(
        self, jp_fetch, jp_serverapp, inject_echo_persona
    ):
        """Without `file_id_manager` the handler fails cleanly with a 500 rather
        than an uncaught `AttributeError` on `None.index` (the pre-fix behavior
        for #125). We remove the manager after startup to simulate its
        absence."""
        from tornado.httpclient import HTTPClientError

        jp_serverapp.web_app.settings.pop("file_id_manager", None)
        with pytest.raises(HTTPClientError) as exc:
            await jp_fetch(
                "api", "ai", "message", "EchoPersona",
                method="POST", body=json.dumps({"message": "hi"}),
            )
        assert exc.value.code == 500
