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


# ---------------------------------------------------------------------------
# MessageHandler
# ---------------------------------------------------------------------------
#
# The magics feature (jupyter-ai-magic-commands) POSTs here directly, bypassing
# the live chat path (`BasePersona.on_message`). PR #160 fixed this handler to
# route through the shared entry point instead of reimplementing its steps
# inline, so a persona's `prepare()` hook runs before `process_message()`, and
# the handler's wait loop only matters because `processing` is tracked for the
# duration of the call.
#
# These tests exercise the real (non-mocked) handler end-to-end: a real
# PersonaManager discovers a real persona class written to
# `<jp_root_dir>/.jupyter/personas/`, exactly as it would for a real magics
# request. `MessageHandler` always creates its ephemeral chat at `root_dir`
# itself (no subdirectory — see `ychat.initial_path` in handlers.py), so the
# fixture persona must live directly under `jp_root_dir`, not a nested dir.

_PREPARE_REQUIRED_PERSONA_SOURCE = '''
from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message


class PrepareRequiredPersona(BasePersona):
    """Mirrors an ACP persona: process_message() depends on state that only
    prepare() sets up (e.g. a spawned agent session)."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Prepare Required Persona",
            description="test",
            avatar_path="",
            system_prompt="unused",
        )

    async def prepare(self) -> None:
        self._session_ready = True

    async def process_message(self, message: Message) -> None:
        if not getattr(self, "_session_ready", False):
            raise RuntimeError("session not ready: prepare() never ran")
        self.send_message(f"session ready: {message.body}")
'''

_BROKEN_PREPARE_PERSONA_SOURCE = '''
from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message


class BrokenPreparePersona(BasePersona):
    """A persona whose one-time prepare() hook always fails, mirroring an ACP
    agent that can't spawn (e.g. not authenticated)."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Broken Prepare Persona",
            description="test",
            avatar_path="",
            system_prompt="unused",
        )

    async def prepare(self) -> None:
        raise RuntimeError("boom: prepare failed")

    async def process_message(self, message: Message) -> None:
        self.send_message("should never run")
'''

_STREAMING_PERSONA_SOURCE = '''
import asyncio

from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message


class StreamingPersona(BasePersona):
    """Streams its reply over a few chunks, so a test can assert the handler
    waits for the full stream rather than returning early."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Streaming Persona",
            description="test",
            avatar_path="",
            system_prompt="unused",
        )

    async def _chunks(self):
        for chunk in ("one ", "two ", "three"):
            await asyncio.sleep(0.05)
            yield chunk

    async def process_message(self, message: Message) -> None:
        await self.stream_message(self._chunks())
'''


def _install_message_persona(jp_root_dir, filename: str, source: str) -> None:
    """Writes a fixture persona `.py` file into `<jp_root_dir>/.jupyter/personas/`
    — the directory `MessageHandler`'s ephemeral, root-level chat resolves to,
    since that chat has no directory of its own. Exercises real on-disk persona
    discovery rather than mocking the persona.
    """
    personas_dir = Path(jp_root_dir) / ".jupyter" / "personas"
    personas_dir.mkdir(parents=True, exist_ok=True)
    (personas_dir / filename).write_text(source)


async def test_message_handler_runs_prepare_before_processing(jp_fetch, jp_root_dir):
    """Regression test for #160: the magics endpoint must run a persona's
    prepare() hook before process_message(), just like the live chat path.
    Before the fix, this handler called process_message() directly, so a
    persona relying on prepare() (as ACP personas do) would raise here."""
    _install_message_persona(
        jp_root_dir, "prepare-required_persona.py", _PREPARE_REQUIRED_PERSONA_SOURCE
    )

    response = await jp_fetch(
        "api", "ai", "message", "Prepare Required Persona",
        method="POST",
        body=json.dumps({"message": "hi"}),
    )

    assert response.code == 200
    body = json.loads(response.body)
    assert body["response"] == "session ready: hi"


async def test_message_handler_surfaces_prepare_failure(jp_fetch, jp_root_dir):
    """A persona whose prepare() fails must deliver a readable error into the
    response rather than raising an opaque 500 out of the handler."""
    _install_message_persona(
        jp_root_dir, "broken-prepare_persona.py", _BROKEN_PREPARE_PERSONA_SOURCE
    )

    response = await jp_fetch(
        "api", "ai", "message", "Broken Prepare Persona",
        method="POST",
        body=json.dumps({"message": "hi"}),
    )

    assert response.code == 200
    body = json.loads(response.body)
    assert "An error occurred while processing your message" in body["response"]
    assert "boom: prepare failed" in body["response"]


async def test_message_handler_waits_for_streaming_reply(jp_fetch, jp_root_dir):
    """A persona that replies via `stream_message()` (rather than the
    single-shot `send_message()` used above) must have every chunk captured,
    not just whatever landed first — covering the streaming reply path the
    handler's `while target_persona.processing` wait loop exists for."""
    _install_message_persona(
        jp_root_dir, "streaming_persona.py", _STREAMING_PERSONA_SOURCE
    )

    response = await jp_fetch(
        "api", "ai", "message", "Streaming Persona",
        method="POST",
        body=json.dumps({"message": "go"}),
    )

    assert response.code == 200
    body = json.loads(response.body)
    assert body["response"] == "one two three"
