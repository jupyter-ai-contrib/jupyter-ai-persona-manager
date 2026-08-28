"""Tests for BasePersona.handle_uncaught_exception() and stream_message() re-raise."""

import asyncio
from unittest.mock import MagicMock

import pytest

from jupyter_ai_persona_manager.base_persona import BasePersona, PersonaDefaults
from jupyter_ai_persona_manager.mcp_server_models import (
    HttpHeader,
    McpServerHttp,
    McpSettings,
)
from jupyter_ai_persona_manager.persona_events import PersonaSessionState


@pytest.fixture
def mock_ychat():
    """Minimal YChat mock — avoids the jupyterlab_chat circular import in conftest."""
    mock = MagicMock()
    mock.add_message = MagicMock(return_value="msg-123")
    mock.update_message = MagicMock()
    mock.get_message = MagicMock(return_value=None)
    return mock


class _ConcretePersona(BasePersona):
    """Minimal concrete subclass for testing BasePersona methods."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="TestPersona",
            description="A test persona",
            avatar_path="",
            system_prompt="",
        )

    async def process_message(self, message):
        pass


def _make_persona(mock_ychat):
    """Create a _ConcretePersona instance wired to a mock YChat.

    Uses __new__ to bypass __init__ (which would require a real PersonaAwareness
    and YChat). Sets only the attributes needed by handle_uncaught_exception and
    stream_message — neither accesses self.parent.
    """
    persona = _ConcretePersona.__new__(_ConcretePersona)
    persona.chat = mock_ychat
    persona.log = MagicMock()
    persona.state = MagicMock()
    persona._processing_count = 0
    persona._processing_message = None
    persona._processing_lock = None
    return persona


# ---------------------------------------------------------------------------
# TestHandleUncaughtException
# ---------------------------------------------------------------------------

class TestHandleUncaughtException:

    @pytest.mark.asyncio
    async def test_sends_message_to_chat(self, mock_ychat):
        persona = _make_persona(mock_ychat)
        await persona.handle_uncaught_exception(RuntimeError("boom"))
        mock_ychat.add_message.assert_called_once()
        body = mock_ychat.add_message.call_args[0][0].body
        assert "<details" in body
        assert "</details>" in body

    @pytest.mark.asyncio
    async def test_summary_contains_exception_type(self, mock_ychat):
        persona = _make_persona(mock_ychat)
        await persona.handle_uncaught_exception(RuntimeError("boom"))
        body = mock_ychat.add_message.call_args[0][0].body
        assert "RuntimeError" in body
        assert "<summary>" in body

    @pytest.mark.asyncio
    async def test_summary_contains_exception_message(self, mock_ychat):
        persona = _make_persona(mock_ychat)
        await persona.handle_uncaught_exception(RuntimeError("something went wrong"))
        body = mock_ychat.add_message.call_args[0][0].body
        assert "something went wrong" in body

    @pytest.mark.asyncio
    async def test_summary_truncates_long_exception_message(self, mock_ychat):
        persona = _make_persona(mock_ychat)
        long_msg = "x" * 200
        await persona.handle_uncaught_exception(RuntimeError(long_msg))
        body = mock_ychat.add_message.call_args[0][0].body
        # The summary line is truncated (shows "…") — the full message still
        # appears in the traceback section, which is intentional.
        assert "…" in body
        truncated = "x" * 120 + "…"
        assert truncated in body

    @pytest.mark.asyncio
    async def test_body_contains_traceback(self, mock_ychat):
        persona = _make_persona(mock_ychat)
        try:
            raise ValueError("traceback test")
        except ValueError as exc:
            await persona.handle_uncaught_exception(exc)
        body = mock_ychat.add_message.call_args[0][0].body
        assert "ValueError" in body
        assert "traceback test" in body
        assert "<pre" in body

    @pytest.mark.asyncio
    async def test_html_special_chars_are_escaped(self, mock_ychat):
        persona = _make_persona(mock_ychat)
        await persona.handle_uncaught_exception(RuntimeError("<script>alert(1)</script>"))
        body = mock_ychat.add_message.call_args[0][0].body
        assert "<script>" not in body
        assert "&lt;script&gt;" in body

    @pytest.mark.asyncio
    async def test_default_is_overridable(self, mock_ychat):
        custom_called_with = []

        class _CustomPersona(_ConcretePersona):
            async def handle_uncaught_exception(self, exc: Exception) -> None:
                custom_called_with.append(exc)

        persona = _make_persona(mock_ychat)
        persona.__class__ = _CustomPersona
        exc = RuntimeError("custom")
        await _CustomPersona.handle_uncaught_exception(persona, exc)
        assert custom_called_with == [exc]
        mock_ychat.add_message.assert_not_called()


# ---------------------------------------------------------------------------
# TestStreamMessageReRaise
# ---------------------------------------------------------------------------

class TestStreamMessageReRaise:

    @pytest.mark.asyncio
    async def test_re_raises_after_logging(self, mock_ychat):
        persona = _make_persona(mock_ychat)

        async def _failing_stream():
            yield "first chunk"
            raise ValueError("stream error")

        with pytest.raises(ValueError, match="stream error"):
            await persona.stream_message(_failing_stream())

        persona.log.error.assert_called_once()
        persona.log.exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_awareness_cleaned_up_on_raise(self, mock_ychat):
        persona = _make_persona(mock_ychat)

        async def _failing_stream():
            raise RuntimeError("fail")
            yield  # make it an async generator

        with pytest.raises(RuntimeError):
            await persona.stream_message(_failing_stream())

        # The `finally` clears the status via clear_status(), which drives the
        # chat's writers indicator (broadcast_writing_status).
        persona.chat.broadcast_writing_status.assert_called_with(
            persona.as_user(), None
        )


# ---------------------------------------------------------------------------
# TestCancelResponse
# ---------------------------------------------------------------------------

class TestCancelResponse:

    @pytest.mark.asyncio
    async def test_default_is_a_noop(self, mock_ychat):
        # The base implementation is optional: awaiting it does nothing and does
        # not touch the chat or awareness. A persona with nothing cancellable
        # inherits this.
        persona = _make_persona(mock_ychat)

        await persona.cancel_response()

        mock_ychat.add_message.assert_not_called()
        mock_ychat.update_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_subclass_can_override(self, mock_ychat):
        # A streaming/long-running persona overrides cancel_response to interrupt
        # its backend; PersonaManager calls it the same way regardless.
        cancelled = False

        class _CancellablePersona(_ConcretePersona):
            async def cancel_response(self) -> None:
                nonlocal cancelled
                cancelled = True

        persona = _CancellablePersona.__new__(_CancellablePersona)
        persona.chat = mock_ychat
        persona.log = MagicMock()
        persona.state = MagicMock()

        await persona.cancel_response()

        assert cancelled is True


# ---------------------------------------------------------------------------
# TestProcessing
# ---------------------------------------------------------------------------

class TestProcessing:

    def test_not_processing_by_default(self, mock_ychat):
        persona = _make_persona(mock_ychat)
        assert persona.processing is False
        assert persona.processing_message is None

    @pytest.mark.asyncio
    async def test_track_processing_sets_message_and_flag(self, mock_ychat):
        persona = _make_persona(mock_ychat)
        message = object()
        async with persona.track_processing(message):
            assert persona.processing is True
            assert persona.processing_message is message
        assert persona.processing is False
        assert persona.processing_message is None

    @pytest.mark.asyncio
    async def test_track_processing_serializes(self, mock_ychat):
        # Processing is serial: two concurrent messages never overlap, so a
        # persona replies to at most one web client at a time.
        persona = _make_persona(mock_ychat)
        events: list[tuple[str, str]] = []

        async def worker(tag: str, message: object) -> None:
            async with persona.track_processing(message):
                events.append(("enter", tag))
                assert persona.processing_message is message
                await asyncio.sleep(0.05)
                events.append(("exit", tag))

        await asyncio.gather(worker("a", object()), worker("b", object()))

        # Each enter is immediately followed by its own exit (no interleaving).
        assert events in (
            [("enter", "a"), ("exit", "a"), ("enter", "b"), ("exit", "b")],
            [("enter", "b"), ("exit", "b"), ("enter", "a"), ("exit", "a")],
        )
        assert persona.processing is False
        assert persona.processing_message is None

    @pytest.mark.asyncio
    async def test_track_processing_restores_on_exception(self, mock_ychat):
        persona = _make_persona(mock_ychat)
        with pytest.raises(ValueError, match="boom"):
            async with persona.track_processing(object()):
                raise ValueError("boom")
        assert persona.processing is False
        assert persona.processing_message is None

    @pytest.mark.asyncio
    async def test_track_processing_publishes_processing_state(self, mock_ychat):
        # The frontend enables the stop button off this: track_processing sets
        # `state.processing` True while a message is in flight and False after.
        persona = _make_persona(mock_ychat)
        persona.state = PersonaSessionState(
            event_logger=None, persona_id="p1", chat_id="c1", log=MagicMock()
        )
        assert persona.state.processing is False
        async with persona.track_processing(object()):
            assert persona.state.processing is True
        assert persona.state.processing is False

    @pytest.mark.asyncio
    async def test_track_processing_clears_processing_on_exception(self, mock_ychat):
        persona = _make_persona(mock_ychat)
        persona.state = PersonaSessionState(
            event_logger=None, persona_id="p1", chat_id="c1", log=MagicMock()
        )
        with pytest.raises(ValueError, match="boom"):
            async with persona.track_processing(object()):
                raise ValueError("boom")
        assert persona.state.processing is False


# ---------------------------------------------------------------------------
# TestMcpIdentityHeaders
# ---------------------------------------------------------------------------

BUILTIN_URL = "http://localhost:3001/mcp"
THIRD_PARTY_URL = "http://example.com/mcp"


def _persona_with_parent(mock_ychat, mcp_settings):
    """A persona whose parent PersonaManager returns ``mcp_settings`` and
    advertises the built-in Jupyter MCP server."""
    persona = _make_persona(mock_ychat)
    parent = MagicMock()
    parent.builtin_mcp_servers = [
        {
            "type": "http",
            "name": "Jupyter MCP Server",
            "url": BUILTIN_URL,
            "headers": [],
        }
    ]
    parent.chat.get_id.return_value = "chat-XYZ"
    parent.get_mcp_settings.return_value = mcp_settings
    return persona, parent


class TestMcpIdentityHeaders:
    def test_stamps_identity_headers_on_builtin_server(self, mock_ychat):
        settings = McpSettings(
            mcp_servers=[
                McpServerHttp(
                    type="http", name="Jupyter MCP Server", url=BUILTIN_URL, headers=[]
                )
            ]
        )
        persona, parent = _persona_with_parent(mock_ychat, settings)
        from unittest.mock import patch

        with patch.object(type(persona), "parent", parent):
            result = persona.get_mcp_settings()
            expected_persona_id = persona.id

        builtin = next(s for s in result.mcp_servers if s.url == BUILTIN_URL)
        headers = {h.name: h.value for h in builtin.headers}
        assert headers["X-Jupyter-Chat-Id"] == "chat-XYZ"
        assert headers["X-JupyterAI-Persona-Id"] == expected_persona_id

    def test_stamps_all_http_servers_preserving_existing(self, mock_ychat):
        settings = McpSettings(
            mcp_servers=[
                McpServerHttp(
                    type="http", name="Jupyter MCP Server", url=BUILTIN_URL, headers=[]
                ),
                McpServerHttp(
                    type="http",
                    name="Third Party",
                    url=THIRD_PARTY_URL,
                    headers=[HttpHeader(name="X-Existing", value="keep")],
                ),
            ]
        )
        persona, parent = _persona_with_parent(mock_ychat, settings)
        from unittest.mock import patch

        with patch.object(type(persona), "parent", parent):
            result = persona.get_mcp_settings()
            expected_persona_id = persona.id

        third = next(s for s in result.mcp_servers if s.url == THIRD_PARTY_URL)
        headers = {h.name: h.value for h in third.headers}
        # Identity headers are added to every HTTP server ...
        assert headers["X-Jupyter-Chat-Id"] == "chat-XYZ"
        assert headers["X-JupyterAI-Persona-Id"] == expected_persona_id
        # ... without dropping headers the server already had.
        assert headers["X-Existing"] == "keep"

    def test_returns_none_when_no_servers(self, mock_ychat):
        persona, parent = _persona_with_parent(mock_ychat, None)
        from unittest.mock import patch

        with patch.object(type(persona), "parent", parent):
            assert persona.get_mcp_settings() is None
