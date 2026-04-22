"""Tests for BasePersona.handle_uncaught_exception() and stream_message() re-raise."""

from unittest.mock import MagicMock

import pytest

from jupyter_ai_persona_manager.base_persona import BasePersona, PersonaDefaults


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
    persona.ychat = mock_ychat
    persona.log = MagicMock()
    persona.awareness = MagicMock()
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

        persona.awareness.set_local_state_field.assert_called_with("isWriting", False)
