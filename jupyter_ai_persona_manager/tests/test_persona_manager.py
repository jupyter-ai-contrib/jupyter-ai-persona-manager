"""
Test the persona manager functionality.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from jupyterlab_chat.models import Message
from jupyter_ai_persona_manager.base_persona import BasePersona
from jupyter_ai_persona_manager.persona_manager import (
    SYSTEM_USERNAME,
    PersonaManager,
    _safe_process,
    find_persona_files,
    load_from_dir,
)


@pytest.fixture
def tmp_persona_dir():
    """Create a temporary directory for testing LocalPersonaLoader with guaranteed cleanup."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    return Mock()


class TestFindPersonaFiles:
    """Test cases for find_persona_files function."""

    def test_nonexistent_directory_returns_empty_list(self):
        """Test that a non-existent directory returns an empty list."""
        result = find_persona_files("/nonexistent/directory/path")
        assert result == []

    def test_empty_directory_returns_empty_list(self, tmp_persona_dir):
        """Test that an empty directory returns an empty list."""
        result = find_persona_files(str(tmp_persona_dir))
        assert result == []

    def test_finds_valid_ignores_invalid_persona_files(self, tmp_persona_dir):
        """Test that persona files with valid filenames are found while private, hidden, and non-valid files are ignored."""
        (tmp_persona_dir / "my_persona.py").write_text("pass")
        (tmp_persona_dir / "PersonalAssistant.py").write_text("pass")

        (tmp_persona_dir / "my_other_code.py").write_text("pass")
        (tmp_persona_dir / "_private_persona.py").write_text("pass")
        (tmp_persona_dir / ".hidden_persona.py").write_text("pass")

        result = find_persona_files(str(tmp_persona_dir))
        result_names = [Path(f).name for f in result]

        assert "my_persona.py" in result_names
        assert "PersonalAssistant.py" in result_names

        assert "my_other_code.py" not in result_names
        assert "_private_persona.py" not in result_names
        assert ".hidden_persona.py" not in result_names


class TestLoadPersonaClassesFromDirectory:
    """Test cases for load_from_dir function."""

    def test_empty_directory_returns_empty_list(self, tmp_persona_dir, mock_logger):
        """Test that an empty directory returns an empty list of persona classes."""
        result = load_from_dir(str(tmp_persona_dir), mock_logger)
        assert result == []

    def test_non_persona_file_returns_empty_list(self, tmp_persona_dir, mock_logger):
        """Test that a Python file without persona classes returns an empty list."""
        # Create a file that doesn't contain "persona" in the name
        non_persona_file = tmp_persona_dir / "no_personas.py"
        non_persona_file.write_text("pass")

        result = load_from_dir(str(tmp_persona_dir), mock_logger)
        assert result == []

    def test_simple_persona_file_returns_persona_class(
        self, tmp_persona_dir, mock_logger
    ):
        """Test that a file with a BasePersona subclass returns that class."""
        # Create a simple persona file
        persona_file = tmp_persona_dir / "simple_personas.py"
        persona_content = """
from jupyter_ai_persona_manager.base_persona import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message

class TestPersona(BasePersona):
    @property
    def defaults(self):
        return PersonaDefaults(
            name="Test Persona",
            description="A simple test persona",
            avatar_path="/test/avatar.svg",
            system_prompt="Test system prompt"
        )

    async def process_message(self, message: Message):
        pass
"""
        persona_file.write_text(persona_content)

        result = load_from_dir(str(tmp_persona_dir), mock_logger)

        assert len(result) == 1
        assert result[0]["persona_class"].__name__ == "TestPersona"
        assert issubclass(result[0]["persona_class"], BasePersona)
        assert result[0]["traceback"] is None

    def test_bad_persona_file_returns_error_entry(self, tmp_persona_dir, mock_logger):
        """Test that a file with syntax errors returns an error entry."""
        # Create a file with invalid Python code
        bad_persona_file = tmp_persona_dir / "bad_persona.py"
        bad_persona_file.write_text("1/0  # This will cause a syntax error")

        result = load_from_dir(str(tmp_persona_dir), mock_logger)

        assert len(result) == 1
        assert result[0]["persona_class"] is None
        assert result[0]["traceback"] is not None
        assert "ZeroDivisionError" in result[0]["traceback"]


# ---------------------------------------------------------------------------
# TestPersonaErrorMessage
# ---------------------------------------------------------------------------


class TestPersonaErrorMessage:
    """The system message shown when a persona fails to load."""

    def test_label_prefers_class_name(self):
        """When the class loaded but its constructor raised, name the class."""

        class BrokenPersona:
            pass

        label = PersonaManager._persona_item_label(
            {"persona_class": BrokenPersona, "module": "some.module"}
        )
        assert label == "`BrokenPersona`"

    def test_label_falls_back_to_file_name(self):
        """When the module itself failed to import, name its file."""
        label = PersonaManager._persona_item_label(
            {"persona_class": None, "module": "/tmp/foo/bad_persona.py"}
        )
        assert label == "`bad_persona.py`"

    def test_label_handles_missing_source(self):
        label = PersonaManager._persona_item_label(
            {"persona_class": None, "module": None}
        )
        assert "unknown source" in label

    def test_no_message_without_traceback(self):
        """No traceback means nothing failed — send no message."""
        mgr = MagicMock(spec=PersonaManager)
        PersonaManager._display_persona_error_message(
            mgr, {"traceback": None, "persona_class": None, "module": "x"}
        )
        mgr.send_system_message.assert_not_called()

    def test_message_names_persona_and_includes_traceback(self):
        """The message names the failed persona and carries the traceback."""

        class BrokenPersona:
            pass

        mgr = MagicMock(spec=PersonaManager)
        mgr._persona_item_label = PersonaManager._persona_item_label
        PersonaManager._display_persona_error_message(
            mgr,
            {
                "traceback": "Traceback ... RuntimeError: boom",
                "persona_class": BrokenPersona,
                "module": "some.module",
            },
        )
        (body,), _ = mgr.send_system_message.call_args
        assert "`BrokenPersona`" in body
        assert "failed to initialize" in body
        assert "RuntimeError: boom" in body


# ---------------------------------------------------------------------------
# TestSafeProcess
# ---------------------------------------------------------------------------

def _make_mock_persona():
    persona = MagicMock()
    persona.name = "TestPersona"
    persona.log = MagicMock()
    persona.process_message = AsyncMock()
    persona.apply_specs_in_message = AsyncMock()
    persona.handle_uncaught_exception = AsyncMock()
    return persona


def _make_mock_message():
    return MagicMock(spec=Message)


class TestSafeProcess:

    @pytest.mark.asyncio
    async def test_calls_process_message(self):
        persona = _make_mock_persona()
        message = _make_mock_message()
        await _safe_process(persona, message)
        persona.process_message.assert_awaited_once_with(message)

    @pytest.mark.asyncio
    async def test_calls_handle_uncaught_exception_on_failure(self):
        exc = RuntimeError("test error")
        persona = _make_mock_persona()
        persona.process_message.side_effect = exc
        message = _make_mock_message()

        await _safe_process(persona, message)

        persona.handle_uncaught_exception.assert_awaited_once_with(exc)

    @pytest.mark.asyncio
    async def test_logs_error_before_handle(self):
        persona = _make_mock_persona()
        persona.process_message.side_effect = RuntimeError("fail")
        message = _make_mock_message()

        await _safe_process(persona, message)

        persona.log.error.assert_called_once()
        persona.log.exception.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_exception_propagates_on_process_message_failure(self):
        persona = _make_mock_persona()
        persona.process_message.side_effect = RuntimeError("fail")
        message = _make_mock_message()

        # Should not raise
        await _safe_process(persona, message)

    @pytest.mark.asyncio
    async def test_catches_secondary_exception_from_handle_uncaught_exception(self):
        persona = _make_mock_persona()
        persona.process_message.side_effect = RuntimeError("primary")
        persona.handle_uncaught_exception.side_effect = RuntimeError("secondary")
        message = _make_mock_message()

        # Should not raise even when handle_uncaught_exception also raises
        await _safe_process(persona, message)

        # Secondary exception logged
        assert persona.log.exception.call_count == 2

    @pytest.mark.asyncio
    async def test_handle_not_called_on_success(self):
        persona = _make_mock_persona()
        message = _make_mock_message()

        await _safe_process(persona, message)

        persona.handle_uncaught_exception.assert_not_called()


PERSONA_SENDER = "jupyter-ai-personas::pkg::Bot"
HUMAN_SENDER = "human-1"


def _routing_manager(personas=None):
    """A PersonaManager with only the state on_chat_message routing reads.

    Uses ``__new__`` to get the traitlets machinery without the heavy ``__init__``
    (which loads personas), then sets just the attributes routing touches.
    """
    pm = PersonaManager.__new__(PersonaManager)
    pm._personas = personas or {}
    pm.log = logging.getLogger("test-persona-manager")
    pm.event_loop = Mock()
    return pm


def _message(sender, metadata=None):
    message = Mock(spec=Message)
    message.sender = sender
    message.metadata = metadata
    return message


class TestOnChatMessageRouting:
    """Who a message is routed to. Patches ``_safe_process`` (the per-persona
    delivery coroutine) to assert which persona a message reaches."""

    def _route(self, pm, message):
        """Run routing with ``_safe_process`` patched; return the personas it
        was called with."""
        # Patch with a plain (non-async) mock so no coroutine is created — the
        # real `_safe_process` is a coroutine function, and we don't run it here.
        with patch(
            "jupyter_ai_persona_manager.persona_manager._safe_process",
            new=MagicMock(),
        ) as safe_process:
            pm.on_chat_message("room", message)
        return [call.args[0] for call in safe_process.call_args_list]

    def test_routes_to_the_persona_named_in_metadata(self):
        target = _make_mock_persona()
        pm = _routing_manager(personas={"p1": target})

        routed = self._route(pm, _message(HUMAN_SENDER, {"to_persona": "p1"}))

        assert routed == [target]

    def test_no_target_persona_means_no_one_is_routed_to(self):
        pm = _routing_manager(personas={"p1": _make_mock_persona()})

        routed = self._route(pm, _message(HUMAN_SENDER, metadata=None))

        assert routed == []

    def test_unknown_target_persona_means_no_one_is_routed_to(self):
        pm = _routing_manager(personas={"p1": _make_mock_persona()})

        routed = self._route(pm, _message(HUMAN_SENDER, {"to_persona": "gone"}))

        assert routed == []

    def test_routes_by_metadata_regardless_of_sender(self):
        # Routing keys only on `to_persona`; the sender is not consulted. A
        # persona addressing another persona via metadata is explicit and
        # intentional, so there is no sender-based guard.
        target = _make_mock_persona()
        pm = _routing_manager(personas={"p1": target})

        for sender in (PERSONA_SENDER, SYSTEM_USERNAME):
            routed = self._route(pm, _message(sender, {"to_persona": "p1"}))
            assert routed == [target]
