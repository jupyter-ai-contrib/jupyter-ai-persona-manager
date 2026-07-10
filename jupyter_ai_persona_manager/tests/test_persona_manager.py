"""
Test the persona manager functionality.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

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
# TestSafeProcess
# ---------------------------------------------------------------------------

def _make_mock_persona():
    persona = MagicMock()
    persona.name = "TestPersona"
    persona.log = MagicMock()
    persona.process_message = AsyncMock()
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


def _routing_manager(active_persona=None, personas=None):
    """A PersonaManager with only the state on_chat_message routing reads.

    Uses ``__new__`` to get the traitlets machinery without the heavy ``__init__``
    (which loads personas), then sets just the attributes routing touches.
    """
    pm = PersonaManager.__new__(PersonaManager)
    pm.active_persona = active_persona
    pm._personas = personas or {}
    pm.ychat = Mock()
    pm._broadcast = Mock()
    return pm


def _message(sender, metadata=None):
    message = Mock(spec=Message)
    message.sender = sender
    message.metadata = metadata
    return message


def _replied_to(pm):
    """The personas on_chat_message decided should reply."""
    return pm._broadcast.call_args.kwargs["to_personas"]


class TestOnChatMessageRouting:
    """Who replies to a message. Asserts on the routing decision (the persona
    list handed to delivery), not on how delivery happens."""

    def test_routes_to_the_persona_named_in_metadata(self):
        target = _make_mock_persona()
        active = _make_mock_persona()
        pm = _routing_manager(active_persona=active, personas={"p1": target})

        pm.on_chat_message("room", _message(HUMAN_SENDER, {"persona": "p1"}))

        assert _replied_to(pm) == [target]

    def test_falls_back_to_active_persona_when_metadata_has_no_persona(self):
        active = _make_mock_persona()
        pm = _routing_manager(active_persona=active)

        pm.on_chat_message("room", _message(HUMAN_SENDER, metadata=None))

        assert _replied_to(pm) == [active]

    def test_falls_back_to_active_persona_when_named_persona_is_unknown(self):
        active = _make_mock_persona()
        pm = _routing_manager(active_persona=active, personas={})

        pm.on_chat_message("room", _message(HUMAN_SENDER, {"persona": "gone"}))

        assert _replied_to(pm) == [active]

    def test_no_active_persona_and_no_metadata_means_no_one_replies(self):
        pm = _routing_manager(active_persona=None)

        pm.on_chat_message("room", _message(HUMAN_SENDER, metadata=None))

        assert _replied_to(pm) == []

    def test_metadata_persona_wins_over_active_persona(self):
        target = _make_mock_persona()
        active = _make_mock_persona()
        pm = _routing_manager(active_persona=active, personas={"p1": target})

        pm.on_chat_message("room", _message(HUMAN_SENDER, {"persona": "p1"}))

        assert _replied_to(pm) == [target]
        # Routing by metadata does not mutate the active persona.
        assert pm.active_persona is active

    def test_persona_sender_is_ignored(self):
        pm = _routing_manager(
            active_persona=_make_mock_persona(),
            personas={"p1": _make_mock_persona()},
        )

        pm.on_chat_message("room", _message(PERSONA_SENDER, {"persona": "p1"}))

        pm._broadcast.assert_not_called()

    def test_system_sender_is_ignored(self):
        pm = _routing_manager(
            active_persona=_make_mock_persona(),
            personas={"p1": _make_mock_persona()},
        )

        pm.on_chat_message("room", _message(SYSTEM_USERNAME, {"persona": "p1"}))

        pm._broadcast.assert_not_called()


class TestGetTargetPersona:
    """Reading the addressed persona from a message's metadata."""

    def test_returns_the_persona_named_in_metadata(self):
        target = _make_mock_persona()
        pm = _routing_manager(personas={"p1": target})

        assert pm.get_target_persona(_message(HUMAN_SENDER, {"persona": "p1"})) is target

    def test_returns_none_when_metadata_is_missing(self):
        pm = _routing_manager(personas={"p1": _make_mock_persona()})

        assert pm.get_target_persona(_message(HUMAN_SENDER, metadata=None)) is None

    def test_returns_none_when_metadata_has_no_persona(self):
        pm = _routing_manager(personas={"p1": _make_mock_persona()})

        assert pm.get_target_persona(_message(HUMAN_SENDER, {"other": "x"})) is None

    def test_returns_none_when_named_persona_is_unknown(self):
        pm = _routing_manager(personas={})

        assert pm.get_target_persona(_message(HUMAN_SENDER, {"persona": "gone"})) is None


class TestActivePersonaResolution:
    """How the active persona is resolved at startup (_init_active_persona)."""

    def _manager(self, *, metadata, personas, default_id=""):
        pm = PersonaManager.__new__(PersonaManager)
        pm.default_persona_id = default_id
        pm.room_id = "room"
        pm._personas = personas
        pm.ychat = Mock()
        pm.ychat.get_metadata.return_value = metadata
        return pm

    def test_uses_the_stored_active_persona_when_it_still_exists(self):
        stored = _make_mock_persona()
        pm = self._manager(metadata={"active_persona_id": "p1"}, personas={"p1": stored})

        assert pm._init_active_persona() is stored

    def test_empty_stored_value_means_no_one(self):
        pm = self._manager(
            metadata={"active_persona_id": ""}, personas={"p1": _make_mock_persona()}
        )

        assert pm._init_active_persona() is None

    def test_falls_back_to_default_when_the_stored_persona_is_gone(self):
        default = _make_mock_persona()
        pm = self._manager(
            metadata={"active_persona_id": "gone"},
            personas={"d": default},
            default_id="d",
        )

        assert pm._init_active_persona() is default

    def test_uses_the_default_persona_when_nothing_is_stored(self):
        default = _make_mock_persona()
        pm = self._manager(metadata={}, personas={"d": default}, default_id="d")

        assert pm._init_active_persona() is default

    def test_uses_the_sole_persona_when_no_default_and_only_one(self):
        only = _make_mock_persona()
        pm = self._manager(metadata={}, personas={"only": only})

        assert pm._init_active_persona() is only

    def test_no_one_when_no_default_and_several_personas(self):
        pm = self._manager(
            metadata={},
            personas={"a": _make_mock_persona(), "b": _make_mock_persona()},
        )

        assert pm._init_active_persona() is None
