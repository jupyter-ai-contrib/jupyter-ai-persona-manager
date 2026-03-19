"""
Test the persona manager functionality.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from traitlets.config import LoggingConfigurable

from jupyter_ai_persona_manager.base_persona import BasePersona, PersonaDefaults
from jupyter_ai_persona_manager.persona_manager import (
    PersonaManager,
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


class TestInitPersonasKwargs:
    """Test that _init_personas passes extra kwargs to persona constructors."""

    def test_kwargs_passed_to_persona_constructor(self, mock_ychat):
        """Verify that 'kwargs' from persona class entries are forwarded."""

        received_kwargs = {}

        class StubPersona(BasePersona):
            def __init__(self, *args, extra_arg=None, **kwargs):
                received_kwargs["extra_arg"] = extra_arg
                super().__init__(*args, **kwargs)

            @property
            def defaults(self):
                return PersonaDefaults(
                    name="Stub",
                    description="stub",
                    avatar_path="/stub.svg",
                    system_prompt="stub",
                )

            async def process_message(self, message):
                pass

        manager = LoggingConfigurable.__new__(PersonaManager)
        LoggingConfigurable.__init__(manager)
        manager.ychat = mock_ychat
        manager.log = Mock()
        manager._local_persona_classes = None

        # Inject persona classes directly, bypassing entry point loading
        PersonaManager._ep_persona_classes = [
            {
                "module": "test",
                "persona_class": StubPersona,
                "traceback": None,
                "kwargs": {"extra_arg": "hello"},
            }
        ]

        try:
            # Patch asyncio.create_task to avoid "no running event loop"
            # from PersonaAwareness heartbeat initialization
            with patch(
                "jupyter_ai_persona_manager.persona_awareness.asyncio.create_task"
            ):
                personas = manager._init_personas()
            assert received_kwargs["extra_arg"] == "hello"
            assert len(personas) == 1
        finally:
            PersonaManager._ep_persona_classes = None