"""
Tests for markdown-based persona definitions with YAML frontmatter.
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import ValidationError
from traitlets.config import LoggingConfigurable

from jupyter_ai_persona_manager.markdown_persona import (
    MarkdownFrontmatter,
    MarkdownPersona,
    _DEFAULT_AVATAR_PATH,
    create_markdown_persona_class,
    parse_markdown_persona,
)
from jupyter_ai_persona_manager.persona_manager import (
    find_markdown_persona_files,
    load_markdown_personas_from_dir,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_MD = """\
---
name: Test Bot
description: A test persona
target_persona: TestTargetPersona
---
You are a helpful assistant.
"""

MINIMAL_MD = """\
---
name: Minimal
target_persona: TestTargetPersona
---
"""


def _write_md(dir_path: Path, filename: str, content: str) -> Path:
    p = dir_path / filename
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# TestParseMarkdownPersona
# ---------------------------------------------------------------------------


class TestParseMarkdownPersona:
    """Tests for the parse_markdown_persona function."""

    def test_valid_parsing(self, tmp_dir):
        md_file = _write_md(tmp_dir, "bot.md", VALID_MD)
        fm, body = parse_markdown_persona(str(md_file))

        assert fm.name == "Test Bot"
        assert fm.description == "A test persona"
        assert fm.target_persona == "TestTargetPersona"
        assert body == "You are a helpful assistant."

    def test_missing_name_raises_validation_error(self, tmp_dir):
        content = """\
---
target_persona: SomePersona
---
body
"""
        md_file = _write_md(tmp_dir, "no_name.md", content)
        with pytest.raises(ValidationError):
            parse_markdown_persona(str(md_file))

    def test_missing_target_persona_defaults_to_none(self, tmp_dir):
        content = """\
---
name: No Target
---
body
"""
        md_file = _write_md(tmp_dir, "no_target.md", content)
        fm, body = parse_markdown_persona(str(md_file))
        assert fm.target_persona is None
        assert fm.name == "No Target"
        assert body == "body"

    def test_optional_fields_default(self, tmp_dir):
        md_file = _write_md(tmp_dir, "minimal.md", MINIMAL_MD)
        fm, body = parse_markdown_persona(str(md_file))

        assert fm.description == ""
        assert fm.avatar_path is None
        assert fm.tools == []
        assert fm.mcp_servers == []
        assert fm.slash_commands == {"*"}
        assert body == ""

    def test_tools_and_mcp_servers_parsed(self, tmp_dir):
        content = """\
---
name: Tooled
target_persona: SomePersona
tools:
  - read_file
  - list_files
mcp_servers:
  - my-server
---
prompt
"""
        md_file = _write_md(tmp_dir, "tooled.md", content)
        fm, _ = parse_markdown_persona(str(md_file))

        assert fm.tools == ["read_file", "list_files"]
        assert fm.mcp_servers == ["my-server"]

    def test_empty_body(self, tmp_dir):
        md_file = _write_md(tmp_dir, "empty.md", MINIMAL_MD)
        _, body = parse_markdown_persona(str(md_file))
        assert body == ""

    def test_malformed_yaml_raises(self, tmp_dir):
        content = """\
---
name: [invalid yaml
---
body
"""
        md_file = _write_md(tmp_dir, "bad_yaml.md", content)
        with pytest.raises(Exception):
            parse_markdown_persona(str(md_file))

    def test_missing_opening_delimiter_raises(self, tmp_dir):
        content = "name: No Delimiters\n---\nbody"
        md_file = _write_md(tmp_dir, "no_open.md", content)
        with pytest.raises(ValueError, match="must start with '---'"):
            parse_markdown_persona(str(md_file))

    def test_missing_closing_delimiter_raises(self, tmp_dir):
        content = "---\nname: No Close\n"
        md_file = _write_md(tmp_dir, "no_close.md", content)
        with pytest.raises(ValueError, match="missing closing '---'"):
            parse_markdown_persona(str(md_file))

    def test_non_mapping_yaml_raises(self, tmp_dir):
        content = """\
---
- just
- a
- list
---
body
"""
        md_file = _write_md(tmp_dir, "list_yaml.md", content)
        with pytest.raises(ValueError, match="must be a mapping"):
            parse_markdown_persona(str(md_file))


# ---------------------------------------------------------------------------
# TestFindMarkdownPersonaFiles
# ---------------------------------------------------------------------------


class TestFindMarkdownPersonaFiles:
    """Tests for find_markdown_persona_files function."""

    def test_empty_directory(self, tmp_dir):
        result = find_markdown_persona_files(str(tmp_dir))
        assert result == []

    def test_finds_md_files(self, tmp_dir):
        _write_md(tmp_dir, "helper.md", VALID_MD)
        _write_md(tmp_dir, "coder.md", VALID_MD)

        result = find_markdown_persona_files(str(tmp_dir))
        names = [Path(f).name for f in result]

        assert "helper.md" in names
        assert "coder.md" in names

    def test_skips_underscore_and_dot_prefixed(self, tmp_dir):
        _write_md(tmp_dir, "_draft.md", VALID_MD)
        _write_md(tmp_dir, ".hidden.md", VALID_MD)
        _write_md(tmp_dir, "visible.md", VALID_MD)

        result = find_markdown_persona_files(str(tmp_dir))
        names = [Path(f).name for f in result]

        assert "visible.md" in names
        assert "_draft.md" not in names
        assert ".hidden.md" not in names

    def test_nonexistent_directory(self):
        result = find_markdown_persona_files("/nonexistent/path")
        assert result == []


# ---------------------------------------------------------------------------
# TestLoadMarkdownPersonasFromDir
# ---------------------------------------------------------------------------


class TestLoadMarkdownPersonasFromDir:
    """Tests for load_markdown_personas_from_dir function."""

    def test_valid_md_returns_entry(self, tmp_dir, mock_logger):
        _write_md(tmp_dir, "helper.md", VALID_MD)

        result = load_markdown_personas_from_dir(str(tmp_dir), mock_logger)

        assert len(result) == 1
        # Without ep_persona_classes, falls back to MarkdownPersona(BasePersona)
        assert result[0]["persona_class"] is MarkdownPersona
        assert result[0]["traceback"] is None
        assert result[0]["kwargs"]["md_path"].endswith("helper.md")

    def test_resolved_target_uses_dynamic_class(self, tmp_dir, mock_logger):
        """When target_persona matches an entry point, a dynamic class is created."""
        from jupyter_ai_persona_manager.base_persona import BasePersona

        class FakeTargetPersona(BasePersona):
            @property
            def defaults(self):
                return None

            async def process_message(self, message):
                pass

        _write_md(tmp_dir, "helper.md", VALID_MD)
        ep_classes = [
            {"module": "TestTargetPersona", "persona_class": FakeTargetPersona, "traceback": None}
        ]

        result = load_markdown_personas_from_dir(str(tmp_dir), mock_logger, ep_persona_classes=ep_classes)

        assert len(result) == 1
        assert result[0]["persona_class"] is not MarkdownPersona
        assert issubclass(result[0]["persona_class"], FakeTargetPersona)
        assert result[0]["traceback"] is None

    def test_invalid_md_returns_error_entry(self, tmp_dir, mock_logger):
        bad_content = """\
---
description: missing name and target_persona
---
body
"""
        _write_md(tmp_dir, "bad.md", bad_content)

        result = load_markdown_personas_from_dir(str(tmp_dir), mock_logger)

        assert len(result) == 1
        assert result[0]["persona_class"] is None
        assert result[0]["traceback"] is not None

    def test_no_target_persona_uses_default_persona_id(self, tmp_dir, mock_logger):
        """When target_persona is omitted, the default persona class is used."""
        from jupyter_ai_persona_manager.base_persona import BasePersona

        class DefaultTargetPersona(BasePersona):
            @property
            def defaults(self):
                return None

            async def process_message(self, message):
                pass

        no_target_md = """\
---
name: No Target Bot
description: A bot with no explicit target
---
You are a helpful bot.
"""
        _write_md(tmp_dir, "no_target.md", no_target_md)
        ep_classes = [
            {"module": "some_ep", "persona_class": DefaultTargetPersona, "traceback": None}
        ]

        result = load_markdown_personas_from_dir(
            str(tmp_dir),
            mock_logger,
            ep_persona_classes=ep_classes,
            default_persona_id="jupyter-ai-personas::some_pkg::DefaultTargetPersona",
        )

        assert len(result) == 1
        assert result[0]["persona_class"] is not MarkdownPersona
        assert issubclass(result[0]["persona_class"], DefaultTargetPersona)
        assert result[0]["traceback"] is None

    def test_no_target_persona_no_default_falls_back_to_base(self, tmp_dir, mock_logger):
        """When target_persona is omitted and no default_persona_id, falls back to BasePersona."""
        no_target_md = """\
---
name: Fallback Bot
---
prompt
"""
        _write_md(tmp_dir, "fallback.md", no_target_md)

        result = load_markdown_personas_from_dir(str(tmp_dir), mock_logger)

        assert len(result) == 1
        assert result[0]["persona_class"] is MarkdownPersona
        assert result[0]["traceback"] is None

    def test_empty_dir_returns_empty_list(self, tmp_dir, mock_logger):
        result = load_markdown_personas_from_dir(str(tmp_dir), mock_logger)
        assert result == []


# ---------------------------------------------------------------------------
# TestMarkdownPersona
# ---------------------------------------------------------------------------


class TestMarkdownPersona:
    """Tests for the MarkdownPersona class."""

    def _make_persona(self, tmp_dir, mock_ychat, content=VALID_MD, filename="bot.md"):
        md_file = _write_md(tmp_dir, filename, content)
        parent = LoggingConfigurable()
        parent.base_url = "/"  # type: ignore[attr-defined]
        with patch(
            "jupyter_ai_persona_manager.persona_awareness.asyncio.create_task"
        ):
            persona = MarkdownPersona(
                parent=parent,
                ychat=mock_ychat,
                md_path=str(md_file),
            )
        return persona

    def test_id_derived_from_filename(self, tmp_dir, mock_ychat):
        persona = self._make_persona(tmp_dir, mock_ychat, filename="my_helper.md")
        assert persona.id == "jupyter-ai-personas::md::my_helper"

    def test_defaults_property(self, tmp_dir, mock_ychat):
        persona = self._make_persona(tmp_dir, mock_ychat)
        defaults = persona.defaults

        assert defaults.name == "Test Bot"
        assert defaults.description == "A test persona"
        assert defaults.system_prompt == "You are a helpful assistant."

    def test_avatar_path_default(self, tmp_dir, mock_ychat):
        """When avatar_path is omitted, the default avatar is used."""
        persona = self._make_persona(tmp_dir, mock_ychat)
        assert persona.defaults.avatar_path == _DEFAULT_AVATAR_PATH

    def test_avatar_path_relative(self, tmp_dir, mock_ychat):
        """Relative avatar_path is resolved against the .md file's directory."""
        # Create a dummy avatar file
        avatar_file = tmp_dir / "icon.svg"
        avatar_file.write_text("<svg/>")

        content = """\
---
name: Bot
target_persona: TestTargetPersona
avatar_path: icon.svg
---
prompt
"""
        persona = self._make_persona(tmp_dir, mock_ychat, content=content)
        assert persona.defaults.avatar_path == str(avatar_file.resolve())

    def test_avatar_path_absolute(self, tmp_dir, mock_ychat):
        """Absolute avatar_path is used as-is."""
        content = f"""\
---
name: Bot
target_persona: TestTargetPersona
avatar_path: /absolute/path/icon.svg
---
prompt
"""
        persona = self._make_persona(tmp_dir, mock_ychat, content=content)
        assert persona.defaults.avatar_path == "/absolute/path/icon.svg"

    @pytest.mark.asyncio
    async def test_get_system_prompt_returns_body(self, tmp_dir, mock_ychat):
        """_get_system_prompt returns the markdown body."""
        persona = self._make_persona(tmp_dir, mock_ychat)
        assert persona._get_system_prompt() == "You are a helpful assistant."


# ---------------------------------------------------------------------------
# TestCreateMarkdownPersonaClass
# ---------------------------------------------------------------------------


class TestCreateMarkdownPersonaClass:
    """Tests for the create_markdown_persona_class factory."""

    def test_inherits_from_target(self, tmp_dir, mock_ychat):
        """The created class should be a subclass of the target class."""
        from jupyter_ai_persona_manager.base_persona import BasePersona

        DynamicClass = create_markdown_persona_class(BasePersona)
        assert issubclass(DynamicClass, BasePersona)

    def test_factory_creates_distinct_classes(self):
        """Each call to the factory should create a distinct class."""
        from jupyter_ai_persona_manager.base_persona import BasePersona

        ClassA = create_markdown_persona_class(BasePersona)
        ClassB = create_markdown_persona_class(BasePersona)
        # Same name but different class objects
        assert ClassA is not ClassB

    def test_process_message_inherited_from_target(self, tmp_dir, mock_ychat):
        """process_message should come from the target class, not BasePersona."""
        from jupyter_ai_persona_manager.base_persona import BasePersona

        class FakePersona(BasePersona):
            """A fake persona class to test inheritance."""

            _process_called = False

            @property
            def defaults(self):
                return None  # Not used in this test

            async def process_message(self, message):
                FakePersona._process_called = True

        DynamicClass = create_markdown_persona_class(FakePersona)
        assert issubclass(DynamicClass, FakePersona)
