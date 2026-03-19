from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml
from pydantic import BaseModel

from .base_persona import BasePersona, PersonaDefaults

if TYPE_CHECKING:
    from jupyterlab_chat.models import Message
    from jupyterlab_chat.ychat import YChat

# Path to the default avatar SVG shipped with this package
_DEFAULT_AVATAR_PATH = str(Path(__file__).parent / "static" / "default_avatar.svg")


class MarkdownFrontmatter(BaseModel):
    """Parsed YAML frontmatter from a markdown persona file."""

    name: str
    description: str = ""
    target_persona: Optional[str] = None
    """
    Entry point name or class name of the persona class to inherit from.
    For example, ``ClaudeCodePersona`` would resolve to the
    ``ClaudeCodePersona`` entry point registered under the
    ``jupyter_ai.personas`` group.

    When omitted (``None``), the markdown persona inherits from whatever
    persona class is configured as the default persona in
    ``PersonaManager.default_persona_id``.
    """
    avatar_path: Optional[str] = None

    # Limitation: `tools` is parsed but not acted on. The codebase currently
    # has no built-in tool-use abstraction -- each Python persona brings its
    # own. A future iteration would map tool names to concrete implementations
    # and use litellm's function-calling API.
    tools: list[str] = []

    # Limitation: `mcp_servers` references are parsed but no MCP server
    # connections are established. BasePersona.get_mcp_settings() can read the
    # MCP config, but wiring MCP tools into litellm function-calling is
    # deferred.
    mcp_servers: list[str] = []

    slash_commands: set[str] = set("*")


def parse_markdown_persona(md_path: str) -> tuple[MarkdownFrontmatter, str]:
    """
    Parse a markdown persona file into its YAML frontmatter and body.

    The file must begin with a ``---`` delimiter, followed by YAML, followed
    by a closing ``---`` delimiter. Everything after the closing delimiter is
    the system prompt body (markdown).

    Args:
        md_path: Absolute path to the ``.md`` file.

    Returns:
        A tuple of ``(frontmatter, system_prompt_body)``.

    Raises:
        ValueError: If the file does not contain valid ``---`` delimiters.
        pydantic.ValidationError: If the YAML does not match the schema.
    """
    text = Path(md_path).read_text(encoding="utf-8")

    if not text.startswith("---"):
        raise ValueError(
            f"Markdown persona file must start with '---' delimiter: {md_path}"
        )

    # Find the closing delimiter (skip the first '---')
    close_idx = text.find("---", 3)
    if close_idx == -1:
        raise ValueError(
            f"Markdown persona file is missing closing '---' delimiter: {md_path}"
        )

    yaml_block = text[3:close_idx].strip()
    body = text[close_idx + 3:].strip()

    raw = yaml.safe_load(yaml_block)
    if not isinstance(raw, dict):
        raise ValueError(
            f"YAML frontmatter must be a mapping, got {type(raw).__name__}: {md_path}"
        )

    frontmatter = MarkdownFrontmatter(**raw)
    return frontmatter, body


def create_markdown_persona_class(target_class: type[BasePersona]) -> type:
    """
    Dynamically create a ``MarkdownPersona`` class that inherits from
    *target_class*.

    The returned class overrides identity and configuration properties
    (``id``, ``defaults``, ``_get_system_prompt``) so that values come from
    the ``.md`` file, while message-processing behaviour is inherited from
    *target_class*.
    """

    class MarkdownPersona(target_class):  # type: ignore[misc]
        """
        A persona defined by a markdown file with YAML frontmatter.

        The frontmatter specifies name, description, target persona, and
        optional avatar.  The markdown body becomes the system prompt.
        Message processing is inherited from the *target_class* resolved at
        load time.
        """

        def __init__(
            self,
            *args,
            md_path: str,
            ychat: "YChat",
            **kwargs,
        ):
            # Parse the markdown file *before* calling super().__init__(),
            # which accesses self.defaults (and therefore needs
            # _frontmatter / _body).
            self._md_path = md_path
            self._frontmatter, self._body = parse_markdown_persona(md_path)

            # Resolve avatar_path: relative paths are resolved against the
            # .md file's directory; absent paths use the default avatar
            # shipped with this package.
            if self._frontmatter.avatar_path is not None:
                avatar = Path(self._frontmatter.avatar_path)
                if not avatar.is_absolute():
                    avatar = Path(md_path).parent / avatar
                self._resolved_avatar_path = str(avatar.resolve())
            else:
                self._resolved_avatar_path = _DEFAULT_AVATAR_PATH

            super().__init__(*args, ychat=ychat, **kwargs)

        @property
        def id(self) -> str:
            """
            Returns a unique ID for this markdown persona.

            Uses ``md`` as the "package" segment to distinguish from Python
            personas. The filename stem ensures uniqueness within the
            directory.
            """
            return f"jupyter-ai-personas::md::{Path(self._md_path).stem}"

        @property
        def defaults(self) -> PersonaDefaults:
            return PersonaDefaults(
                name=self._frontmatter.name,
                description=self._frontmatter.description,
                avatar_path=self._resolved_avatar_path,
                system_prompt=self._body,
                slash_commands=self._frontmatter.slash_commands,
            )

        def _get_system_prompt(self) -> str:
            """Return the markdown body as the system prompt."""
            return self._body

        async def process_message(self, message: "Message") -> None:
            """
            Process a message by delegating to the target persona's
            implementation.

            If this class was created with ``BasePersona`` as the target
            (i.e. the ``target_persona`` could not be resolved), an error
            message is sent to the chat instead.
            """
            if target_class is BasePersona:
                self.send_message(
                    "This persona's `target_persona` could not be resolved "
                    "to an installed persona class. Please check the "
                    "frontmatter and ensure the referenced persona package "
                    "is installed."
                )
                return

            return await super().process_message(message)

    return MarkdownPersona


# Default concrete class backed by BasePersona.  Used as a fallback when
# ``target_persona`` cannot be resolved to an installed entry-point class.
MarkdownPersona = create_markdown_persona_class(BasePersona)
