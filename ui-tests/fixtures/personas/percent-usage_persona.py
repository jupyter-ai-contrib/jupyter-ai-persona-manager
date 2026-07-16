"""
Fixture persona for E2E tests: a `BasePersona` that reports context usage as a
bare percentage only — no token counts, no cost (the shape agents like kiro-cli
report). The toolbar's usage chip renders a context ring + percent, and its
popover shows the percent alone: no "X of Y" token counts, no session-token
breakdown, no cost.

The number is fixed so the UI text is deterministic:

    context_percent 42 -> ring + "42%", popover "Context 42%"

Usage is advertised statically in `__init__`, so the chip is present as soon as
the persona is selected. Not part of the shipped package; see AGENTS.md.
"""

import os

from jupyter_ai_persona_manager import BasePersona, PersonaDefaults, Usage
from jupyterlab_chat.models import Message

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")

_USAGE = Usage(context_percent=42.0)


class PercentUsagePersona(BasePersona):
    """Test-only persona that reports percent-only context usage."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.report_usage(_USAGE.model_copy(deep=True))

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Percent Usage Persona",
            description="Test-only persona that reports percent-only usage.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        self.report_usage(_USAGE.model_copy(deep=True))
        self.send_message("usage reported")
