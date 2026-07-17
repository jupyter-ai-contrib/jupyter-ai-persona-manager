"""
Fixture persona for E2E tests: a `BasePersona` that reports token-based usage —
a context-window snapshot, cumulative session token counts, and a session cost.
The toolbar's usage chip renders a context ring + percent, and its popover lists
the token breakdown and cost.

Numbers are fixed so the expected UI text is deterministic (the client formats
token counts compactly via `formatTokens`, so the tests assert the compact
form):

    context 1200 / 4000        -> ring + "30%", popover "1.2k of 4k (30%)"
    total_tokens 1500          -> popover "Session tokens: 1.5k"
    input 1000 / output 500    -> popover Input "1k" / Output "500"
    cost 0.42 USD              -> popover "$0.42"

Usage is advertised statically in `__init__` (deterministic, mirrors how
acp-client's fixtures advertise), so the chip is present as soon as the persona
is selected — no message needed. Not part of the shipped package; see AGENTS.md.
"""

import os

from jupyter_ai_persona_manager import BasePersona, PersonaDefaults, Usage
from jupyterlab_chat.models import Message

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")

_USAGE = Usage(
    context_tokens=1200,
    context_size=4000,
    input_tokens=1000,
    output_tokens=500,
    total_tokens=1500,
    cost_amount=0.42,
    cost_currency="USD",
)


class UsagePersona(BasePersona):
    """Test-only persona that reports token-based usage."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.report_usage(_USAGE.model_copy(deep=True))

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Usage Persona",
            description="Test-only persona that reports token-based usage.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        # Re-report on each turn too (a real persona updates usage as it works);
        # replace-mode keeps the numbers fixed and deterministic.
        self.report_usage(_USAGE.model_copy(deep=True))
        self.send_message("usage reported")
