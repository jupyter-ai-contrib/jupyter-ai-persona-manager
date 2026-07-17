"""
Fixture persona for E2E tests: a `BasePersona` that reports context fill as a
percentage plus a session cost metered in a vendor unit ("credits") rather than
an ISO currency — the shape kiro-cli reports. The usage chip shows a ring +
percent; the popover shows the cost with the unit name after the amount
("0.09 credits"), not a currency symbol.

The numbers are fixed so the UI text is deterministic:

    context_percent 12                -> ring + "12%"
    cost 0.09 "credits"               -> popover "0.09 credits"

Usage is advertised statically in `__init__`, so the chip is present as soon as
the persona is selected. Not part of the shipped package; see AGENTS.md.
"""

import os

from jupyter_ai_persona_manager import BasePersona, PersonaDefaults, Usage
from jupyterlab_chat.models import Message

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")

_USAGE = Usage(context_percent=12.0, cost_amount=0.09, cost_currency="credits")


class CreditsUsagePersona(BasePersona):
    """Test-only persona that reports a cost in a non-currency unit."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.report_usage(_USAGE.model_copy(deep=True))

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Credits Usage Persona",
            description="Test-only persona that reports cost in credits.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        self.report_usage(_USAGE.model_copy(deep=True))
        self.send_message("usage reported")
