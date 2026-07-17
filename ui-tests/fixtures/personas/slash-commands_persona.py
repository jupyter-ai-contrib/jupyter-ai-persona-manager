"""
Fixture persona for E2E tests: a `BasePersona` that advertises a fixed set of
slash commands over awareness. The client reads that list from the selected
persona's awareness slot and offers it as chat-input completions when the user
types `/`, filtered by what they've typed so far.

The command list is fixed so the test can assert it exactly:

    /compact, /clear, /help

Advertised statically in `__init__`, so the completions are available as soon as
the persona is selected. Not part of the shipped package; see AGENTS.md.
"""

import os

from jupyter_ai_persona_manager import BasePersona, CommandOption, PersonaDefaults
from jupyterlab_chat.models import Message

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")

_COMMANDS = [
    CommandOption(name="/compact", description="Compact the conversation context"),
    CommandOption(name="/clear", description="Clear the conversation history"),
    CommandOption(name="/help", description="Show available commands"),
]


class SlashCommandsPersona(BasePersona):
    """Test-only persona that advertises slash commands."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.report_slash_commands([c.model_copy(deep=True) for c in _COMMANDS])

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Slash Commands Persona",
            description="Test-only persona that advertises slash commands.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        self.send_message("ok")
