"""
Fixture persona for E2E tests: a `BasePersona` that, on any message, posts a
*system* message to the chat via `PersonaManager.send_system_message()` (reached
through `self.parent`, as the Refresher fixture does).

This exercises the system-message path end to end: the PersonaManager registers
a single "System" user at initialization (with ``bot=True`` so Jupyter Chat
hides it from the ``@``-mention menu), and `send_system_message` posts a message
attributed to that user. A test drives this persona and asserts the system
message renders in the chat.

Not part of the shipped package; see AGENTS.md.
"""

import os

from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")

SYSTEM_TEXT = "System check: all systems nominal."


class SystemMessagePersona(BasePersona):
    """Test-only persona that posts a system message on any input."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="System Message Persona",
            description="Test-only persona that posts a system message.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        self.parent.send_system_message(SYSTEM_TEXT)
