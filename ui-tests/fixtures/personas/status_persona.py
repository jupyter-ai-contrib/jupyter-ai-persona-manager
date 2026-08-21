"""
Fixture persona for E2E tests: a `BasePersona` that walks its status indicator
through the `set_status` / `clear_status` API on any message.

On input it: sets the default status ("is typing..."), waits, sets a custom
status ("is thinking..."), waits, then clears it. A test asserts the chat's
writing indicator reflects each step. This exercises the status path in the
real (RTC and non-RTC) transports, not just against a mock.

Not part of the shipped package; see AGENTS.md.
"""

import asyncio
import os

from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")

# Dwell on each status long enough for the browser to observe it. Kept as a
# constant so the spec can reason about the sequence timing.
_DWELL_S = 2.0


class StatusPersona(BasePersona):
    """Test-only persona that steps through set_status/clear_status."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Status Persona",
            description="Test-only persona that steps through its status.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        # Default status.
        self.set_status()
        await asyncio.sleep(_DWELL_S)
        # Caller-set status.
        self.set_status("is thinking...")
        await asyncio.sleep(_DWELL_S)
        # Cleared.
        self.clear_status()
