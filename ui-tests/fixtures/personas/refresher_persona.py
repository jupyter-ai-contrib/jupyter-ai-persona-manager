"""
Fixture persona for E2E tests: a `BasePersona` that triggers a full persona
refresh (`PersonaManager.refresh_personas()`, the public method behind the
`/refresh-personas` command) when the test signals it — so a test can force the
manager to rebuild and republish its persona list at a moment it controls.

The republish is what drives the toolbar's selection reconciliation: reloaded
personas get new awareness client IDs, so the list content genuinely changes and
the client's awareness `change` handler re-runs `reconcileSelection`. A test
uses this to prove an explicit "No one" selection survives a real list update
(the sole-persona convenience must not reseed over it).

Sequencing: a message "arms" the persona; it then polls its chat directory for a
`go-refresh` file, which the test uploads only *after* making the selection under
test. That orders selection strictly before the republish without any timing
guesswork. The poll runs in a detached task so it survives this instance's own
shutdown during the refresh.

Not part of the shipped package; see AGENTS.md.
"""

import asyncio
import os

from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")

# Poll cadence for the signal file: 0.2s * 150 = up to 30s.
_POLL_TRIES = 150
_POLL_DELAY_S = 0.2


class RefresherPersona(BasePersona):
    """Test-only persona that refreshes the persona list on a file signal."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Refresher Persona",
            description="Test-only persona that refreshes personas on signal.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        asyncio.create_task(self._refresh_when_signalled())
        self.send_message("armed")

    async def _refresh_when_signalled(self) -> None:
        signal = os.path.join(self.get_chat_dir(), "go-refresh")
        for _ in range(_POLL_TRIES):
            if os.path.exists(signal):
                # Rebuilds all personas and republishes the list (which also
                # posts the "Refreshed all AI personas" system message the test
                # waits for).
                await self.parent.refresh_personas()
                return
            await asyncio.sleep(_POLL_DELAY_S)
