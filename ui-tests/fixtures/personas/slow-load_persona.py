"""
Fixture persona for E2E tests: a `BasePersona` whose *module* is slow to import,
so the chat toolbar shows its "loading personas" placeholder while the persona
list is still resolving, then resolves to a working picker once loading finishes
(see tests/slow-load.spec.ts, issue #77).

## How this pins the loading state

The persona list the browser renders comes from the `PersonaManager`'s awareness
slot, which the manager publishes only after it has loaded and instantiated every
persona for the chat. Loading a local persona runs the module's top level
(`exec_module`), so a delay there delays the whole manager, and the toolbar stays
on its loading placeholder (`PersonaManagerAwareness.from()` polls for the slot)
until it finishes.

## Why the delay is bounded, not forever

`PersonaManager` is constructed *synchronously on the server's event loop* (a
chat-init event → `router.connect_chat` → `PersonaManager(...)`), and module
import happens inside that construction. A delay here therefore blocks the single
event loop for its duration. A *forever* stall would hang the whole shared test
server and every other suite with it (verified — it does exactly that). So this
sleeps for a bounded interval: long enough for the frontend to render the loading
placeholder and a test to observe it, short enough that the server recovers and
this suite (and any concurrent one) still finishes well within its timeout. The
test then also asserts the toolbar *resolves* to a working picker afterward.

> The eventual give-up/timeout behavior — the toolbar reporting a persona that
> never finished loading — is intentionally out of scope (issue #77 defers it
> until persona init is split into a lazy `async init_session()`, so a persona no
> longer does all its work up front during load).

Not part of the shipped package; see AGENTS.md.
"""

import os
import time

from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")

# Bounded delay applied at import time (see module docstring for why bounded).
# Long enough to reliably observe the loading placeholder; short enough that the
# shared server recovers and no suite exceeds its timeout.
_LOAD_DELAY_S = 8

# Runs during `exec_module`, i.e. while the PersonaManager is constructing and
# before it publishes its persona list — which is exactly the window the loading
# placeholder covers.
time.sleep(_LOAD_DELAY_S)


class SlowLoadPersona(BasePersona):
    """Test-only persona whose module is slow to import."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Slow Load Persona",
            description="Test-only persona whose module is slow to import.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        self.send_message("loaded")
