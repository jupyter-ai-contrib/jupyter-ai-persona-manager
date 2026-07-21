"""
Fixture persona for E2E tests: a `BasePersona` whose constructor raises, to
verify a persona that fails to initialize degrades gracefully — the rest of the
chat toolbar still loads, and the user is told which persona failed (see
tests/broken-init.spec.ts).

The `PersonaManager` instantiates each persona class inside a `try/except
Exception`: when one raises, it logs, posts a system message naming the failed
persona, and *continues* with the others. So a single broken persona must not
take down the picker — the other personas in the chat stay usable.

This fixture is installed alongside a working persona (Hello). The test asserts
the picker still lists and routes to Hello, and that a system message naming this
persona's class appears.

Not part of the shipped package; see AGENTS.md.
"""

import os

from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")


class BrokenInitPersona(BasePersona):
    """Test-only persona that raises during initialization."""

    def __init__(self, *args, **kwargs) -> None:
        # A plain exception, the way a real persona's constructor might fail
        # (bad config, a missing dependency, a failed network call). The manager
        # catches it per-persona, so this must not stop other personas loading.
        raise RuntimeError("BrokenInitPersona failed to initialize (on purpose).")

    @property
    def defaults(self) -> PersonaDefaults:
        # Abstract on BasePersona, so it must exist for the class to be
        # instantiable at all. Never reached: `__init__` always raises first.
        return PersonaDefaults(
            name="Broken Init Persona",
            description="Test-only persona that fails to initialize.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        # Abstract on BasePersona; unreachable (see `defaults`).
        pass
