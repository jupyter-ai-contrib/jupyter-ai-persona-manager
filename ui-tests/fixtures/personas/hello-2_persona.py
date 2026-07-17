"""
Fixture persona for E2E tests: a second minimal `BasePersona`, replying with a
distinct word ("bonjour") so a test can tell which persona answered. Used by
picker.spec.ts to verify that switching the picker routes each message to the
selected persona.

Not part of the shipped package. A suite that requests the `hello-2` fixture
installs it at runtime: its `beforeAll` calls `installPersonas` (see
`tests/test-helpers.ts`), which uploads this file to the suite's
`<dir>/.jupyter/personas/`, where the PersonaManager auto-loads any
`*persona*.py`. Because the loader only keeps classes defined in this module
(`obj.__module__ == module stem`), the persona class must be declared here
rather than imported.
"""

import os

from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")


class Hello2Persona(BasePersona):
    """Test-only persona that always replies 'bonjour'."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Bonjour Persona",
            description="Test-only persona that always replies 'bonjour'.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        self.send_message("bonjour")
