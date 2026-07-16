"""
Fixture persona for E2E tests: a minimal `BasePersona` that always replies
"hello" and advertises no configuration.

This file is not part of the shipped package. A suite that requests the `hello`
fixture installs it at runtime: its `beforeAll` calls `installPersonas` (see
`tests/test-helpers.ts`), which uploads this file to the suite's
`<dir>/.jupyter/personas/`, where the PersonaManager auto-loads any
`*persona*.py`. Because the loader only keeps classes defined in this module
(`obj.__module__ == module stem`), the persona class must be declared here
rather than imported.

The avatar is the shared asset located via `JAI_TEST_ASSETS_DIR`, exported by
the server config, so this file works regardless of where it is copied. This is
a plain `BasePersona` — no ACP, no subprocess, no model.
"""

import os

from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")


class HelloPersona(BasePersona):
    """Test-only persona that always replies 'hello'."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Hello Persona",
            description="Test-only persona that always replies 'hello'.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        self.send_message("hello")
