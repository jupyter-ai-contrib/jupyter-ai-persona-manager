"""
Fixture persona for E2E tests: a `BasePersona` that gates `prepare()` on
authentication and is *never* authenticated — a stand-in for an agent like Kiro
that requires sign-in, without depending on any external CLI.

It proves the auth lifecycle contract: selecting the persona eagerly runs
`prepare()`, which fails the auth check, but that must stay **silent** — no
sign-in prompt appears until the user actually sends a message. Sending a
message while unauthenticated is what triggers `handle_message_no_auth`.

See tests/auth-gated.spec.ts. Not part of the shipped package; see AGENTS.md.
"""

import os

from jupyter_ai_persona_manager import (
    BasePersona,
    PersonaAuthManager,
    PersonaDefaults,
)
from jupyterlab_chat.models import Message

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")

# A distinctive marker the spec asserts on, so the sign-in prompt is
# unambiguous in the rendered chat.
NOAUTH_MESSAGE = "AUTH-GATED: please sign in to continue."


class AuthGatedPersona(BasePersona):
    """Test-only persona that is never authenticated."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Never authenticated. No `check_auth_fn` toggle is needed: this fixture
        # exists to prove selection stays silent and a message prompts — not to
        # exercise the resume-after-sign-in path (covered by Python tests).
        self.auth = PersonaAuthManager(parent=self, check_auth_fn=lambda: False)

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Auth Gated Persona",
            description="Test-only persona that requires sign-in.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def prepare(self) -> None:
        # Gate startup on auth, like an ACP persona. Raises
        # `PersonaNotAuthenticated`, so `preparation_state` becomes NOT_AUTHED.
        await self.auth.assert_auth()

    async def handle_message_no_auth(self, message: Message) -> None:
        # Only reached when a message arrives while unauthenticated — never on
        # mere selection.
        self.send_message(NOAUTH_MESSAGE)

    async def process_message(self, message: Message) -> None:
        # Unreachable here (the persona never authenticates), but required by
        # `BasePersona`.
        self.send_message("processed")
