"""
Fixture persona for E2E tests: an auth-gated `BasePersona` whose auth check
flips from failing to passing when a sentinel file appears. It proves the resume
poll actually runs until ``check_auth_fn`` passes and then fires `handle_auth`
with no further user message.

The test sends a message while unauthenticated (the sign-in prompt appears and
the resume poll starts), then creates the sentinel file; the running poll
observes it and the persona posts a distinctive "signed in" message on its own.

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

# The sentinel file (relative to the server root) whose presence means the user
# has "signed in". The spec creates it via the contents API to flip auth.
SIGNAL_FILE = ".auth-resume-signal"
NOAUTH_MESSAGE = "AUTH-RESUME: please sign in to continue."
RESUMED_MESSAGE = "AUTH-RESUME: signed in, resuming."


class AuthResumePersona(BasePersona):
    """Test-only persona that authenticates once a sentinel file appears."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.auth = PersonaAuthManager(parent=self, check_auth_fn=self._signed_in)

    def _signed_in(self) -> bool:
        return os.path.exists(os.path.join(self.parent.root_dir, SIGNAL_FILE))

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Auth Resume Persona",
            description="Test-only persona that resumes after sign-in.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def prepare(self) -> None:
        # Gate startup on auth. Raises `PersonaNotAuthenticated` until the
        # sentinel exists, so `preparation_state` is NOT_AUTHED.
        await self.auth.assert_auth()

    async def handle_message_no_auth(self, message: Message) -> None:
        self.send_message(NOAUTH_MESSAGE)
        # Poll quickly so the E2E test does not wait long for the resume.
        self.auth.start_poll(interval=0.5)

    async def handle_auth(self, was_unauthenticated: bool = False) -> None:
        # Fired by the resume poll once the sentinel appears — no user message.
        # ``was_unauthenticated`` is True here because the poll only runs after a
        # message arrived while signed out.
        self.send_message(RESUMED_MESSAGE)

    async def process_message(self, message: Message) -> None:
        self.send_message("processed")
