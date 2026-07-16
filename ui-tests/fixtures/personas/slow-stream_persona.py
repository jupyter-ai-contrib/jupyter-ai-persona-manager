"""
Fixture persona for E2E tests: a `BasePersona` that streams a long reply slowly
— one chunk every ~200ms for up to ~60s — so a test has a comfortable window to
interrupt it mid-stream with the toolbar's stop button.

It counts up ("1 2 3 …"), one number per streamed chunk, via
`BasePersona.stream_message` (which manages the `is_writing` awareness flag that
drives the chat's writers list, and hence the stop button's enabled state).
Between chunks it checks a cancel flag set by `cancel_response`, the override the
PersonaManager's cancel endpoint invokes — so when the user stops the reply, the
loop halts promptly and no further chunks land. That lets a test assert the
message stops growing and the persona stops writing.

Not part of the shipped package; see AGENTS.md.
"""

import asyncio
import os
from collections.abc import AsyncIterator

from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")

# Stream cadence: total ~= CHUNKS * DELAY seconds (300 * 0.2s = 60s).
_CHUNKS = 300
_DELAY_S = 0.2


class SlowStreamPersona(BasePersona):
    """Test-only persona that streams a long, cancellable reply."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cancelled = False

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Slow Stream Persona",
            description="Test-only persona that streams a long, cancellable reply.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def _count(self) -> AsyncIterator[str]:
        for i in range(1, _CHUNKS + 1):
            if self._cancelled:
                return
            yield f"{i} "
            await asyncio.sleep(_DELAY_S)

    async def process_message(self, message: Message) -> None:
        self._cancelled = False
        await self.stream_message(self._count())

    async def cancel_response(self) -> None:
        # Invoked by the PersonaManager cancel endpoint (only while processing).
        # Setting the flag stops the generator before its next chunk, so
        # stream_message unwinds and clears the is_writing awareness flag.
        self._cancelled = True
