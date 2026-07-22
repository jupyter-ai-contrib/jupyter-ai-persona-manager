"""
Shared fixture persona for the `/restart` E2E tests, used by BOTH
jupyter-ai-persona-manager and jupyter-ai-chat-commands.

It makes a persona's lifecycle observable in the chat: it posts a **start**
message every time it is constructed and a **stop** message every time it is
shut down. A restart tears one instance down and builds a fresh one under the
same ID, so a restart shows up in the chat as a stop message followed by a new
start message — which is exactly what both suites assert.

Two ways to trigger a restart, one per suite:

- **persona-manager** drives the API directly: send this persona a message whose
  body is exactly ``restart`` and its ``process_message`` calls
  ``self.restart()`` (the public `BasePersona` method behind `/restart`). The
  restart runs in a detached task so it survives this instance's own shutdown.
- **chat-commands** drives the command path: send ``/restart`` routed to this
  persona; the chat-commands server extension reads ``metadata['to_persona']``
  and calls ``PersonaManager.restart_persona()`` on it. This fixture needs no
  special handling for that path — the stop/start messages fall out of the
  lifecycle hooks.

Keep this file identical in both repos' ``ui-tests/fixtures/personas/`` (each
suite installs its own copy at runtime — see AGENTS.md). Not part of either
shipped package. The avatar is the shared asset located via
``JAI_TEST_ASSETS_DIR``, exported by the server config.
"""

import asyncio
import os

from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")

# Stable markers the specs assert on. Kept plain-ASCII and distinct so a spec can
# match them with a substring check without tripping over each other.
START_MARKER = "lifecycle:started"
STOP_MARKER = "lifecycle:stopped"


class LifecyclePersona(BasePersona):
    """Test-only persona that announces its own start and stop in the chat."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Announce startup. Guarded so a hiccup posting the message never breaks
        # persona loading (which would fail the whole suite).
        try:
            self.send_message(START_MARKER)
        except Exception:
            self.log.exception("LifecyclePersona failed to post its start message.")

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Lifecycle Persona",
            description="Test-only persona that announces start and stop.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        # The persona-manager suite triggers a restart via the API by sending the
        # literal message "restart". Run it detached so it outlives this
        # instance's shutdown (restart() shuts `self` down and replaces it).
        if message.body.strip() == "restart":
            asyncio.create_task(self.restart())
            return
        # Any other message just gets an acknowledgement, so a plain reply path
        # still works if a spec needs it.
        self.send_message("ack")

    async def shutdown(self) -> None:
        # Announce shutdown before tearing the awareness slot down.
        try:
            self.send_message(STOP_MARKER)
        except Exception:
            self.log.exception("LifecyclePersona failed to post its stop message.")
        await super().shutdown()
