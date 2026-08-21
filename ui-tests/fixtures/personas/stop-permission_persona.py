"""
E2E fixture: a persona that requests permission while marked as "writing", so
the toolbar stop button is enabled during the wait. When the user clicks stop,
the persona-manager cancel flow cancels the pending permission request; the
awaiting `request_permission` returns cancelled and the persona reports it.

Not part of the shipped package; see AGENTS.md.
"""

import os
import uuid

from jupyter_ai_persona_manager import (
    BasePersona,
    PermissionOption,
    PermissionRequest,
    PersonaDefaults,
)
from jupyterlab_chat.models import Message, NewMessage

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")


def _ensure_chat_id(chat) -> None:
    """Bridging shim until chat.get_id() is guaranteed unique (test-only)."""
    if chat.get_id():
        return
    new_id = uuid.uuid4().hex
    if hasattr(chat, "set_id"):
        chat.set_id(new_id)
    else:
        try:
            chat._metadata["id"] = new_id
        except Exception:
            pass


class StopPermissionPersona(BasePersona):
    """Requests permission while 'writing' so the stop button can cancel it."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Stop Requester Persona",
            description="Requests permission while writing; stop cancels it.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        _ensure_chat_id(self.chat)
        message_id = self.chat.add_message(
            NewMessage(body="", sender=self.id), trigger_actions=[]
        )
        # Mark writing so the toolbar stop button is enabled during the wait.
        self.set_writing_status(message_id)
        try:
            outcome = await self.request_permission(
                PermissionRequest(
                    title="Approve action?",
                    message_id=message_id,
                    options=[
                        PermissionOption(
                            option_id="allow", name="Allow", kind="allow_once"
                        ),
                        PermissionOption(
                            option_id="deny", name="Deny", kind="reject_once"
                        ),
                    ],
                )
            )
        finally:
            self.set_writing_status(False)

        if outcome.cancelled or outcome.option_id is None:
            self.send_message("stop decision: cancelled")
        else:
            self.send_message(f"stop decision: {outcome.option_id}")
