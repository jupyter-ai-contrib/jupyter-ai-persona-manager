"""
E2E fixture: a persona that first writes a normal message ("Sure"), then attaches
TWO permission requests to that same message (by passing its ``message_id`` to
both requests). Exercises multiple permission requests sharing one message — the
case that must not clobber.

Not part of the shipped package; see AGENTS.md.
"""

import asyncio
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


def _opts() -> list[PermissionOption]:
    return [
        PermissionOption(option_id="allow", name="Allow", kind="allow_once"),
        PermissionOption(option_id="deny", name="Deny", kind="reject_once"),
    ]


class VerbosePermissionRequesterPersona(BasePersona):
    """Writes a message, then attaches two permission requests to it."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Verbose Requester Persona",
            description="Writes a message, then adds two permission requests to it.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        _ensure_chat_id(self.chat)
        # Write a normal message first, then hang both requests off of it.
        message_id = self.chat.add_message(
            NewMessage(body="Sure", sender=self.id), trigger_actions=[]
        )
        first, second = await asyncio.gather(
            self.request_permission(
                PermissionRequest(
                    title="Verbose one", message_id=message_id, options=_opts()
                )
            ),
            self.request_permission(
                PermissionRequest(
                    title="Verbose two", message_id=message_id, options=_opts()
                )
            ),
        )
        self.send_message(
            f"verbose decisions: {first.option_id}, {second.option_id}"
        )
