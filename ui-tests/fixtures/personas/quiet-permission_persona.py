"""
E2E fixture: a persona that raises TWO permission requests immediately on each
message, without first writing any message of its own. Each request auto-creates
its own hosting message (``message_id`` left unset). Exercises multiple
concurrent permission requests from one persona.

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
from jupyterlab_chat.models import Message

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


class QuietPermissionRequesterPersona(BasePersona):
    """Raises two permission requests at once, with no preamble message."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Quiet Requester Persona",
            description="Raises two permission requests immediately, no preamble.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        _ensure_chat_id(self.chat)
        # Two concurrent requests, each auto-creating its own hosting message.
        first, second = await asyncio.gather(
            self.request_permission(
                PermissionRequest(title="Quiet one", options=_opts())
            ),
            self.request_permission(
                PermissionRequest(title="Quiet two", options=_opts())
            ),
        )
        self.send_message(
            f"quiet decisions: {first.option_id}, {second.option_id}"
        )
