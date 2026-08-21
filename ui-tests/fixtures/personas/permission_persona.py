"""
Fixture persona for E2E tests: a `BasePersona` that requests the user's
permission before "acting", exercising the general permission API
(`BasePersona.request_permission`).

On each message it raises a permission request with an Allow/Deny pair. The
request is reflected in the chat as a `permission_request` metadata block, which
the persona-manager frontend renders as buttons. When the user clicks a button,
the frontend emits a `permission_response` Jupyter Event; the manager routes it
to this persona, resolving `request_permission`, and the persona posts the
decision it received — so a test can assert the full client -> server round-trip.

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
from jupyterlab_chat.models import Message

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")


def _ensure_chat_id(chat) -> None:
    """Bridging shim: guarantee ``chat.get_id()`` returns a stable unique id.

    Permission routing keys on ``chat.get_id()``. A separate effort makes that
    always return a unique id; until it lands, freshly-created ``.chat`` files
    have no id, so seed one here (RTC: ``set_id``; RTC-free: ``_metadata``).
    Test-only — real personas rely on the platform providing the id.
    """
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


class PermissionPersona(BasePersona):
    """Test-only persona that asks permission, then reports the decision."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Permission Persona",
            description="Test-only persona that requests permission before acting.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        _ensure_chat_id(self.chat)
        outcome = await self.request_permission(
            PermissionRequest(
                title="Approve action?",
                detail="permission-fixture-detail",
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
        if outcome.cancelled or outcome.option_id is None:
            self.send_message("decision: cancelled")
        else:
            self.send_message(f"decision: {outcome.option_id}")
