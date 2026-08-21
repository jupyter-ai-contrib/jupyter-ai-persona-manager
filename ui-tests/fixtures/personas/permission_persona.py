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

from jupyter_ai_persona_manager import (
    BasePersona,
    PermissionOption,
    PermissionRequest,
    PersonaDefaults,
)
from jupyterlab_chat.models import Message

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")


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
