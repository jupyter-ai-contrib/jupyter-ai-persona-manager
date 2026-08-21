"""
E2E fixture: a persona exercising the general tool-call UI. On each message it
reports a completed read tool call, then an edit tool call (same message, so
they group) carrying a diff and requiring permission. The approve/deny buttons
render on the edit tool call's row; the decision drives the final status.

Not part of the shipped package; see AGENTS.md.
"""

import os
import uuid

from jupyter_ai_persona_manager import (
    BasePersona,
    PermissionDiff,
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


class ToolCallPersona(BasePersona):
    """Reports tool calls (completed read + edit-with-diff needing permission)."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Tool Call Persona",
            description="Reports tool calls and asks permission on an edit.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        _ensure_chat_id(self.chat)

        # A completed read tool call.
        read = self.report_tool_call(
            "Reading example.py", kind="read", locations=["example.py"]
        )
        self.update_tool_call(read, status="completed", raw_output="print('hi')")

        # An edit tool call in the same message, with a diff, needing approval.
        edit = self.report_tool_call(
            "Editing config.py",
            kind="edit",
            message_id=self.tool_call_message_id(read),
            diffs=[
                PermissionDiff(
                    path="config.py",
                    old_text="value = 1\n",
                    new_text="value = 2\n",
                )
            ],
        )
        outcome = await self.request_permission(
            PermissionRequest(
                title="Editing config.py",
                tool_call_id=edit,
                options=[
                    PermissionOption(option_id="allow", name="Allow", kind="allow_once"),
                    PermissionOption(option_id="deny", name="Deny", kind="reject_once"),
                ],
            )
        )
        if outcome.option_id == "allow":
            self.update_tool_call(edit, status="completed")
            self.send_message("tool decision: allow")
        else:
            self.update_tool_call(edit, status="failed")
            self.send_message("tool decision: denied")
