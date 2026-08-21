"""General, backend-agnostic tool-call reporting for personas.

A persona reports a tool call (an action it is taking — reading a file, running
a command, editing code) via :meth:`BasePersona.report_tool_call`, then updates
its status/output via :meth:`BasePersona.update_tool_call`. Tool calls are
reflected in the chat as a ``tool_calls`` list on a message's ``metadata``; the
frontend renders each with a status glyph, an optional detail line, file diffs,
and expandable output.

A permission request can attach to a tool call: calling
``request_permission(PermissionRequest(..., tool_call_id=...))`` renders the
approve/deny buttons on that tool-call row (rather than as a standalone block),
mirroring how ACP surfaces permissions on its tool calls.

Nothing here is ACP-specific. ACP maps its protocol (``ToolCallStart`` /
``ToolCallProgress``, ``kind``, ``raw_input``, diff extraction) onto these
calls; the general model just carries the already-computed fields.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel

from .permissions import PermissionDiff, PermissionOption

# Metadata key holding a message's tool calls (a list, in creation order). A
# message can host several tool calls (consecutive calls grouped together).
TOOL_CALLS_METADATA_KEY = "tool_calls"


class ToolCall(BaseModel):
    """A single tool call reported by a persona, rendered in the chat.

    Permission fields are populated only when a permission request is attached
    to this tool call (see ``request_permission(tool_call_id=...)``)."""

    tool_call_id: str
    title: str
    kind: Optional[str] = None
    """Category hint for display, e.g. ``read``/``edit``/``execute``/``search``.
    Free-form; the frontend styles known kinds and falls back gracefully."""

    status: Optional[str] = None
    """``in_progress`` | ``completed`` | ``failed`` (free-form; ``failed`` is
    treated as terminal by :meth:`update`)."""

    locations: Optional[list[str]] = None
    """File paths / resource URIs this call touches."""

    raw_input: Optional[Any] = None
    raw_output: Optional[Any] = None
    diffs: Optional[list[PermissionDiff]] = None

    # --- permission fields (set when a request is attached to this tool call) --
    permission_options: Optional[list[PermissionOption]] = None
    permission_status: Optional[str] = None  # "pending" | "resolved"
    selected_option_id: Optional[str] = None
    # Identifiers the frontend echoes back in the permission_response event.
    request_id: Optional[str] = None
    chat_id: Optional[str] = None
    persona_id: Optional[str] = None


def upsert_tool_call(
    existing: Optional[list[dict[str, Any]]], tool_call: ToolCall
) -> list[dict[str, Any]]:
    """Return the tool-call list with ``tool_call`` inserted or replaced (by id).

    Preserves creation order: an existing entry with the same ``tool_call_id``
    is replaced in place; a new one is appended.
    """
    entries = list(existing) if isinstance(existing, list) else []
    dumped = tool_call.model_dump(exclude_none=True)
    for i, entry in enumerate(entries):
        if entry.get("tool_call_id") == tool_call.tool_call_id:
            entries[i] = dumped
            break
    else:
        entries.append(dumped)
    return entries
