"""General, backend-agnostic permission handling for personas.

A persona asks the user to approve an action by calling
:meth:`BasePersona.request_permission`. The request is reflected in the chat by
writing a ``permission_request`` block into a chat message's ``metadata`` (the
same mechanism the frontend already reads to render UI). The user's decision
travels back to the server over the **Jupyter Events** plane: the frontend emits
a ``permission_response/v1`` event (``POST /api/events``), which a server-side
listener on the :class:`PersonaManager` routes directly to the target persona by
``(chat_id, persona_id, request_id)`` — where ``chat_id`` is the chat's stable
unique id (``chat.get_id()``, stable across file moves) — so there is no
endpoint-side search and no backend-specific identifiers.

Nothing here is ACP-specific. ACP concepts such as ``session_id`` /
``tool_call_id`` are carried by the persona inside the opaque
:attr:`PermissionRequest.context` dict and echoed back verbatim in
:attr:`PermissionOutcome.request`, so the general API never mentions them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from jupyter_events import EventLogger


# Metadata key holding a message's pending/resolved permission request. The
# frontend renders buttons from this block and emits the decision back.
PERMISSION_METADATA_KEY = "permission_request"


class PermissionOption(BaseModel):
    """A single choice offered to the user for a permission request."""

    option_id: str
    """Stable identifier returned to the persona when this option is chosen."""

    name: str
    """Human-readable button label."""

    kind: Optional[str] = None
    """Optional hint for styling/semantics, e.g. ``"allow_once"``,
    ``"allow_always"``, ``"reject_once"``. Backends map their own vocabularies
    onto this freely."""


class PermissionRequest(BaseModel):
    """A request for the user to approve (or reject) an action."""

    title: str
    """Short, human-readable summary of the action, e.g. ``"Run: rm -rf build/"``."""

    options: list[PermissionOption]
    """The choices offered to the user. Typically an allow/deny pair, but any
    number of options is supported."""

    detail: Optional[str] = None
    """Optional longer body shown under the title (a command, path, or preview)."""

    message_id: Optional[str] = None
    """When set, the request is attached to this existing chat message's
    metadata. When ``None``, a new message is created to host the request."""

    correlation_id: Optional[str] = None
    """Optional grouping key the persona can use to correlate this request with
    its own state (ACP maps ``tool_call_id`` here). Opaque to the manager."""

    context: Optional[dict[str, Any]] = None
    """Opaque, persona-defined data echoed back verbatim in
    :attr:`PermissionOutcome.request`. This is where backends stash their own
    identifiers (e.g. ACP's ``session_id``). It is **server-private**: it is
    never placed in the chat metadata nor put on the wire to the frontend."""


class PermissionOutcome(BaseModel):
    """The result of a resolved :class:`PermissionRequest`."""

    option_id: Optional[str] = None
    """The ``option_id`` the user selected, or ``None`` if the request was
    cancelled/denied without a selection."""

    request: PermissionRequest
    """The originating request, so the caller can recover its ``context`` and
    ``correlation_id``."""

    cancelled: bool = False
    """``True`` when the request was cancelled (e.g. the turn was interrupted or
    the user dismissed it) rather than answered with an option."""


# ---------------------------------------------------------------------------
# Jupyter Event schema: the user's decision, emitted by the frontend
# (client -> server) via POST /api/events and consumed by a server-side
# listener on the PersonaManager.
# ---------------------------------------------------------------------------

PERMISSION_RESPONSE_EVENT_SCHEMA_ID = (
    "https://schema.jupyter.org/jupyter_ai_persona_manager/permission_response/v1"
)

PERMISSION_RESPONSE_EVENT_SCHEMA = {
    "$id": PERMISSION_RESPONSE_EVENT_SCHEMA_ID,
    "version": "1",
    "title": "Permission response",
    "personal-data": True,
    "description": (
        "A user's decision on a pending permission request, emitted by the "
        "frontend and routed to the requesting persona."
    ),
    "type": "object",
    "required": ["chat_id", "persona_id", "request_id"],
    "properties": {
        "chat_id": {
            "type": "string",
            "description": (
                "The chat's stable unique id (chat.get_id()); routes to the "
                "PersonaManager. Stable across file moves, unlike a path."
            ),
        },
        "persona_id": {
            "type": "string",
            "description": "The requesting persona's stable id.",
        },
        "request_id": {
            "type": "string",
            "description": "The permission request's unique id.",
        },
        "option_id": {
            "type": ["string", "null"],
            "description": (
                "The chosen option's id, or null to cancel/deny without a "
                "selection."
            ),
        },
    },
    "additionalProperties": False,
}


def register_permission_event_schemas(event_logger: "EventLogger") -> None:
    """Register the permission event schema on ``event_logger`` (idempotent)."""
    try:
        event_logger.register_event_schema(PERMISSION_RESPONSE_EVENT_SCHEMA)
    except Exception:  # pragma: no cover - already registered / defensive
        pass


def build_permission_metadata(
    *,
    request_id: str,
    persona_id: str,
    chat_id: str,
    request: PermissionRequest,
    status: str = "pending",
    selected_option_id: Optional[str] = None,
) -> dict[str, Any]:
    """Build the ``permission_request`` metadata block written to a chat message.

    Deliberately excludes :attr:`PermissionRequest.context` — that stays
    server-side.
    """
    return {
        "request_id": request_id,
        "persona_id": persona_id,
        "chat_id": chat_id,
        "title": request.title,
        "detail": request.detail,
        "correlation_id": request.correlation_id,
        "options": [opt.model_dump() for opt in request.options],
        "status": status,
        "selected_option_id": selected_option_id,
    }
