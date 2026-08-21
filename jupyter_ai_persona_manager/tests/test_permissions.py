"""Tests for the general persona permission API (permissions.py + BasePersona)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jupyter_ai_persona_manager.base_persona import BasePersona, PersonaDefaults
from jupyter_ai_persona_manager.permissions import (
    PERMISSION_METADATA_KEY,
    PERMISSION_RESPONSE_EVENT_SCHEMA,
    PERMISSION_RESPONSE_EVENT_SCHEMA_ID,
    PermissionOption,
    PermissionOutcome,
    PermissionRequest,
    build_permission_metadata,
    register_permission_event_schemas,
)
from jupyter_ai_persona_manager.persona_manager import PersonaManager


class _ConcretePersona(BasePersona):
    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="TestPersona",
            description="A test persona",
            avatar_path="",
            system_prompt="",
        )

    async def process_message(self, message):
        pass


def _make_persona():
    """Create a persona bypassing __init__, wired to a mock chat."""
    persona = _ConcretePersona.__new__(_ConcretePersona)
    persona.chat = MagicMock()
    persona.chat.add_message = MagicMock(return_value="msg-123")
    persona.chat.get_message = MagicMock(
        side_effect=lambda _id: SimpleNamespace(metadata=None)
    )
    persona.chat.update_message = MagicMock()
    persona.log = MagicMock()
    persona._pending_permissions = {}
    return persona


# ---------------------------------------------------------------------------
# Models + helpers
# ---------------------------------------------------------------------------
class TestPermissionModels:
    def test_build_metadata_shape(self):
        req = PermissionRequest(
            title="Run cmd",
            detail="$ ls",
            correlation_id="tc-1",
            options=[PermissionOption(option_id="allow", name="Allow", kind="allow_once")],
        )
        block = build_permission_metadata(
            request_id="r1",
            persona_id="p1",
            room_id="room-1",
            request=req,
            status="pending",
        )
        assert block["request_id"] == "r1"
        assert block["persona_id"] == "p1"
        assert block["room_id"] == "room-1"
        assert block["title"] == "Run cmd"
        assert block["detail"] == "$ ls"
        assert block["correlation_id"] == "tc-1"
        assert block["status"] == "pending"
        assert block["selected_option_id"] is None
        assert block["options"] == [
            {"option_id": "allow", "name": "Allow", "kind": "allow_once"}
        ]

    def test_build_metadata_excludes_context(self):
        # context is server-private and must never appear on the wire/metadata.
        req = PermissionRequest(
            title="t",
            options=[PermissionOption(option_id="a", name="Allow")],
            context={"session_id": "s1", "tool_call_id": "tc-1"},
        )
        block = build_permission_metadata(
            request_id="r1", persona_id="p1", room_id="room-1", request=req
        )
        assert "context" not in block

    def test_schema_registration_idempotent(self):
        logger = MagicMock()
        register_permission_event_schemas(logger)
        logger.register_event_schema.assert_called_once_with(
            PERMISSION_RESPONSE_EVENT_SCHEMA
        )
        # A second call must not raise even if the logger rejects duplicates.
        logger.register_event_schema.side_effect = Exception("already registered")
        register_permission_event_schemas(logger)

    def test_schema_required_fields(self):
        assert PERMISSION_RESPONSE_EVENT_SCHEMA["$id"] == PERMISSION_RESPONSE_EVENT_SCHEMA_ID
        assert set(PERMISSION_RESPONSE_EVENT_SCHEMA["required"]) == {
            "room_id",
            "persona_id",
            "request_id",
        }


# ---------------------------------------------------------------------------
# BasePersona.request_permission / resolve_permission / cancel_permissions
# ---------------------------------------------------------------------------
class TestRequestPermission:
    @pytest.mark.asyncio
    async def test_creates_message_and_pending_metadata(self):
        persona = _make_persona()
        req = PermissionRequest(
            title="t", options=[PermissionOption(option_id="a", name="Allow")]
        )
        task = asyncio.create_task(persona.request_permission(req))
        await asyncio.sleep(0)

        # A message was created and pending metadata written.
        persona.chat.add_message.assert_called_once()
        first_msg = persona.chat.update_message.call_args_list[0][0][0]
        block = first_msg.metadata[PERMISSION_METADATA_KEY]
        assert block["status"] == "pending"
        assert block["persona_id"] == persona.id
        assert len(persona._pending_permissions) == 1

        # Resolve it and confirm the outcome.
        request_id = next(iter(persona._pending_permissions))
        assert persona.resolve_permission(request_id, "a") is True
        outcome = await task
        assert isinstance(outcome, PermissionOutcome)
        assert outcome.option_id == "a"
        assert outcome.cancelled is False
        assert persona._pending_permissions == {}

    @pytest.mark.asyncio
    async def test_uses_existing_message_id(self):
        persona = _make_persona()
        req = PermissionRequest(
            title="t",
            message_id="existing-msg",
            options=[PermissionOption(option_id="a", name="Allow")],
        )
        task = asyncio.create_task(persona.request_permission(req))
        await asyncio.sleep(0)
        # No new message created — the request attached to the given message.
        persona.chat.add_message.assert_not_called()
        persona.chat.get_message.assert_called_with("existing-msg")
        request_id = next(iter(persona._pending_permissions))
        persona.resolve_permission(request_id, "a")
        await task

    @pytest.mark.asyncio
    async def test_echoes_context_and_correlation(self):
        persona = _make_persona()
        req = PermissionRequest(
            title="t",
            correlation_id="tc-9",
            options=[PermissionOption(option_id="a", name="Allow")],
            context={"session_id": "s1", "tool_call_id": "tc-9"},
        )
        task = asyncio.create_task(persona.request_permission(req))
        await asyncio.sleep(0)
        request_id = next(iter(persona._pending_permissions))
        persona.resolve_permission(request_id, "a")
        outcome = await task
        # ACP identifiers survive the round-trip via the opaque context.
        assert outcome.request.context == {"session_id": "s1", "tool_call_id": "tc-9"}
        assert outcome.request.correlation_id == "tc-9"

    @pytest.mark.asyncio
    async def test_resolved_metadata_written(self):
        persona = _make_persona()
        req = PermissionRequest(
            title="t", options=[PermissionOption(option_id="a", name="Allow")]
        )
        task = asyncio.create_task(persona.request_permission(req))
        await asyncio.sleep(0)
        request_id = next(iter(persona._pending_permissions))
        persona.resolve_permission(request_id, "a")
        await task
        # Last metadata write reflects the resolution.
        last_msg = persona.chat.update_message.call_args_list[-1][0][0]
        block = last_msg.metadata[PERMISSION_METADATA_KEY]
        assert block["status"] == "resolved"
        assert block["selected_option_id"] == "a"

    @pytest.mark.asyncio
    async def test_cancel_returns_none_outcome(self):
        persona = _make_persona()
        req = PermissionRequest(
            title="t", options=[PermissionOption(option_id="a", name="Allow")]
        )
        task = asyncio.create_task(persona.request_permission(req))
        await asyncio.sleep(0)
        assert persona.cancel_permissions() == 1
        outcome = await task
        assert outcome.option_id is None
        assert outcome.cancelled is True

    def test_resolve_unknown_request(self):
        persona = _make_persona()
        assert persona.resolve_permission("nope", "a") is False

    @pytest.mark.asyncio
    async def test_resolve_already_done(self):
        persona = _make_persona()
        req = PermissionRequest(
            title="t", options=[PermissionOption(option_id="a", name="Allow")]
        )
        task = asyncio.create_task(persona.request_permission(req))
        await asyncio.sleep(0)
        request_id = next(iter(persona._pending_permissions))
        assert persona.resolve_permission(request_id, "a") is True
        # Second resolve is a no-op (already resolved).
        assert persona.resolve_permission(request_id, "b") is False
        await task


# ---------------------------------------------------------------------------
# PersonaManager routing of permission_response events
# ---------------------------------------------------------------------------
class TestManagerRouting:
    @pytest.mark.asyncio
    async def test_routes_to_persona(self):
        persona = _make_persona()
        persona.resolve_permission = MagicMock(return_value=True)
        mgr = SimpleNamespace(
            room_id="room-1", _personas={persona.id: persona}, log=MagicMock()
        )
        data = {
            "room_id": "room-1",
            "persona_id": persona.id,
            "request_id": "r1",
            "option_id": "a",
        }
        await PersonaManager._on_permission_response(
            mgr, None, PERMISSION_RESPONSE_EVENT_SCHEMA_ID, data
        )
        persona.resolve_permission.assert_called_once_with("r1", "a")

    @pytest.mark.asyncio
    async def test_ignores_other_rooms(self):
        persona = _make_persona()
        persona.resolve_permission = MagicMock(return_value=True)
        mgr = SimpleNamespace(
            room_id="room-1", _personas={persona.id: persona}, log=MagicMock()
        )
        data = {
            "room_id": "OTHER-room",
            "persona_id": persona.id,
            "request_id": "r1",
            "option_id": "a",
        }
        await PersonaManager._on_permission_response(
            mgr, None, PERMISSION_RESPONSE_EVENT_SCHEMA_ID, data
        )
        persona.resolve_permission.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_persona_logs_warning(self):
        mgr = SimpleNamespace(room_id="room-1", _personas={}, log=MagicMock())
        data = {
            "room_id": "room-1",
            "persona_id": "ghost",
            "request_id": "r1",
            "option_id": "a",
        }
        await PersonaManager._on_permission_response(
            mgr, None, PERMISSION_RESPONSE_EVENT_SCHEMA_ID, data
        )
        mgr.log.warning.assert_called_once()
