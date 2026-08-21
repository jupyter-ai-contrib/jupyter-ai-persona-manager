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
    PermissionDiff,
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
    persona.chat.get_id = MagicMock(return_value="chat-1")
    persona.log = MagicMock()
    persona.state = MagicMock()
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
            chat_id="chat-1",
            request=req,
            status="pending",
        )
        assert block["request_id"] == "r1"
        assert block["persona_id"] == "p1"
        assert block["chat_id"] == "chat-1"
        assert block["title"] == "Run cmd"
        assert block["detail"] == "$ ls"
        assert block["correlation_id"] == "tc-1"
        assert block["status"] == "pending"
        assert block["selected_option_id"] is None
        assert block["options"] == [
            {"option_id": "allow", "name": "Allow", "kind": "allow_once"}
        ]

    def test_build_metadata_includes_diffs(self):
        req = PermissionRequest(
            title="edit",
            diffs=[PermissionDiff(path="a.py", old_text="x\n", new_text="y\n")],
            options=[PermissionOption(option_id="a", name="Allow")],
        )
        block = build_permission_metadata(
            request_id="r1", persona_id="p1", chat_id="chat-1", request=req
        )
        assert block["diffs"] == [
            {"path": "a.py", "old_text": "x\n", "new_text": "y\n"}
        ]

    def test_build_metadata_excludes_context(self):
        # context is server-private and must never appear on the wire/metadata.
        req = PermissionRequest(
            title="t",
            options=[PermissionOption(option_id="a", name="Allow")],
            context={"session_id": "s1", "tool_call_id": "tc-1"},
        )
        block = build_permission_metadata(
            request_id="r1", persona_id="p1", chat_id="chat-1", request=req
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
            "chat_id",
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
        requests = first_msg.metadata[PERMISSION_METADATA_KEY]
        assert isinstance(requests, list) and len(requests) == 1
        block = requests[0]
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
        block = last_msg.metadata[PERMISSION_METADATA_KEY][0]
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

    @pytest.mark.asyncio
    async def test_shutdown_cancels_pending(self):
        # On shutdown, an awaiting request_permission must unwind (cancelled)
        # rather than hang.
        persona = _make_persona()
        req = PermissionRequest(
            title="t", options=[PermissionOption(option_id="a", name="Allow")]
        )
        task = asyncio.create_task(persona.request_permission(req))
        await asyncio.sleep(0)
        assert len(persona._pending_permissions) == 1

        await persona.shutdown()

        outcome = await task
        assert outcome.cancelled is True
        assert outcome.option_id is None
        assert persona._pending_permissions == {}

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

    @pytest.mark.asyncio
    async def test_multiple_requests_same_message_do_not_clobber(self):
        # Two requests attached to the SAME message_id must coexist as separate
        # entries (keyed by request_id), not overwrite each other — this is the
        # ACP grouped-tool-calls scenario.
        persona = _make_persona()
        # Stateful message whose metadata persists across get/update, so we can
        # observe both requests accumulating in the one message.
        msg = SimpleNamespace(metadata={})
        persona.chat.get_message = MagicMock(return_value=msg)

        req1 = PermissionRequest(
            title="one",
            message_id="m1",
            options=[PermissionOption(option_id="a", name="A")],
        )
        req2 = PermissionRequest(
            title="two",
            message_id="m1",
            options=[PermissionOption(option_id="b", name="B")],
        )
        t1 = asyncio.create_task(persona.request_permission(req1))
        t2 = asyncio.create_task(persona.request_permission(req2))
        await asyncio.sleep(0)

        # Both pending requests live in the same message, distinct entries.
        reqs = msg.metadata[PERMISSION_METADATA_KEY]
        assert len(reqs) == 2
        assert {r["title"] for r in reqs} == {"one", "two"}
        assert len(persona._pending_permissions) == 2

        # Resolve each independently; each updates only its own entry.
        for rid in list(persona._pending_permissions):
            persona.resolve_permission(rid, "a")
        await asyncio.gather(t1, t2)

        reqs = msg.metadata[PERMISSION_METADATA_KEY]
        assert len(reqs) == 2
        assert all(r["status"] == "resolved" for r in reqs)
        assert {r["title"] for r in reqs} == {"one", "two"}


# ---------------------------------------------------------------------------
# Transport independence: request_permission's contract does not depend on how
# the decision is delivered back — any caller of resolve_permission works, and
# how the request is surfaced is an overridable, separate concern.
# ---------------------------------------------------------------------------
class TestTransportIndependence:
    @pytest.mark.asyncio
    async def test_resolves_via_arbitrary_transport(self):
        # A REST handler and an event listener are both just callers of the
        # resolve_permission seam. Simulate one as a plain callable.
        persona = _make_persona()

        def some_transport(request_id, option_id):
            return persona.resolve_permission(request_id, option_id)

        req = PermissionRequest(
            title="t", options=[PermissionOption(option_id="ok", name="OK")]
        )
        task = asyncio.create_task(persona.request_permission(req))
        await asyncio.sleep(0)
        rid = next(iter(persona._pending_permissions))
        assert some_transport(rid, "ok") is True
        outcome = await task
        assert outcome.option_id == "ok"

    @pytest.mark.asyncio
    async def test_decision_during_publish_is_not_lost(self):
        # The future must be registered BEFORE the request is surfaced. An
        # "instantaneous transport" that resolves from within publish only
        # succeeds if that ordering holds — this guards against the race.
        persona = _make_persona()
        seen = {}

        def instant_publish(request_id, request):
            seen["resolved"] = persona.resolve_permission(request_id, "a")
            return None  # not surfaced in chat

        persona._publish_permission_request = instant_publish
        req = PermissionRequest(
            title="t", options=[PermissionOption(option_id="a", name="Allow")]
        )
        outcome = await persona.request_permission(req)
        assert seen["resolved"] is True
        assert outcome.option_id == "a"
        # publish returned None -> no chat message was written to on resolve.
        persona.chat.update_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_publish_and_finalize_hooks(self):
        # A backend whose renderer is behind an async accessor (e.g. ACP's
        # client) can override the hooks as coroutines.
        persona = _make_persona()
        calls = []

        async def async_publish(request_id, request):
            calls.append(("publish", request_id))
            return "acp-msg"

        async def async_finalize(request_id, request, message_id, option_id):
            calls.append(("finalize", message_id, option_id))

        persona._publish_permission_request = async_publish
        persona._finalize_permission_request = async_finalize
        req = PermissionRequest(
            title="t", options=[PermissionOption(option_id="a", name="A")]
        )
        task = asyncio.create_task(persona.request_permission(req))
        await asyncio.sleep(0)
        rid = next(iter(persona._pending_permissions))
        persona.resolve_permission(rid, "a")
        outcome = await task
        assert outcome.option_id == "a"
        assert ("publish", rid) in calls
        assert ("finalize", "acp-msg", "a") in calls
        # The backend owns rendering: no generic metadata writes happened.
        persona.chat.update_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_publish_hook_is_overridable(self):
        # A subclass can surface the request however it likes (or not at all)
        # without changing the request/await/resolve lifecycle.
        persona = _make_persona()
        persona._publish_permission_request = MagicMock(return_value=None)
        req = PermissionRequest(
            title="t", options=[PermissionOption(option_id="a", name="Allow")]
        )
        task = asyncio.create_task(persona.request_permission(req))
        await asyncio.sleep(0)
        persona._publish_permission_request.assert_called_once()
        rid = next(iter(persona._pending_permissions))
        persona.resolve_permission(rid, "a")
        outcome = await task
        assert outcome.option_id == "a"
        persona.chat.add_message.assert_not_called()
        persona.chat.update_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_requests_resolve_independently(self):
        persona = _make_persona()
        req1 = PermissionRequest(
            title="1", options=[PermissionOption(option_id="a", name="A")]
        )
        req2 = PermissionRequest(
            title="2", options=[PermissionOption(option_id="b", name="B")]
        )
        t1 = asyncio.create_task(persona.request_permission(req1))
        t2 = asyncio.create_task(persona.request_permission(req2))
        await asyncio.sleep(0)
        assert len(persona._pending_permissions) == 2

        # Resolve exactly one: the other stays pending and unfinished.
        first_id = next(iter(persona._pending_permissions))
        assert persona.resolve_permission(first_id, "x") is True
        await asyncio.sleep(0)
        assert len(persona._pending_permissions) == 1
        assert sum(t.done() for t in (t1, t2)) == 1

        # Resolve the remaining one.
        second_id = next(iter(persona._pending_permissions))
        assert persona.resolve_permission(second_id, "y") is True
        outcomes = await asyncio.gather(t1, t2)
        assert {o.option_id for o in outcomes} == {"x", "y"}
        assert persona._pending_permissions == {}


# ---------------------------------------------------------------------------
# PersonaManager routing of permission_response events
# ---------------------------------------------------------------------------
class TestManagerRouting:
    @pytest.mark.asyncio
    async def test_routes_to_persona(self):
        persona = _make_persona()
        persona.resolve_permission = MagicMock(return_value=True)
        mgr = SimpleNamespace(
            chat=MagicMock(get_id=MagicMock(return_value="chat-1")),
            _personas={persona.id: persona},
            log=MagicMock(),
        )
        data = {
            "chat_id": "chat-1",
            "persona_id": persona.id,
            "request_id": "r1",
            "option_id": "a",
        }
        await PersonaManager._on_permission_response(
            mgr, None, PERMISSION_RESPONSE_EVENT_SCHEMA_ID, data
        )
        persona.resolve_permission.assert_called_once_with("r1", "a")

    @pytest.mark.asyncio
    async def test_ignores_other_chats(self):
        persona = _make_persona()
        persona.resolve_permission = MagicMock(return_value=True)
        mgr = SimpleNamespace(
            chat=MagicMock(get_id=MagicMock(return_value="chat-1")),
            _personas={persona.id: persona},
            log=MagicMock(),
        )
        data = {
            "chat_id": "OTHER-chat",
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
        mgr = SimpleNamespace(
            chat=MagicMock(get_id=MagicMock(return_value="chat-1")),
            _personas={},
            log=MagicMock(),
        )
        data = {
            "chat_id": "chat-1",
            "persona_id": "ghost",
            "request_id": "r1",
            "option_id": "a",
        }
        await PersonaManager._on_permission_response(
            mgr, None, PERMISSION_RESPONSE_EVENT_SCHEMA_ID, data
        )
        mgr.log.warning.assert_called_once()
