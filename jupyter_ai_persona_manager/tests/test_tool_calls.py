"""Tests for the general tool-call API (tool_calls.py + BasePersona).

Mirrors the behaviors ACP covers in test_tool_call_manager.py: create, grouping
consecutive calls in one message, status transitions (failed is terminal),
raw_output updates, cancel -> failed, and attaching a permission to a tool call.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jupyter_ai_persona_manager.base_persona import BasePersona, PersonaDefaults
from jupyter_ai_persona_manager.permissions import (
    PermissionDiff,
    PermissionOption,
    PermissionRequest,
)
from jupyter_ai_persona_manager.tool_calls import TOOL_CALLS_METADATA_KEY


class _ConcretePersona(BasePersona):
    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="TP", description="d", avatar_path="", system_prompt=""
        )

    async def process_message(self, message):
        pass


def _make_persona():
    """Persona bypassing __init__, wired to a stateful in-memory chat.

    get_message returns the same object per message_id and metadata persists,
    so tool-call re-renders accumulate as they would against a real chat model.
    """
    persona = _ConcretePersona.__new__(_ConcretePersona)
    messages: dict[str, SimpleNamespace] = {}
    counter = {"n": 0}

    def add_message(new_message, trigger_actions=None):
        counter["n"] += 1
        mid = f"msg-{counter['n']}"
        messages[mid] = SimpleNamespace(id=mid, metadata={})
        return mid

    persona.chat = MagicMock()
    persona.chat.add_message = MagicMock(side_effect=add_message)
    persona.chat.get_message = MagicMock(side_effect=lambda mid: messages.get(mid))
    persona.chat.update_message = MagicMock()
    persona.chat.get_id = MagicMock(return_value="chat-1")
    persona.log = MagicMock()
    persona.state = MagicMock()
    persona._pending_permissions = {}
    persona._tool_calls = {}
    persona._tool_call_message = {}
    persona._messages = messages  # test accessor
    return persona


def _tool_calls_of(persona, message_id):
    return persona._messages[message_id].metadata.get(TOOL_CALLS_METADATA_KEY, [])


class TestReportToolCall:
    def test_creates_message_and_entry(self):
        persona = _make_persona()
        tcid = persona.report_tool_call("Reading a.py", kind="read", locations=["a.py"])
        mid = persona.tool_call_message_id(tcid)
        entries = _tool_calls_of(persona, mid)
        assert len(entries) == 1
        assert entries[0]["tool_call_id"] == tcid
        assert entries[0]["title"] == "Reading a.py"
        assert entries[0]["kind"] == "read"
        assert entries[0]["status"] == "in_progress"

    def test_consecutive_tool_calls_share_message(self):
        persona = _make_persona()
        first = persona.report_tool_call("one")
        mid = persona.tool_call_message_id(first)
        second = persona.report_tool_call("two", message_id=mid)
        assert persona.tool_call_message_id(second) == mid
        entries = _tool_calls_of(persona, mid)
        assert [e["title"] for e in entries] == ["one", "two"]

    def test_separate_messages_when_not_grouped(self):
        persona = _make_persona()
        first = persona.report_tool_call("one")
        second = persona.report_tool_call("two")
        assert persona.tool_call_message_id(first) != persona.tool_call_message_id(
            second
        )

    def test_update_status_and_output(self):
        persona = _make_persona()
        tcid = persona.report_tool_call("run", kind="execute")
        persona.update_tool_call(tcid, status="completed", raw_output="done")
        entry = _tool_calls_of(persona, persona.tool_call_message_id(tcid))[0]
        assert entry["status"] == "completed"
        assert entry["raw_output"] == "done"

    def test_failed_is_terminal(self):
        persona = _make_persona()
        tcid = persona.report_tool_call("run")
        persona.update_tool_call(tcid, status="failed")
        persona.update_tool_call(tcid, status="completed")  # must not override
        entry = _tool_calls_of(persona, persona.tool_call_message_id(tcid))[0]
        assert entry["status"] == "failed"

    def test_update_unknown_is_noop(self):
        persona = _make_persona()
        persona.update_tool_call("nope", status="completed")  # no raise
        persona.log.warning.assert_called()

    def test_cancel_marks_in_progress_failed(self):
        persona = _make_persona()
        a = persona.report_tool_call("a")
        b = persona.report_tool_call("b", message_id=persona.tool_call_message_id(a))
        persona.update_tool_call(b, status="completed")
        assert persona.cancel_tool_calls() == 1  # only 'a' was in progress
        entries = _tool_calls_of(persona, persona.tool_call_message_id(a))
        by_id = {e["tool_call_id"]: e for e in entries}
        assert by_id[a]["status"] == "failed"
        assert by_id[b]["status"] == "completed"

    def test_diffs_serialized(self):
        persona = _make_persona()
        tcid = persona.report_tool_call(
            "edit",
            kind="edit",
            diffs=[PermissionDiff(path="a.py", old_text="x\n", new_text="y\n")],
        )
        entry = _tool_calls_of(persona, persona.tool_call_message_id(tcid))[0]
        assert entry["diffs"] == [{"path": "a.py", "old_text": "x\n", "new_text": "y\n"}]


class TestPermissionAttachedToToolCall:
    @pytest.mark.asyncio
    async def test_permission_renders_on_tool_call_row(self):
        persona = _make_persona()
        tcid = persona.report_tool_call("edit", kind="edit")
        req = PermissionRequest(
            title="Approve edit?",
            tool_call_id=tcid,
            options=[
                PermissionOption(option_id="allow", name="Allow"),
                PermissionOption(option_id="deny", name="Deny"),
            ],
        )
        task = asyncio.create_task(persona.request_permission(req))
        await asyncio.sleep(0)

        # No standalone permission block; the buttons live on the tool call.
        mid = persona.tool_call_message_id(tcid)
        entry = _tool_calls_of(persona, mid)[0]
        assert entry["permission_status"] == "pending"
        assert entry["request_id"]
        assert entry["chat_id"] == "chat-1"
        assert entry["persona_id"] == persona.id
        assert [o["option_id"] for o in entry["permission_options"]] == [
            "allow",
            "deny",
        ]
        assert "permission_requests" not in persona._messages[mid].metadata

        # Resolve via the request_id carried on the tool call.
        assert persona.resolve_permission(entry["request_id"], "allow") is True
        outcome = await task
        assert outcome.option_id == "allow"
        entry = _tool_calls_of(persona, mid)[0]
        assert entry["permission_status"] == "resolved"
        assert entry["selected_option_id"] == "allow"

    @pytest.mark.asyncio
    async def test_unknown_tool_call_id_does_not_crash(self):
        persona = _make_persona()
        req = PermissionRequest(
            title="t",
            tool_call_id="ghost",
            options=[PermissionOption(option_id="a", name="A")],
        )
        task = asyncio.create_task(persona.request_permission(req))
        await asyncio.sleep(0)
        # Future still registered; resolve it to finish.
        rid = next(iter(persona._pending_permissions))
        persona.resolve_permission(rid, "a")
        outcome = await task
        assert outcome.option_id == "a"
        persona.log.warning.assert_called()
