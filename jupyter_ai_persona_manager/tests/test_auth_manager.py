"""
Tests for the persona authentication lifecycle: the `PersonaAuthManager`
mechanism and how `BasePersona.on_message` reacts to an unauthenticated
`prepare()` (the NOT_AUTHED state).
"""

import asyncio
import contextlib
import logging
import uuid
from unittest.mock import MagicMock

import pytest
from traitlets.config import Config, LoggingConfigurable

from jupyter_ai_persona_manager import (
    PersonaAuthManager,
    PersonaNotAuthenticated,
    PreparationState,
)
from jupyter_ai_persona_manager.base_persona import BasePersona, PersonaDefaults


@pytest.fixture(autouse=True)
def _clear_shared_polls():
    """The poll registry is class-level and shared across managers; clear it
    after every test so one test's poll never leaks into the next."""
    yield
    for task in list(PersonaAuthManager._poll_tasks.values()):
        if not task.done():
            task.cancel()
    PersonaAuthManager._poll_tasks.clear()
    PersonaAuthManager._poll_deadlines.clear()
    PersonaAuthManager._poll_waiters.clear()


# ---------------------------------------------------------------------------
# PersonaAuthManager (mechanism)
# ---------------------------------------------------------------------------


class _FakeParent(LoggingConfigurable):
    """A minimal stand-in for the persona a manager serves.

    ``id`` and ``chat_id`` are what `PersonaAuthManager` reads to build its
    `poll_key`; pass a shared ``persona_id`` to two fakes to exercise
    ``scope="global"`` sharing, or distinct ``chat_id``s under ``scope="chat"``.
    """

    def __init__(self, persona_id=None, chat_id="chat", **kwargs):
        super().__init__(**kwargs)
        self.auth_calls = 0
        self.timeout_calls = 0
        self.id = persona_id or f"test-persona::{uuid.uuid4()}"
        self.chat_id = chat_id
        self.name = "Fake"

    async def handle_auth(self):
        self.auth_calls += 1

    async def handle_auth_timeout(self):
        self.timeout_calls += 1


def _task_for(mgr):
    return PersonaAuthManager._poll_tasks.get(mgr.poll_key)


async def _drain(task):
    """Await a cancelled poll task so its `finally` cleanup runs deterministically."""
    with contextlib.suppress(asyncio.CancelledError):
        await task


class TestPersonaAuthManager:
    @pytest.mark.asyncio
    async def test_default_always_authed(self):
        mgr = PersonaAuthManager(parent=_FakeParent())
        assert await mgr.check_auth() is True
        await mgr.assert_auth()  # does not raise

    @pytest.mark.asyncio
    async def test_check_auth_caches_true(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            return True

        mgr = PersonaAuthManager(parent=_FakeParent(), check_auth_fn=fn)
        assert await mgr.check_auth() is True
        assert await mgr.check_auth() is True
        assert calls["n"] == 1  # cached after first success

    @pytest.mark.asyncio
    async def test_assert_auth_raises_when_unauthenticated(self):
        mgr = PersonaAuthManager(parent=_FakeParent(), check_auth_fn=lambda: False)
        with pytest.raises(PersonaNotAuthenticated):
            await mgr.assert_auth()

    @pytest.mark.asyncio
    async def test_async_check_fn_supported(self):
        async def fn():
            return True

        mgr = PersonaAuthManager(parent=_FakeParent(), check_auth_fn=fn)
        assert await mgr.check_auth() is True

    @pytest.mark.asyncio
    async def test_poll_resumes_on_auth(self):
        state = {"authed": False}
        parent = _FakeParent()
        mgr = PersonaAuthManager(
            parent=parent,
            check_auth_fn=lambda: state["authed"],
            default_poll_interval=0.01,
        )
        mgr.start_poll()
        await asyncio.sleep(0.03)
        assert parent.auth_calls == 0  # still waiting
        state["authed"] = True
        await asyncio.sleep(0.05)
        assert parent.auth_calls == 1  # resumed exactly once

    @pytest.mark.asyncio
    async def test_reset_clears_cache_and_stops(self):
        state = {"authed": True}
        mgr = PersonaAuthManager(
            parent=_FakeParent(), check_auth_fn=lambda: state["authed"]
        )
        assert await mgr.check_auth() is True
        state["authed"] = False
        mgr.reset()
        assert await mgr.check_auth() is False  # cache cleared, re-checks

    @pytest.mark.asyncio
    async def test_inert_with_no_check_fn(self):
        # With no check_auth_fn the manager is authed from the start and
        # start_poll() is a no-op (nothing to poll).
        mgr = PersonaAuthManager(parent=_FakeParent())
        assert mgr.authed is True
        mgr.start_poll()
        assert _task_for(mgr) is None

    @pytest.mark.asyncio
    async def test_start_poll_noop_when_already_authed(self):
        mgr = PersonaAuthManager(parent=_FakeParent(), check_auth_fn=lambda: True)
        assert await mgr.check_auth() is True  # caches authed
        mgr.start_poll()
        assert _task_for(mgr) is None  # no poll: already authenticated

    @pytest.mark.asyncio
    async def test_start_poll_interval_overrides_default(self):
        # A slow default must not delay a caller that passes an explicit
        # interval to start_poll().
        state = {"authed": False}
        parent = _FakeParent()
        mgr = PersonaAuthManager(
            parent=parent,
            check_auth_fn=lambda: state["authed"],
            default_poll_interval=100.0,
        )
        mgr.start_poll(interval=0.01)
        state["authed"] = True
        await asyncio.sleep(0.05)
        assert parent.auth_calls == 1  # resumed on the fast override, not the default

    def test_default_poll_interval_is_configurable_trait(self):
        cfg = Config()
        cfg.PersonaAuthManager.default_poll_interval = 2.5
        mgr = PersonaAuthManager(parent=_FakeParent(), config=cfg)
        assert mgr.default_poll_interval == 2.5

    # -- poll_timeout ------------------------------------------------------

    @pytest.mark.asyncio
    async def test_poll_timeout_fires_handler_and_stops(self):
        parent = _FakeParent()
        mgr = PersonaAuthManager(
            parent=parent,
            check_auth_fn=lambda: False,  # never authenticates
            default_poll_interval=0.005,
            default_poll_timeout=0.03,
        )
        mgr.start_poll()
        await asyncio.sleep(0.08)
        assert parent.timeout_calls == 1  # gave up and notified once
        assert parent.auth_calls == 0  # never resumed
        assert _task_for(mgr) is None  # registry cleaned up on exit

    @pytest.mark.asyncio
    async def test_start_poll_timeout_overrides_default(self):
        # A long default must not delay a caller that passes an explicit timeout.
        parent = _FakeParent()
        mgr = PersonaAuthManager(
            parent=parent,
            check_auth_fn=lambda: False,
            default_poll_interval=0.005,
            default_poll_timeout=100.0,
        )
        mgr.start_poll(timeout=0.03)
        await asyncio.sleep(0.08)
        assert parent.timeout_calls == 1  # gave up on the fast override
        assert _task_for(mgr) is None

    @pytest.mark.asyncio
    async def test_start_poll_extends_deadline(self):
        parent = _FakeParent()
        mgr = PersonaAuthManager(
            parent=parent,
            check_auth_fn=lambda: False,
            default_poll_interval=0.01,
            default_poll_timeout=10.0,
        )
        mgr.start_poll()
        first = PersonaAuthManager._poll_deadlines[mgr.poll_key]
        await asyncio.sleep(0.02)
        mgr.start_poll()  # a fresh message must push the deadline out
        assert PersonaAuthManager._poll_deadlines[mgr.poll_key] > first

    def test_default_poll_timeout_is_configurable_trait(self):
        cfg = Config()
        cfg.PersonaAuthManager.default_poll_timeout = 42.0
        mgr = PersonaAuthManager(parent=_FakeParent(), config=cfg)
        assert mgr.default_poll_timeout == 42.0

    # -- scope (constructor argument) --------------------------------------

    def test_scope_defaults_to_global(self):
        mgr = PersonaAuthManager(parent=_FakeParent())
        assert mgr.scope == "global"

    def test_invalid_scope_rejected(self):
        with pytest.raises(ValueError):
            PersonaAuthManager(parent=_FakeParent(), scope="bogus")

    def test_global_scope_key_is_persona_id(self):
        mgr = PersonaAuthManager(
            parent=_FakeParent(persona_id="p1"), check_auth_fn=lambda: False
        )
        assert mgr.poll_key == "p1"

    def test_chat_scope_key_includes_chat_id(self):
        mgr = PersonaAuthManager(
            parent=_FakeParent(persona_id="p1", chat_id="c9"),
            check_auth_fn=lambda: False,
            scope="chat",
        )
        assert mgr.poll_key == "p1::c9"

    @pytest.mark.asyncio
    async def test_global_scope_shares_one_poll_and_resumes_all(self):
        # Two persona instances (two chats) of the same persona share one poll
        # under global scope, and both are resumed when auth succeeds.
        state = {"authed": False}
        p1 = _FakeParent(persona_id="shared")
        p2 = _FakeParent(persona_id="shared")
        m1 = PersonaAuthManager(
            parent=p1, check_auth_fn=lambda: state["authed"], default_poll_interval=0.01
        )
        m2 = PersonaAuthManager(
            parent=p2, check_auth_fn=lambda: state["authed"], default_poll_interval=0.01
        )
        m1.start_poll()
        m2.start_poll()
        # Exactly one shared task, two registered waiters.
        assert len(PersonaAuthManager._poll_tasks) == 1
        assert PersonaAuthManager._poll_waiters["shared"] == {p1, p2}

        state["authed"] = True
        await asyncio.sleep(0.05)
        assert p1.auth_calls == 1 and p2.auth_calls == 1  # both chats resumed

    @pytest.mark.asyncio
    async def test_chat_scope_runs_separate_polls(self):
        p1 = _FakeParent(persona_id="p", chat_id="c1")
        p2 = _FakeParent(persona_id="p", chat_id="c2")
        m1 = PersonaAuthManager(
            parent=p1, check_auth_fn=lambda: False, default_poll_interval=0.01, scope="chat"
        )
        m2 = PersonaAuthManager(
            parent=p2, check_auth_fn=lambda: False, default_poll_interval=0.01, scope="chat"
        )
        m1.start_poll()
        m2.start_poll()
        assert set(PersonaAuthManager._poll_tasks) == {"p::c1", "p::c2"}

    # -- ref-counted stop --------------------------------------------------

    @pytest.mark.asyncio
    async def test_stop_keeps_shared_poll_for_remaining_waiters(self):
        # Closing one chat must NOT strand the others sharing a global poll.
        state = {"authed": False}
        p1 = _FakeParent(persona_id="shared")
        p2 = _FakeParent(persona_id="shared")
        m1 = PersonaAuthManager(
            parent=p1, check_auth_fn=lambda: state["authed"], default_poll_interval=0.01
        )
        m2 = PersonaAuthManager(
            parent=p2, check_auth_fn=lambda: state["authed"], default_poll_interval=0.01
        )
        m1.start_poll()
        m2.start_poll()
        task = PersonaAuthManager._poll_tasks["shared"]

        m1.stop()  # chat 1 closes
        await asyncio.sleep(0.02)
        assert not task.done()  # poll still running for chat 2
        assert PersonaAuthManager._poll_waiters["shared"] == {p2}

        # chat 2 signs in -> only the remaining waiter is resumed
        state["authed"] = True
        await asyncio.sleep(0.05)
        assert p2.auth_calls == 1
        assert p1.auth_calls == 0  # the closed chat is not called

    @pytest.mark.asyncio
    async def test_stop_cancels_poll_when_last_waiter_leaves(self):
        parent = _FakeParent()
        mgr = PersonaAuthManager(
            parent=parent, check_auth_fn=lambda: False, default_poll_interval=0.01
        )
        mgr.start_poll()
        task = _task_for(mgr)
        assert task is not None and not task.done()

        mgr.stop()  # sole waiter leaves -> cancels the shared task
        await _drain(task)  # let the task's finally run
        assert task.cancelled()
        assert _task_for(mgr) is None  # registry cleaned up by the task's finally


# ---------------------------------------------------------------------------
# BasePersona.on_message auth lifecycle
# ---------------------------------------------------------------------------


class _AuthGatedPersona(BasePersona):
    """A persona whose `prepare()` gates on auth, like an ACP persona."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Gate", description="", avatar_path="", system_prompt=""
        )

    async def prepare(self) -> None:
        await self.auth.assert_auth()

    async def process_message(self, message) -> None:
        self.processed.append(message)


def _make_auth_gated_persona(authed: dict):
    persona = _AuthGatedPersona.__new__(_AuthGatedPersona)
    persona.chat = MagicMock()
    persona.chat.add_message = MagicMock(return_value="m")
    persona.log = logging.getLogger("test-auth-persona")
    persona.state = MagicMock()
    persona._processing_count = 0
    persona._processing_message = None
    persona._processing_lock = None
    persona._prepare_task = None
    persona._login_terminal_opened = False
    persona.processed = []
    persona.auth = PersonaAuthManager(
        parent=persona, check_auth_fn=lambda: authed["v"], default_poll_interval=0.01
    )
    return persona


def _message():
    from jupyterlab_chat.models import Message

    msg = MagicMock(spec=Message)
    msg.metadata = {}
    return msg


class TestPreparationStateNotAuthed:
    @pytest.mark.asyncio
    async def test_prepare_not_authed_maps_to_not_authed_state(self):
        persona = _make_auth_gated_persona({"v": False})
        try:
            await persona._ensure_prepared()
        except PersonaNotAuthenticated:
            pass
        assert persona.preparation_state == PreparationState.NOT_AUTHED

    @pytest.mark.asyncio
    async def test_eager_prepare_is_silent_when_unauthed(self):
        # Selecting the persona runs prepare() eagerly. It must NOT prompt the
        # user — the whole point of the fix.
        persona = _make_auth_gated_persona({"v": False})
        try:
            await persona._ensure_prepared()
        except PersonaNotAuthenticated:
            pass
        assert persona.preparation_state == PreparationState.NOT_AUTHED
        persona.chat.add_message.assert_not_called()  # silent on selection


class TestOnMessageAuth:
    @pytest.mark.asyncio
    async def test_message_while_unauthed_prompts_and_does_not_process(self):
        persona = _make_auth_gated_persona({"v": False})

        await persona.on_message(_message())

        # A message (unlike selection) prompts for sign-in, and does not process.
        persona.chat.add_message.assert_called_once()
        assert persona.processed == []
        assert persona.preparation_state == PreparationState.NOT_AUTHED

    @pytest.mark.asyncio
    async def test_authed_message_processes(self):
        persona = _make_auth_gated_persona({"v": True})
        msg = _message()

        await persona.on_message(msg)

        assert persona.processed == [msg]
        assert persona.preparation_state == PreparationState.PREPARED

    @pytest.mark.asyncio
    async def test_resume_after_sign_in(self):
        # Unauthed message starts the resume poll; once auth flips, handle_auth
        # fires. Here we assert the poll observes the sign-in.
        state = {"v": False}
        persona = _make_auth_gated_persona(state)
        resumed = {"n": 0}

        async def _resume():
            resumed["n"] += 1

        persona.handle_auth = _resume

        await persona.on_message(_message())
        assert persona.preparation_state == PreparationState.NOT_AUTHED

        state["v"] = True
        await asyncio.sleep(0.05)
        assert resumed["n"] == 1


class TestHandleAuthTimeout:
    @pytest.mark.asyncio
    async def test_default_posts_a_nudge(self):
        persona = _make_auth_gated_persona({"v": False})
        persona.send_message = MagicMock()
        await persona.handle_auth_timeout()
        persona.send_message.assert_called_once()


class TestShutdownCleansUpAuth:
    @pytest.mark.asyncio
    async def test_shutdown_deregisters_and_cancels_last_waiter(self):
        # A persona that started its resume poll must not leave a background
        # task running once it is shut down (it is the only waiter here).
        persona = _make_auth_gated_persona({"v": False})
        persona.auth.start_poll()
        task = PersonaAuthManager._poll_tasks.get(persona.auth.poll_key)
        assert task is not None and not task.done()

        await persona.shutdown()
        await _drain(task)  # let the cancelled task's finally run

        assert task.cancelled()  # no background task persists
        assert _task_for(persona.auth) is None  # registry cleaned up


class TestOpenLoginTerminal:
    @pytest.mark.asyncio
    async def test_returns_false_when_toolkit_unavailable(self):
        # jupyterlab_commands_toolkit is not a test dependency, so the soft
        # import fails and the helper degrades to False rather than raising.
        persona = _make_auth_gated_persona({"v": False})
        assert await persona._open_login_terminal() is False
