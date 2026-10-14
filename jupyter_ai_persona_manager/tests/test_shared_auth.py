"""
Tests for auth state shared across a persona's chats, and its publication on
`state.auth`.
"""

import asyncio
import logging
from unittest.mock import MagicMock

import pytest

from jupyter_ai_persona_manager import PersonaAuthManager
from jupyter_ai_persona_manager.base_persona import BasePersona, PersonaDefaults
from jupyter_ai_persona_manager.persona_events import PersonaSessionState


class _SharedPersona(BasePersona):
    """A persona whose instances (one per chat) share their auth state."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Shared", description="", avatar_path="", system_prompt=""
        )

    async def process_message(self, message) -> None:
        pass

    async def handle_auth(self) -> None:
        self.auth_calls += 1


def _make_persona(chat_id="chat-1", check_auth_fn=None):
    """A `_SharedPersona` for one chat, with a real (non-emitting) state so the
    published `state.auth` can be asserted on."""
    persona = _SharedPersona.__new__(_SharedPersona)
    persona.chat = MagicMock()
    persona.log = logging.getLogger("test-shared-auth")
    persona._processing_count = 0
    persona._processing_message = None
    persona._processing_lock = None
    persona._prepare_task = None
    persona._login_terminal_opened = False
    persona.auth_calls = 0
    persona.state = PersonaSessionState(
        event_logger=None, persona_id=persona.id, chat_id=chat_id, log=persona.log
    )
    persona.auth = PersonaAuthManager(
        parent=persona, check_auth_fn=check_auth_fn, default_poll_interval=0.01
    )
    return persona


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0)


class TestSharedAuthState:
    @pytest.mark.asyncio
    async def test_auth_result_is_shared_across_chats(self):
        calls = {"a": 0, "b": 0}

        def check_a():
            calls["a"] += 1
            return True

        def check_b():
            calls["b"] += 1
            return False

        a = _make_persona("chat-a", check_auth_fn=check_a)
        b = _make_persona("chat-b", check_auth_fn=check_b)

        assert await a.auth.check_auth() is True
        # Chat B reuses chat A's result instead of running its own check.
        assert await b.auth.check_auth() is True
        assert calls == {"a": 1, "b": 0}

    @pytest.mark.asyncio
    async def test_set_authed_false_publishes_to_every_chat(self):
        a = _make_persona("chat-a")
        b = _make_persona("chat-b")

        await a.auth.set_authed(False)

        for persona in (a, b):
            assert persona.state.auth.status == "not_authed"
        assert b.auth.authed is False

    @pytest.mark.asyncio
    async def test_set_authed_true_with_via(self):
        persona = _make_persona()
        await persona.auth.set_authed(True, via="Bedrock")
        assert persona.state.auth.status == "authed"
        assert persona.state.auth.via == "Bedrock"

    @pytest.mark.asyncio
    async def test_inert_persona_publishes_nothing(self):
        persona = _make_persona()
        assert await persona.auth.check_auth() is True
        assert persona.state.auth is None

    @pytest.mark.asyncio
    async def test_new_chat_starts_with_the_published_state(self):
        a = _make_persona("chat-a")
        await a.auth.set_authed(False)

        b = _make_persona("chat-b")

        assert b.state.auth.status == "not_authed"

    @pytest.mark.asyncio
    async def test_detached_persona_is_no_longer_updated(self):
        a = _make_persona("chat-a")
        b = _make_persona("chat-b")
        b.auth.detach()

        await a.auth.set_authed(False)

        assert a.state.auth.status == "not_authed"
        assert b.state.auth is None

    @pytest.mark.asyncio
    async def test_reset_forgets_the_shared_result(self):
        state = {"v": True}
        a = _make_persona("chat-a", check_auth_fn=lambda: state["v"])
        b = _make_persona("chat-b", check_auth_fn=lambda: state["v"])
        assert await a.auth.check_auth() is True

        state["v"] = False
        b.auth.reset()

        assert await a.auth.check_auth() is False


class TestCheckAuthRules:
    @pytest.mark.asyncio
    async def test_without_a_check_fn_requests_are_not_blocked(self):
        # Such a persona learns it is signed in again only from its backend
        # accepting a request (e.g. after a terminal login), so it must not
        # stay locked once it reported being signed out.
        persona = _make_persona()
        await persona.auth.set_authed(False)
        assert await persona.auth.check_auth() is True
        await persona.auth.assert_auth()  # does not raise

    @pytest.mark.asyncio
    async def test_a_passing_check_does_not_clear_a_reported_sign_out(self):
        # e.g. a cached token says signed in, but the backend rejected a
        # request: let requests through, but keep showing "Login required"
        # until one succeeds (`set_authed(True)`).
        persona = _make_persona(check_auth_fn=lambda: True)
        await persona.auth.set_authed(False)

        assert await persona.auth.check_auth() is True
        assert persona.state.auth.status == "not_authed"

        await persona.auth.set_authed(True)
        assert persona.state.auth.status == "authed"

    @pytest.mark.asyncio
    async def test_an_outdated_failed_check_does_not_undo_a_sign_in(self):
        release = asyncio.Event()

        async def slow_check():
            await release.wait()
            return False

        a = _make_persona("chat-a", check_auth_fn=slow_check)
        check = asyncio.ensure_future(a.auth.check_auth())
        await _settle()

        # Another chat records the sign-in meanwhile.
        await a.auth.set_authed(True)
        release.set()

        assert await check is True
        assert a.state.auth.status == "authed"


class TestPoll:
    @pytest.mark.asyncio
    async def test_the_poll_clears_a_reported_sign_out(self):
        state = {"v": False}
        persona = _make_persona(check_auth_fn=lambda: state["v"])
        await persona.auth.set_authed(False)
        persona.auth.start_poll()

        state["v"] = True
        await asyncio.sleep(0.05)

        assert persona.state.auth.status == "authed"
        assert persona.auth_calls == 1

    @pytest.mark.asyncio
    async def test_the_poll_logs_a_failing_resume(self, caplog):
        state = {"v": False}
        persona = _make_persona(check_auth_fn=lambda: state["v"])

        async def failing_resume():
            raise RuntimeError("resume failed")

        persona.handle_auth = failing_resume
        persona.auth.start_poll()
        poll = persona.auth._auth_poll_task

        with caplog.at_level(logging.ERROR):
            state["v"] = True
            await asyncio.sleep(0.05)

        assert poll.done() and poll.exception() is None
        assert "raised in handle_auth() after sign-in" in caplog.text
