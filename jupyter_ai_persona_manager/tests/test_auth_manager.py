"""
Tests for the persona authentication lifecycle: the `PersonaAuthManager`
mechanism and how `BasePersona.on_message` reacts to an unauthenticated
`prepare()` (the NOT_AUTHED state).
"""

import asyncio
import logging
from unittest.mock import MagicMock

import pytest
from traitlets.config import LoggingConfigurable

from jupyter_ai_persona_manager import (
    PersonaAuthManager,
    PersonaNotAuthenticated,
    PreparationState,
)
from jupyter_ai_persona_manager.base_persona import BasePersona, PersonaDefaults


# ---------------------------------------------------------------------------
# PersonaAuthManager (mechanism)
# ---------------------------------------------------------------------------


class _FakeParent(LoggingConfigurable):
    """A minimal Configurable to stand in for the persona a manager serves."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.auth_calls = 0
        self.last_was_unauthenticated = None

    async def handle_auth(self, was_unauthenticated: bool = False):
        self.auth_calls += 1
        self.last_was_unauthenticated = was_unauthenticated


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
        # The poll only runs while unauthenticated, so the resume it drives
        # always signals that the persona was signed out.
        assert parent.last_was_unauthenticated is True
        mgr.stop()

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
        assert mgr._auth_poll_task is None

    @pytest.mark.asyncio
    async def test_start_poll_noop_when_already_authed(self):
        mgr = PersonaAuthManager(parent=_FakeParent(), check_auth_fn=lambda: True)
        assert await mgr.check_auth() is True  # caches authed
        mgr.start_poll()
        assert mgr._auth_poll_task is None  # no poll: already authenticated

    def test_default_poll_interval_is_configurable_trait(self):
        from traitlets.config import Config

        cfg = Config()
        cfg.PersonaAuthManager.default_poll_interval = 2.5
        mgr = PersonaAuthManager(parent=_FakeParent(), config=cfg)
        assert mgr.default_poll_interval == 2.5

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
        mgr.stop()


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
        persona.auth.stop()

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
        persona.auth.stop()


class TestOnMessageAuth:
    @pytest.mark.asyncio
    async def test_message_while_unauthed_prompts_and_does_not_process(self):
        persona = _make_auth_gated_persona({"v": False})

        await persona.on_message(_message())

        # A message (unlike selection) prompts for sign-in, and does not process.
        persona.chat.add_message.assert_called_once()
        assert persona.processed == []
        assert persona.preparation_state == PreparationState.NOT_AUTHED
        persona.auth.stop()

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
        resumed = {"n": 0, "was_unauthenticated": None}

        async def _resume(was_unauthenticated: bool = False):
            resumed["n"] += 1
            resumed["was_unauthenticated"] = was_unauthenticated

        persona.handle_auth = _resume

        await persona.on_message(_message())
        assert persona.preparation_state == PreparationState.NOT_AUTHED

        state["v"] = True
        await asyncio.sleep(0.05)
        assert resumed["n"] == 1
        # Resumed via the poll after a message arrived while signed out.
        assert resumed["was_unauthenticated"] is True
        persona.auth.stop()


class TestShutdownCleansUpAuth:
    @pytest.mark.asyncio
    async def test_shutdown_cancels_auth_poll(self):
        # A persona that started its resume poll must not leave a background
        # task running once it is shut down.
        persona = _make_auth_gated_persona({"v": False})
        persona.auth.start_poll()
        task = persona.auth._auth_poll_task
        assert task is not None and not task.done()

        await persona.shutdown()

        assert persona.auth._auth_poll_task is None  # handle cleared
        await asyncio.sleep(0)  # let the cancellation propagate
        assert task.cancelled()  # no background task persists


class TestOpenLoginTerminal:
    @pytest.mark.asyncio
    async def test_returns_false_when_toolkit_unavailable(self):
        # jupyterlab_commands_toolkit is not a test dependency, so the soft
        # import fails and the helper degrades to False rather than raising.
        persona = _make_auth_gated_persona({"v": False})
        assert await persona._open_login_terminal() is False
        persona.auth.stop()
