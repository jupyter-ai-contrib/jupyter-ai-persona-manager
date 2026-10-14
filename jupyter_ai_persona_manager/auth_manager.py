"""
Authentication lifecycle for personas.

`PersonaAuthManager` owns the *mechanism* of a persona's authentication: a cached
auth check and a background poll that resumes the persona once the user signs
in. It deliberately carries no user-facing behavior — what to tell the user when
a message arrives unauthenticated, and what to do once auth succeeds, are decided
by the persona via `BasePersona.handle_message_no_auth` and
`BasePersona.handle_auth`.

This class **does nothing by default**: with no ``check_auth_fn`` the persona is
always considered authenticated, so the default instance every persona gets is
inert. It only does work once a persona passes a ``check_auth_fn``, or reports
its auth state with `set_authed()`.

The auth state is **shared by every instance of a persona** across chats (keyed
by persona ID), since the user is usually signed in to a provider for every
chat or none: signing in from one chat counts for all of them. It is published
to the frontend on each instance's `state.auth`.
"""

from __future__ import annotations

import asyncio
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable, ClassVar, Optional, Union

from traitlets import Float
from traitlets.config import LoggingConfigurable

from .awareness_models import AuthState

if TYPE_CHECKING:
    from .base_persona import BasePersona

# A function returning whether the user is authenticated. May be sync or async.
CheckAuthFn = Callable[[], Union[bool, Awaitable[bool]]]


class PersonaNotAuthenticated(Exception):
    """
    Raised by `PersonaAuthManager.assert_auth()` — typically from a persona's
    `prepare()` — when the user is not signed in.

    `BasePersona.preparation_state` maps a `prepare()` task that failed with this
    exception to `PreparationState.NOT_AUTHED`, which `BasePersona.on_message`
    handles by prompting the user to sign in, as opposed to a generic `FAILED`.
    """


@dataclass(eq=False)
class _SharedAuth:
    """The auth state shared by every instance of one persona."""

    authed: Optional[bool] = None
    """Whether the user is signed in, or None if not known yet."""
    via: Optional[str] = None
    """How the user is signed in, when known."""
    state: Optional[AuthState] = None
    """The last published state, used to bring new instances up to date."""
    personas: "weakref.WeakSet[BasePersona]" = field(default_factory=weakref.WeakSet)
    """The live instances of the persona, one per chat."""


class PersonaAuthManager(LoggingConfigurable):
    """
    Owns a persona's authentication mechanism (see module docstring).

    Pass a ``check_auth_fn`` (sync or async, returning a bool) to make the
    manager do work. **With none, the manager is inert**: the persona is
    considered authenticated from the start, `check_auth()` / `assert_auth()`
    pass, and `start_poll()` is a no-op. The persona is this object's traitlets
    ``parent``; configurable traits (e.g. `default_poll_interval`) are read from
    the persona's ``config`` when it is threaded through at construction.
    """

    _shared_by_persona: ClassVar[dict[str, _SharedAuth]] = {}
    """The shared auth state of each persona, keyed by persona ID."""

    default_poll_interval = Float(
        1.0,
        help=(
            "Default seconds between auth re-checks while the resume poll is "
            "running, used when `start_poll` is called without an explicit "
            "interval. Configurable via traitlets."
        ),
    ).tag(config=True)

    def __init__(
        self,
        *args,
        check_auth_fn: Optional[CheckAuthFn] = None,
        **kwargs,
    ) -> None:
        # `parent` (the persona) and any configurable traits (e.g.
        # `default_poll_interval`) are passed through to LoggingConfigurable.
        super().__init__(*args, **kwargs)
        self._check_auth_fn = check_auth_fn
        self._auth_poll_task: Optional[asyncio.Task] = None
        self._shared = self._get_shared(self.parent)
        if self.parent is not None:
            self._shared.personas.add(self.parent)
        self.sync_state()

    @classmethod
    def _get_shared(cls, persona: object) -> _SharedAuth:
        """Return the auth state shared by ``persona``'s instances. A parent
        without a persona ID (e.g. in unit tests) gets a private one."""
        persona_id = getattr(persona, "id", None)
        if not isinstance(persona_id, str) or not persona_id:
            return _SharedAuth()
        shared = cls._shared_by_persona.get(persona_id)
        if shared is None:
            shared = cls._shared_by_persona[persona_id] = _SharedAuth()
        return shared

    @property
    def authed(self) -> bool:
        """Whether the user is currently considered authenticated (cached once
        the check passes; ``True`` when no ``check_auth_fn`` is set and nothing
        reported otherwise)."""
        if self._shared.authed is not None:
            return self._shared.authed
        return self._check_auth_fn is None

    async def _run_check(self) -> bool:
        if self._check_auth_fn is None:
            return True
        result = self._check_auth_fn()
        if asyncio.iscoroutine(result):
            result = await result
        return bool(result)

    async def check_auth(self) -> bool:
        """
        Return whether the user is authenticated. A ``True`` result is cached
        and shared with the persona's other chats, so subsequent calls
        short-circuit without re-running ``check_auth_fn``.

        A manager with no ``check_auth_fn`` always passes: such a persona learns
        it is signed out from its backend,
        and only another request to that backend can tell it the user signed
        back in, so its requests must not be blocked.
        """
        if self._shared.authed:
            return True
        if self._check_auth_fn is None:
            return True
        authed = await self._run_check()
        if self._shared.authed:
            return True
        if not authed:
            await self.set_authed(False)
        elif self._shared.authed is None:
            await self.set_authed(True)
        return authed

    async def assert_auth(self) -> None:
        """Raise `PersonaNotAuthenticated` if the user is not authenticated."""
        if not await self.check_auth():
            raise PersonaNotAuthenticated()

    async def set_authed(self, authed: bool, *, via: Optional[str] = None) -> None:
        """
        Record whether the user is signed in, for every chat using this persona,
        and publish it. Call with ``False`` when the persona learns the user is
        signed out (e.g. a request was rejected as unauthenticated), with
        ``True`` once a request succeeds again, and optionally with
        ``True, via=...`` when credentials are configured outside the persona
        (e.g. "Bedrock").
        """
        self._shared.authed = authed
        self._shared.via = via if authed else None
        self._publish()

    def start_poll(self, interval: Optional[float] = None) -> None:
        """
        Start the background resume poll if it makes sense to.

        Re-checks every ``interval`` seconds, defaulting to
        `default_poll_interval` when ``interval`` is None so a consumer can pass
        a custom cadence. No-op when the manager has no ``check_auth_fn``
        (nothing to poll) or is already authenticated. Otherwise it spawns
        `_poll_for_auth` once; idempotent, so a poll already running is left
        alone.
        """
        if self._check_auth_fn is None or self.authed:
            return
        if self._auth_poll_task is None or self._auth_poll_task.done():
            self._auth_poll_task = asyncio.create_task(self._poll_for_auth(interval))

    async def _poll_for_auth(self, interval: Optional[float] = None) -> None:
        """
        Re-check auth every ``interval`` seconds (defaulting to
        `default_poll_interval`) until it succeeds, then invoke the persona's
        `handle_auth()`. Runs until the check passes or the task is cancelled
        (see `stop()` / `reset()`, called on persona shutdown).
        """
        interval = self.default_poll_interval if interval is None else interval
        while True:
            if await self.check_auth():
                # The user signed in while the poll waited. Unlike a single
                # check, the poll trusts the check: a persona
                # starts it only when its check is reliable.
                if not self._shared.authed:
                    await self.set_authed(True)
                # Nothing awaits this task, so log a failure rather than lose it.
                try:
                    await self.parent.handle_auth()
                except Exception:
                    self.log.exception(
                        f"Persona '{getattr(self.parent, 'name', self.parent)}' "
                        "raised in handle_auth() after sign-in."
                    )
                return
            await asyncio.sleep(interval)

    def reset(self) -> None:
        """Forget the cached auth result (for every chat using this persona) and
        stop this instance's running poll."""
        self.stop()
        self._shared.authed = None
        self._shared.via = None

    def stop(self) -> None:
        """Cancel the auth poll task, if one is running."""
        if self._auth_poll_task is not None and not self._auth_poll_task.done():
            self._auth_poll_task.cancel()
        self._auth_poll_task = None

    def detach(self) -> None:
        """Stop publishing to this persona instance (called on persona
        shutdown)."""
        if self.parent is not None:
            self._shared.personas.discard(self.parent)

    def sync_state(self) -> None:
        """Bring this persona instance's `state.auth` up to date with the shared
        state, e.g. for a chat opened after the state was published."""
        state = self._shared.state
        persona_state = getattr(self.parent, "state", None)
        if state is not None and persona_state is not None:
            persona_state.auth = state

    def _publish(self) -> None:
        """Publish the shared state to every instance of the persona."""
        shared = self._shared
        state = AuthState(
            status="not_authed" if shared.authed is False else "authed",
            via=shared.via if shared.authed else None,
        )
        shared.state = state
        for persona in list(shared.personas):
            persona_state = getattr(persona, "state", None)
            if persona_state is not None:
                persona_state.auth = state
