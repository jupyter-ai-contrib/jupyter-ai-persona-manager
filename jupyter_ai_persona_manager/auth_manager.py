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
inert. It only does work once a persona passes a ``check_auth_fn``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Optional, Union

from traitlets import Float
from traitlets.config import LoggingConfigurable

if TYPE_CHECKING:
    from .base_persona import BasePersona

# A function returning whether the user is authenticated. May be sync or async.
CheckAuthFn = Callable[[], Union[bool, Awaitable[bool]]]


@dataclass
class PersonaAuthSpec:
    """
    Declarative description of how a persona authenticates.

    A persona passes this to ``super().__init__(auth_spec=...)`` and `BasePersona`
    hands it to the persona's `PersonaAuthManager`. It is pure data — no
    ``parent``, no running poll — so a subclass can construct it *before* the
    persona itself exists, and the persona is what binds the resulting manager
    to itself. With the default (no ``check_auth_fn``) the persona is inert:
    always considered authenticated.
    """

    check_auth_fn: Optional[CheckAuthFn] = None


class PersonaNotAuthenticated(Exception):
    """
    Raised by `PersonaAuthManager.assert_auth()` — typically from a persona's
    `prepare()` — when the user is not signed in.

    `BasePersona.preparation_state` maps a `prepare()` task that failed with this
    exception to `PreparationState.NOT_AUTHED`, which `BasePersona.on_message`
    handles by prompting the user to sign in, as opposed to a generic `FAILED`.
    """


class PersonaAuthManager(LoggingConfigurable):
    """
    Owns a persona's authentication mechanism (see module docstring).

    Pass a `PersonaAuthSpec` via ``spec`` whose ``check_auth_fn`` (sync or async,
    returning a bool) makes the manager do work. **With no spec (or a spec with
    no ``check_auth_fn``), the manager is inert**: the persona is considered
    authenticated from the start, `check_auth()` / `assert_auth()` pass, and
    `start_poll()` is a no-op. The persona is this object's traitlets ``parent``;
    configurable traits (e.g. `default_poll_interval`) are read from the
    persona's ``config`` when it is threaded through at construction.
    """

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
        spec: Optional[PersonaAuthSpec] = None,
        **kwargs,
    ) -> None:
        # `parent` (the persona) and any configurable traits (e.g.
        # `default_poll_interval`) are passed through to LoggingConfigurable.
        super().__init__(*args, **kwargs)
        # The spec declares how this persona authenticates (see PersonaAuthSpec);
        # an absent spec means the default, inert configuration.
        self._spec = spec or PersonaAuthSpec()
        self._check_auth_fn = self._spec.check_auth_fn
        # With no check function there is nothing to authenticate against, so the
        # persona is authed from the start and every method below is a no-op.
        self._authed = self._check_auth_fn is None
        self._auth_poll_task: Optional[asyncio.Task] = None

    @property
    def authed(self) -> bool:
        """Whether the user is currently considered authenticated (cached once
        the check passes; always ``True`` when no ``check_auth_fn`` is set)."""
        return self._authed

    async def _run_check(self) -> bool:
        if self._check_auth_fn is None:
            return True
        result = self._check_auth_fn()
        if asyncio.iscoroutine(result):
            result = await result
        return bool(result)

    async def check_auth(self) -> bool:
        """
        Return whether the user is authenticated. A ``True`` result is cached, so
        subsequent calls short-circuit without re-running ``check_auth_fn`` — and
        a manager with no ``check_auth_fn`` is authed from the start.
        """
        if self._authed:
            return True
        if await self._run_check():
            self._authed = True
        return self._authed

    async def assert_auth(self) -> None:
        """Raise `PersonaNotAuthenticated` if the user is not authenticated."""
        if not await self.check_auth():
            raise PersonaNotAuthenticated()

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
        if self._check_auth_fn is None or self._authed:
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
                await self.parent.handle_auth()
                return
            await asyncio.sleep(interval)

    def reset(self) -> None:
        """Forget the cached auth result and stop any running poll."""
        self.stop()
        self._authed = self._check_auth_fn is None

    def stop(self) -> None:
        """Cancel the auth poll task, if one is running."""
        if self._auth_poll_task is not None and not self._auth_poll_task.done():
            self._auth_poll_task.cancel()
        self._auth_poll_task = None
