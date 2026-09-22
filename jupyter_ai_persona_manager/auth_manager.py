"""
Authentication lifecycle for personas.

`PersonaAuthManager` owns the *mechanism* of a persona's authentication: a cached
auth check and a background poll that resumes the persona once the user signs
in. It deliberately carries no user-facing behavior — what to tell the user when
a message arrives unauthenticated, and what to do once auth succeeds, are decided
by the persona via `BasePersona.handle_message_no_auth` and
`BasePersona.handle_auth`. A persona that needs no auth uses the default
instance, whose check always passes.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Awaitable, Callable, Optional, Union

from traitlets.config import LoggingConfigurable

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


class PersonaAuthManager(LoggingConfigurable):
    """
    Owns a persona's authentication mechanism (see module docstring).

    Configure it with a `check_auth_fn` (sync or async, returning a bool). With
    none, the persona is always considered authenticated — the default for
    personas that need no sign-in. The persona is this object's traitlets
    ``parent``.
    """

    def __init__(
        self,
        *args,
        check_auth_fn: Optional[CheckAuthFn] = None,
        poll_interval: float = 1.0,
        **kwargs,
    ) -> None:
        # `parent` (the persona) is passed through to LoggingConfigurable.
        super().__init__(*args, **kwargs)
        self._check_auth_fn = check_auth_fn
        self._poll_interval = poll_interval
        self._authed = False
        self._auth_poll_task: Optional[asyncio.Task] = None

    @property
    def authed(self) -> bool:
        """The last known auth result (cached ``True`` once the check passes)."""
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
        subsequent calls short-circuit without re-running `check_auth_fn`.
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

    def start_poll(self) -> None:
        """Start polling for auth if a poll is not already running (idempotent)."""
        if self._auth_poll_task is None or self._auth_poll_task.done():
            self._auth_poll_task = asyncio.ensure_future(self.poll_for_auth())

    async def poll_for_auth(self, interval: Optional[float] = None) -> None:
        """
        Re-check auth on an interval until it succeeds, then invoke the persona's
        `handle_auth()`. Runs until the check passes or the task is cancelled
        (see `stop()` / `reset()`, called on persona shutdown).
        """
        interval = self._poll_interval if interval is None else interval
        while True:
            if await self.check_auth():
                await self.parent.handle_auth()
                return
            await asyncio.sleep(interval)

    def reset(self) -> None:
        """Forget the cached auth result and stop any running poll."""
        self.stop()
        self._authed = False

    def stop(self) -> None:
        """Cancel the auth poll task, if one is running."""
        if self._auth_poll_task is not None and not self._auth_poll_task.done():
            self._auth_poll_task.cancel()
        self._auth_poll_task = None
