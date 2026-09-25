"""
Authentication lifecycle for personas.

`PersonaAuthManager` owns the *mechanism* of a persona's authentication: a cached
auth check and a background poll that resumes the persona once the user signs
in. It deliberately carries no user-facing behavior — what to tell the user when
a message arrives unauthenticated, what to do once auth succeeds, and what to do
if the user never signs in are decided by the persona via
`BasePersona.handle_message_no_auth`, `BasePersona.handle_auth`, and
`BasePersona.handle_auth_timeout`.

This class **does nothing by default**: with no ``check_auth_fn`` the persona is
always considered authenticated, so the default instance every persona gets is
inert. It only does work once a persona passes a ``check_auth_fn``.

Polls are shared, not per-instance. Authentication is usually a system-level
resource (you are signed in to a provider for *every* chat or none), so a single
background poll is keyed by ``scope`` (a constructor argument) and shared across
persona instances:

- ``scope="global"`` (default): one poll per persona **class**, keyed by
  ``persona_id``. If the user opens three chats with the same persona and sends a
  message in each while signed out, a single poll runs and resumes all three.
- ``scope="chat"``: one poll per persona **per chat**, keyed by
  ``persona_id::chat_id`` — for the rare persona whose auth is genuinely
  per-chat.

Every poll is bounded by a timeout (``default_poll_timeout``, overridable per
`start_poll` call): it gives up after that many seconds
(each new message extends the window) and calls `handle_auth_timeout` instead of
waiting forever.
"""

from __future__ import annotations

import asyncio
from typing import (
    TYPE_CHECKING,
    Awaitable,
    Callable,
    ClassVar,
    Dict,
    Optional,
    Set,
    Union,
)

from traitlets import Float
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

    Pass a ``check_auth_fn`` (sync or async, returning a bool) to make the
    manager do work. **With none, the manager is inert**: the persona is
    considered authenticated from the start, `check_auth()` / `assert_auth()`
    pass, and `start_poll()` is a no-op. ``scope`` (``"global"`` or ``"chat"``)
    is also passed at construction. The `default_poll_timeout` and
    `default_poll_interval` timing traits are read from the persona's ``config`` (this object's traitlets
    ``parent`` is the persona).
    """

    parent: "BasePersona"  # type: ignore
    """The persona this manager serves. Set via the ``parent`` constructor kwarg
    (traitlets `LoggingConfigurable`)."""

    default_poll_interval = Float(
        1.0,
        help=(
            "Default seconds between auth re-checks while the resume poll is "
            "running, used when `start_poll` is called without an explicit "
            "interval. Configurable via traitlets."
        ),
    ).tag(config=True)

    default_poll_timeout = Float(
        180.0,
        help=(
            "Default seconds a resume poll runs before giving up and calling the "
            "persona's `handle_auth_timeout()`, used when `start_poll` is called "
            "without an explicit timeout. Each `start_poll()` call resets this "
            "countdown, so an active user keeps the poll alive. Configurable via "
            "traitlets."
        ),
    ).tag(config=True)

    # Shared across every manager instance so a `scope="global"` poll is a single
    # background task for the whole server, keyed by `poll_key`. `_poll_waiters`
    # tracks which persona instances are waiting to be resumed by each poll;
    # `_poll_deadlines` holds each poll's give-up time (absolute event-loop time).
    _poll_tasks: ClassVar[Dict[str, asyncio.Task]] = {}
    _poll_deadlines: ClassVar[Dict[str, float]] = {}
    _poll_waiters: ClassVar[Dict[str, Set["BasePersona"]]] = {}

    def __init__(
        self,
        *args,
        check_auth_fn: Optional[CheckAuthFn] = None,
        scope: str = "global",
        **kwargs,
    ) -> None:
        # `parent` (the persona) and the timing traits are passed through to
        # LoggingConfigurable.
        super().__init__(*args, **kwargs)
        if scope not in ("global", "chat"):
            raise ValueError(f"scope must be 'global' or 'chat', got {scope!r}")
        self._check_auth_fn = check_auth_fn
        self._scope = scope
        # With no check function there is nothing to authenticate against, so the
        # persona is authed from the start and every method below is a no-op.
        self._authed = check_auth_fn is None
        # The scope key never changes at runtime, so compute it exactly once.
        self._poll_key = self._compute_poll_key()

    @property
    def scope(self) -> str:
        """This poll's sharing scope: ``"global"`` or ``"chat"``."""
        return self._scope

    @property
    def poll_key(self) -> str:
        """The registry key identifying which shared poll this manager joins."""
        return self._poll_key

    def _compute_poll_key(self) -> str:
        """
        ``global`` scope keys by persona id (one poll per persona class);
        ``chat`` scope appends the chat id (one poll per persona per chat).
        """
        persona_id = self.parent.id
        if self._scope == "chat":
            return f"{persona_id}::{self.parent.chat_id}"
        return persona_id

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

    def start_poll(
        self,
        interval: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Start (or join) the background resume poll for this manager's scope.

        Re-checks every ``interval`` seconds (default `default_poll_interval`)
        and gives up after ``timeout`` seconds (default `default_poll_timeout`),
        calling the persona's `handle_auth_timeout()`. No-op when the manager has
        no ``check_auth_fn`` (nothing to poll) or is already authenticated.
        Otherwise it registers this persona instance as a waiter and (re)sets the
        timeout countdown; the poll task itself is spawned only if one is not
        already running for this scope key, so a ``global``-scoped persona runs a
        single poll across every chat.
        """
        if self._check_auth_fn is None or self._authed:
            return
        # Register this persona instance so the poll can resume it on success
        # (and notify it on timeout).
        self._poll_waiters.setdefault(self.poll_key, set()).add(self.parent)
        # Every new message extends the window rather than starting a rival poll.
        timeout = self.default_poll_timeout if timeout is None else timeout
        now = asyncio.get_running_loop().time()
        self._poll_deadlines[self.poll_key] = now + timeout
        task = self._poll_tasks.get(self.poll_key)
        if task is None or task.done():
            self._poll_tasks[self.poll_key] = asyncio.create_task(
                self._poll_for_auth(interval)
            )

    async def _poll_for_auth(self, interval: Optional[float] = None) -> None:
        """
        Re-check auth every ``interval`` seconds until it succeeds — then resume
        every waiting persona via `on_auth()` — or until the shared timeout
        deadline (set by `start_poll`) passes, in which case every waiter is
        notified via `on_auth_timeout()`. The registry entry is always cleaned up
        on exit (success, timeout, or cancellation) by the `finally` block.
        """
        interval = self.default_poll_interval if interval is None else interval
        try:
            while True:
                if await self.check_auth():
                    await self.on_auth()
                    return
                deadline = self._poll_deadlines.get(self.poll_key)
                now = asyncio.get_running_loop().time()
                if deadline is not None and now >= deadline:
                    await self.on_auth_timeout()
                    return
                await asyncio.sleep(interval)
        finally:
            # Runs on success, timeout, AND cancellation, so the shared registry
            # never leaks a finished/cancelled poll.
            self._cleanup_poll(self.poll_key)

    async def on_auth(self) -> None:
        """Resume every persona waiting on this poll by calling `handle_auth()`.
        One waiter raising does not stop the others."""
        for persona in list(self._poll_waiters.get(self.poll_key, ())):
            try:
                await persona.handle_auth()
            except Exception as exc:
                self.log.error(f"Persona '{persona.name}' raised in handle_auth().")
                self.log.exception(exc)

    async def on_auth_timeout(self) -> None:
        """Notify every persona waiting on this poll that it gave up, by calling
        `handle_auth_timeout()`. One waiter raising does not stop the others."""
        for persona in list(self._poll_waiters.get(self.poll_key, ())):
            try:
                await persona.handle_auth_timeout()
            except Exception as exc:
                self.log.error(
                    f"Persona '{persona.name}' raised in handle_auth_timeout()."
                )
                self.log.exception(exc)

    def reset(self) -> None:
        """Forget the cached auth result and stop this persona's poll."""
        self.stop()
        self._authed = self._check_auth_fn is None

    def stop(self) -> None:
        """
        Detach this persona instance from its shared poll (called on persona
        shutdown). The poll keeps running for the other chats still waiting on
        it; it is cancelled only once this persona was the **last** waiter. This
        is what keeps closing one chat from stranding the others under
        ``scope="global"``.
        """
        waiters = self._poll_waiters.get(self.poll_key)
        if not waiters or self.parent not in waiters:
            return
        waiters.discard(self.parent)
        if not waiters:
            self._cancel_poll(self.poll_key)

    @classmethod
    def _cleanup_poll(cls, poll_key: str) -> None:
        """Forget all shared registry state for a scope key."""
        cls._poll_tasks.pop(poll_key, None)
        cls._poll_deadlines.pop(poll_key, None)
        cls._poll_waiters.pop(poll_key, None)

    @classmethod
    def _cancel_poll(cls, poll_key: str) -> None:
        """
        Cancel the poll task for a scope key and forget its registry state. The
        task's own `finally` also runs `_cleanup_poll` on normal exit, but a task
        cancelled *before its first execution* never runs its `finally`, so the
        cleanup is done here too (idempotent).
        """
        task = cls._poll_tasks.get(poll_key)
        if task is not None and not task.done():
            task.cancel()
        cls._cleanup_poll(poll_key)
