"""
Fixture persona for E2E tests: a `BasePersona` whose initialization never
completes, so the chat toolbar stays on its "loading personas" placeholder (see
tests/loading.spec.ts, issue #77).

## What keeps the placeholder up

The persona list the browser renders comes from the `PersonaManager`'s awareness
slot, which the manager publishes at the *very end* of its constructor — after it
has instantiated every persona. While that slot is absent the frontend shows the
loading placeholder (`PersonaManagerAwareness.from()` polls for the slot for ~60s
before giving up). So to pin the toolbar in its loading state we need the
manager, for this chat, to never reach the publish step.

## Why this raises instead of blocking

The obvious approach — block forever in `__init__` — does not work on the shared
test server: `PersonaManager` is constructed *synchronously on the server's event
loop* (a chat-init event → `router.connect_chat` → `PersonaManager(...)`). A
blocking `__init__` freezes that single loop, hanging the whole server and every
other suite running against it. (Verified: it does exactly that — one blocking
persona took the entire test run down.)

Instead this persona *aborts* its initialization by raising a `BaseException`
subclass. That it is a `BaseException`, not an ordinary `Exception`, is
load-bearing: the manager instantiates each persona inside a `try/except
Exception`, so an ordinary exception would merely skip this persona and let the
manager publish a (persona-less) list — which *hides* the placeholder. A
`BaseException` escapes that `except Exception`, so `PersonaManager.__init__`
unwinds before publishing its slot, and the extension's `_init_persona_manager`
(an `except Exception` plus a `finally: return`) discards the half-built manager
and returns `None` without re-raising. Net effect: no manager slot is ever
published for this chat, the event loop is never blocked, and the placeholder
stays up for the poll window — a *reproducible* loading state with no timing to
race.

This couples the fixture to that control flow (a persona-init `except Exception`
that lets `BaseException` through, and a `finally: return` that swallows it). If
either changes, this test fails loudly — the picker would resolve instead of
staying in the loading state — which is the signal to revisit the fixture.

## Out of scope

The eventual give-up/timeout behavior — the toolbar reporting that a persona
never finished loading — is intentionally NOT tested here. Per issue #77 that
waits until persona init is split into a lazy `async init_session()`, so a persona
no longer does all its work up front in `__init__`.

Not part of the shipped package; see AGENTS.md.
"""

import os

from jupyter_ai_persona_manager import BasePersona, PersonaDefaults
from jupyterlab_chat.models import Message

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")


class _PersonaNeverLoads(BaseException):
    """Raised to abort init so the manager never publishes its persona list.

    A `BaseException` (not `Exception`) so it escapes the manager's
    `try/except Exception` around persona instantiation — see the module
    docstring for why that matters.
    """


class LoadingPersona(BasePersona):
    """Test-only persona whose initialization never completes."""

    def __init__(self, *args, **kwargs) -> None:
        # Abort before `BasePersona.__init__` does anything (no awareness slot,
        # no chat-user registration): raising here unwinds the manager's
        # constructor before it publishes its awareness slot, so the toolbar
        # stays on the loading placeholder. See the module docstring.
        raise _PersonaNeverLoads("Loading Persona never finishes initializing.")

    @property
    def defaults(self) -> PersonaDefaults:
        # Abstract on BasePersona, so it must be implemented for the class to be
        # instantiable at all (an abstract class raises TypeError on
        # instantiation — an ordinary Exception the manager would just skip,
        # hiding the placeholder). Never actually called: `__init__` always
        # raises first.
        return PersonaDefaults(
            name="Loading Persona",
            description="Test-only persona that never finishes initializing.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def process_message(self, message: Message) -> None:
        # Abstract on BasePersona; unreachable (see `defaults`).
        pass
