"""
Tests for the @-mention /command dispatch platform on `BasePersona` and
`PersonaManager._broadcast` / `_extract_persona_command`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from jupyter_ai_persona_manager.base_persona import (
    BasePersona,
    PersonaCommand,
    persona_command,
)
from jupyter_ai_persona_manager.persona_manager import PersonaManager


# --- @persona_command + dispatch ---------------------------------------------


class _DummyAwareness:
    def shutdown(self) -> None:
        pass


class _Persona(BasePersona):
    """A minimal persona that bypasses the real `BasePersona.__init__` so we
    can exercise the command machinery without a YChat / awareness setup."""

    def __init__(self) -> None:
        # Skip BasePersona.__init__ entirely; populate only what we need.
        self.calls: list[tuple[str, str, Any]] = []
        self._init_commands_for_test()

    def _init_commands_for_test(self) -> None:
        self._commands = {}
        for klass in reversed(type(self).__mro__):
            for attr_name, attr in vars(klass).items():
                meta = getattr(attr, "_persona_command_meta", None)
                if not meta:
                    continue
                self._commands[meta["name"]] = {
                    "method_name": attr_name,
                    "description": meta.get("description", ""),
                }

    @property
    def name(self) -> str:  # type: ignore[override]
        return "Test"

    @property
    def defaults(self):  # pragma: no cover - unused in these tests
        raise NotImplementedError

    async def process_message(self, message):  # pragma: no cover - unused
        raise NotImplementedError

    @persona_command("ping", description="Reply with pong")
    async def cmd_ping(self, args: str, message) -> None:
        self.calls.append(("ping", args, message))

    @persona_command("/echo", description="Echo args")
    def cmd_echo(self, args: str, message) -> None:
        self.calls.append(("echo", args, message))


def test_get_commands_lists_decorated_methods() -> None:
    p = _Persona()
    cmds = {c.name: c for c in p.get_commands()}
    assert set(cmds) == {"ping", "echo"}
    assert isinstance(cmds["ping"], PersonaCommand)
    assert cmds["ping"].description == "Reply with pong"
    assert cmds["echo"].description == "Echo args"


def test_has_command_strips_leading_slash() -> None:
    p = _Persona()
    assert p.has_command("ping")
    assert p.has_command("/ping")
    assert not p.has_command("nope")


def test_dispatch_command_invokes_async_handler() -> None:
    p = _Persona()
    asyncio.run(p.dispatch_command("ping", "hello world", object()))
    assert p.calls == [("ping", "hello world", p.calls[0][2])]


def test_dispatch_command_invokes_sync_handler() -> None:
    p = _Persona()
    sentinel = object()
    asyncio.run(p.dispatch_command("/echo", "abc", sentinel))
    assert p.calls == [("echo", "abc", sentinel)]


def test_dispatch_command_unknown_raises() -> None:
    p = _Persona()
    with pytest.raises(ValueError):
        asyncio.run(p.dispatch_command("missing", "", object()))


# --- _extract_persona_command ------------------------------------------------


@dataclass
class _Msg:
    body: str = ""
    deleted: bool = False
    sender: str = "human"
    mentions: list = field(default_factory=list)


@pytest.mark.parametrize(
    "body,expected",
    [
        ("@bot /help", ("help", "")),
        ("@bot /help me please", ("help", "me please")),
        ("  @bot   /run arg1 arg2  ", ("run", "arg1 arg2  ")),
        ("@team @bot /ping", ("ping", "")),
        ("@bot hello", None),
        ("/help directly", ("help", "directly")),
        ("hello world", None),
        ("@bot /", None),
        ("", None),
    ],
)
def test_extract_persona_command(body, expected) -> None:
    msg = _Msg(body=body)
    assert PersonaManager._extract_persona_command(msg) == expected
