"""Tests for the persona message REST endpoint (``/api/ai/message/<persona>``).

The endpoint spins up a throwaway chat, routes a single message to the named
persona, and returns its output. It must work with or without a
``file_id_manager`` present (issue #125): the room is a plain uuid and the chat
model comes from the ``ChatManager`` (or a fallback in-memory ``YChat``), so no
file-id indexing is required.
"""

import json

import pytest
from jupyterlab_chat.models import Message, NewMessage

from jupyter_ai_persona_manager.base_persona import BasePersona, PersonaDefaults
from jupyter_ai_persona_manager.persona_manager import PersonaManager


class _EchoPersona(BasePersona):
    """Minimal persona that echoes the incoming message back into the chat."""

    @property
    def defaults(self):
        return PersonaDefaults(
            name="EchoPersona",
            description="echoes messages",
            avatar_path="/tmp/echo.svg",
            system_prompt="echo",
        )

    async def process_message(self, message: Message) -> None:
        self.chat.add_message(
            NewMessage(body="echo: " + message.body, sender=self.id)
        )

    async def shutdown(self) -> None:
        # Keep teardown trivial; the base awareness cleanup isn't needed here.
        pass


@pytest.fixture
def inject_echo_persona(monkeypatch):
    """Make the temporary PersonaManager created by MessageHandler load the echo
    persona (personas normally come from installed entry points)."""
    monkeypatch.setattr(
        PersonaManager,
        "_ep_persona_classes",
        [{"module": "echo", "persona_class": _EchoPersona, "traceback": None}],
    )


async def test_returns_persona_response(jp_fetch, jp_serverapp, inject_echo_persona):
    """Happy path: the endpoint returns the persona's chat output."""
    response = await jp_fetch(
        "api", "ai", "message", "EchoPersona",
        method="POST", body=json.dumps({"message": "hello"}),
    )
    assert response.code == 200
    assert "echo: hello" in json.loads(response.body)["response"]


async def test_works_without_file_id_manager(
    jp_fetch, jp_serverapp, inject_echo_persona
):
    """The endpoint must not depend on a `file_id_manager` (#125). Remove it
    from settings to prove the request still succeeds."""
    jp_serverapp.web_app.settings.pop("file_id_manager", None)
    response = await jp_fetch(
        "api", "ai", "message", "EchoPersona",
        method="POST", body=json.dumps({"message": "ping"}),
    )
    assert response.code == 200
    assert "echo: ping" in json.loads(response.body)["response"]


async def test_404_for_unknown_persona(jp_fetch, jp_serverapp, inject_echo_persona):
    """An unknown persona name is a clean 404, not a 500."""
    from tornado.httpclient import HTTPClientError

    with pytest.raises(HTTPClientError) as exc:
        await jp_fetch(
            "api", "ai", "message", "NoSuchPersona",
            method="POST", body=json.dumps({"message": "hi"}),
        )
    assert exc.value.code == 404


async def test_400_for_missing_message_field(
    jp_fetch, jp_serverapp, inject_echo_persona
):
    """A body with no `message` field is a 400."""
    from tornado.httpclient import HTTPClientError

    with pytest.raises(HTTPClientError) as exc:
        await jp_fetch(
            "api", "ai", "message", "EchoPersona",
            method="POST", body=json.dumps({}),
        )
    assert exc.value.code == 400
