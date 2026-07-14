"""
End-to-end tests that a persona's awareness state and the manager's persona
list actually land in the shared awareness map, keyed by the expected Yjs
client IDs, using a real YChat + pycrdt Awareness (no mocking of the awareness
layer).
"""

import logging

import pytest
from jupyterlab_chat.ychat import YChat
from pycrdt import Awareness

from jupyter_ai_persona_manager.awareness_models import (
    CommandOption,
    ModelConfiguration,
    ModelOption,
    Usage,
)
from jupyter_ai_persona_manager.base_persona import BasePersona, PersonaDefaults


class _Persona(BasePersona):
    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Bot",
            description="d",
            avatar_path="",
            system_prompt="",
        )

    async def process_message(self, message):
        pass

    async def update_model(self, model_id: str) -> None:
        pass

    async def update_model_settings(self, settings) -> None:
        pass

    async def update_settings(self, settings) -> None:
        pass


@pytest.fixture
def ychat():
    chat = YChat()
    chat.awareness = Awareness(ydoc=chat._ydoc)
    return chat


def _make_persona(ychat) -> _Persona:
    """Instantiate a persona against a real YChat. Uses __new__ + manual wiring
    to skip the heartbeat task and parent dependencies, but keeps a real
    PersonaAwareness so writes hit the shared awareness map."""
    from jupyter_ai_persona_manager.awareness_models import PersonaAwarenessState
    from jupyter_ai_persona_manager.persona_awareness import PersonaAwareness

    persona = _Persona.__new__(_Persona)
    persona.ychat = ychat
    persona.log = logging.getLogger("test-persona")
    persona.parent = None
    # Real awareness, fixed client ID for deterministic assertions. Skip the
    # heartbeat by constructing then cancelling it.
    persona.awareness = PersonaAwareness(
        ychat=ychat, log=persona.log, user=None, client_id=4242
    )
    persona.awareness._heartbeat_task.cancel()
    persona._awareness_state = PersonaAwarenessState(id="bot")
    persona._broadcast_awareness_state()
    return persona


class TestPersonaAwarenessRoundTrip:
    """Each field of the state is a top-level entry of the persona's awareness
    slot (the slot *is* the state), so assertions index the slot directly."""

    async def test_initial_state_present_in_awareness_map(self, ychat):
        persona = _make_persona(ychat)
        state = ychat.awareness.states[persona.awareness.client_id]
        assert state["id"] == "bot"

    async def test_set_configuration_visible_in_map(self, ychat):
        persona = _make_persona(ychat)
        persona.set_configuration(
            ModelConfiguration(current="opus", options=[ModelOption(id="opus")]),
            [],
        )
        state = ychat.awareness.states[persona.awareness.client_id]
        assert state["model"]["current"] == "opus"
        assert state["model"]["options"][0]["id"] == "opus"

    async def test_update_usage_visible_in_map(self, ychat):
        persona = _make_persona(ychat)
        persona.update_usage(Usage(input_tokens=7, context_size=1000))
        usage = ychat.awareness.states[persona.awareness.client_id]["usage"]
        assert usage["input_tokens"] == 7
        assert usage["context_size"] == 1000

    async def test_update_slash_commands_visible_in_map(self, ychat):
        persona = _make_persona(ychat)
        persona.update_slash_commands([CommandOption(name="/compact")])
        cmds = ychat.awareness.states[persona.awareness.client_id][
            "slash_commands"
        ]
        assert cmds == [{"name": "/compact", "description": None}]

    async def test_broadcast_preserves_directly_written_isWriting(self, ychat):
        # `isWriting` is written directly on the streaming hot path; a config
        # rebroadcast must merge fields, not clobber it.
        persona = _make_persona(ychat)
        persona.awareness.set_local_state_field("isWriting", "msg-1")
        persona.update_usage(Usage(input_tokens=1))
        state = ychat.awareness.states[persona.awareness.client_id]
        assert state["isWriting"] == "msg-1"
        assert state["usage"]["input_tokens"] == 1
