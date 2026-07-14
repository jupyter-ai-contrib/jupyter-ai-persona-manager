"""
Tests for the awareness-channel API on BasePersona: the getters, the setters
that broadcast, apply_model_spec / apply_settings_spec, apply_message_metadata,
and update_usage / update_slash_commands.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from jupyterlab_chat.models import Message

from jupyter_ai_persona_manager.awareness_models import (
    CommandOption,
    ModelConfiguration,
    ModelOption,
    ModelSpec,
    PersonaAwarenessState,
    SettingConfiguration,
    SettingOption,
    Usage,
)
from jupyter_ai_persona_manager.base_persona import BasePersona, PersonaDefaults


class _ConcretePersona(BasePersona):
    """Concrete persona that records update_* calls for assertions."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="TestPersona",
            description="A test persona",
            avatar_path="",
            system_prompt="",
        )

    async def process_message(self, message):
        pass

    async def update_model(self, model_id: str) -> None:
        self.calls.append(("update_model", model_id))

    async def update_model_settings(self, settings) -> None:
        self.calls.append(("update_model_settings", settings))

    async def update_settings(self, settings) -> None:
        self.calls.append(("update_settings", settings))


def _make_persona(state: PersonaAwarenessState | None = None) -> _ConcretePersona:
    """Create a persona bypassing __init__; wire a real awareness state and a
    mock awareness object so broadcasts are observable but harmless."""
    persona = _ConcretePersona.__new__(_ConcretePersona)
    persona.log = MagicMock()
    persona.awareness = MagicMock()
    persona._awareness_state = state or PersonaAwarenessState(id="p1")
    persona.calls = []
    return persona


# ---------------------------------------------------------------------------
# Getters
# ---------------------------------------------------------------------------

class TestGetters:
    def test_get_model_returns_current(self):
        persona = _make_persona(
            PersonaAwarenessState(id="p1", model=ModelConfiguration(current="opus"))
        )
        assert persona.get_model() == "opus"

    def test_get_model_none_by_default(self):
        assert _make_persona().get_model() is None

    def test_get_model_settings_maps_id_to_current(self):
        persona = _make_persona(
            PersonaAwarenessState(
                id="p1",
                model=ModelConfiguration(
                    settings=[
                        SettingConfiguration(id="context_size", current="200k"),
                        SettingConfiguration(id="temp", current=None),
                    ]
                ),
            )
        )
        assert persona.get_model_settings() == {"context_size": "200k", "temp": None}

    def test_get_settings_maps_id_to_current(self):
        persona = _make_persona(
            PersonaAwarenessState(
                id="p1",
                settings=[
                    SettingConfiguration(id="mode", current="ask"),
                ],
            )
        )
        assert persona.get_settings() == {"mode": "ask"}

    def test_get_model_configuration_and_setting_configurations(self):
        model = ModelConfiguration(current="opus")
        settings = [SettingConfiguration(id="mode")]
        persona = _make_persona(
            PersonaAwarenessState(id="p1", model=model, settings=settings)
        )
        assert persona.get_model_configuration() == model
        assert persona.get_setting_configurations() == settings

    def test_get_usage_and_slash_commands(self):
        usage = Usage(input_tokens=1)
        cmds = [CommandOption(name="/x")]
        persona = _make_persona(
            PersonaAwarenessState(id="p1", usage=usage, slash_commands=cmds)
        )
        assert persona.get_usage() == usage
        assert persona.get_slash_commands() == cmds


# ---------------------------------------------------------------------------
# Setters broadcast
# ---------------------------------------------------------------------------

class TestSettersBroadcast:
    def test_set_model_configuration_broadcasts(self):
        persona = _make_persona()
        model = ModelConfiguration(current="opus", options=[ModelOption(id="opus")])
        persona.set_model_configuration(model)
        assert persona._awareness_state.model is model
        persona.awareness.set_local_state_field.assert_called_once()
        field, payload = persona.awareness.set_local_state_field.call_args[0]
        assert field == "persona"
        assert payload["model"]["current"] == "opus"

    def test_set_setting_configurations_broadcasts(self):
        persona = _make_persona()
        settings = [SettingConfiguration(id="mode", current="ask")]
        persona.set_setting_configurations(settings)
        assert persona._awareness_state.settings is settings
        payload = persona.awareness.set_local_state_field.call_args[0][1]
        assert payload["settings"][0]["id"] == "mode"

    def test_update_slash_commands_replaces_and_broadcasts(self):
        persona = _make_persona()
        persona.update_slash_commands([CommandOption(name="/compact")])
        assert persona.get_slash_commands() == [CommandOption(name="/compact")]
        payload = persona.awareness.set_local_state_field.call_args[0][1]
        assert payload["slash_commands"] == [
            {"name": "/compact", "description": None}
        ]


# ---------------------------------------------------------------------------
# update_usage merge/append semantics
# ---------------------------------------------------------------------------

class TestUpdateUsage:
    def test_replace_sets_provided_fields(self):
        persona = _make_persona()
        persona.update_usage(Usage(input_tokens=10, output_tokens=5))
        usage = persona.get_usage()
        assert usage.input_tokens == 10
        assert usage.output_tokens == 5

    def test_replace_only_touches_provided_fields(self):
        persona = _make_persona(
            PersonaAwarenessState(
                id="p1", usage=Usage(input_tokens=10, output_tokens=99)
            )
        )
        # Only input_tokens provided; output_tokens untouched.
        persona.update_usage(Usage(input_tokens=20))
        assert persona.get_usage().input_tokens == 20
        assert persona.get_usage().output_tokens == 99

    def test_replace_composes_context_and_tokens_across_calls(self):
        persona = _make_persona()
        persona.update_usage(Usage(context_tokens=100, context_size=1000))
        persona.update_usage(Usage(input_tokens=5, output_tokens=7))
        usage = persona.get_usage()
        assert usage.context_tokens == 100
        assert usage.context_size == 1000
        assert usage.input_tokens == 5
        assert usage.output_tokens == 7

    def test_append_adds_to_existing(self):
        persona = _make_persona(
            PersonaAwarenessState(id="p1", usage=Usage(input_tokens=10))
        )
        persona.update_usage(Usage(input_tokens=5, output_tokens=3), append=True)
        assert persona.get_usage().input_tokens == 15
        assert persona.get_usage().output_tokens == 3

    def test_append_from_none_treats_existing_as_zero(self):
        persona = _make_persona()
        persona.update_usage(Usage(total_tokens=7), append=True)
        assert persona.get_usage().total_tokens == 7

    def test_update_usage_broadcasts(self):
        persona = _make_persona()
        persona.update_usage(Usage(input_tokens=1))
        persona.awareness.set_local_state_field.assert_called_once()


# ---------------------------------------------------------------------------
# apply_model_spec
# ---------------------------------------------------------------------------

class TestApplyModelSpec:
    async def test_none_model_id_is_skipped(self):
        persona = _make_persona(
            PersonaAwarenessState(id="p1", model=ModelConfiguration(current="opus"))
        )
        await persona.apply_model_spec(ModelSpec(id=None))
        assert persona.calls == []

    async def test_model_equal_to_current_is_skipped(self):
        persona = _make_persona(
            PersonaAwarenessState(id="p1", model=ModelConfiguration(current="opus"))
        )
        await persona.apply_model_spec(ModelSpec(id="opus"))
        assert persona.calls == []

    async def test_model_change_calls_update_model(self):
        persona = _make_persona(
            PersonaAwarenessState(id="p1", model=ModelConfiguration(current="opus"))
        )
        await persona.apply_model_spec(ModelSpec(id="fable"))
        assert persona.calls == [("update_model", "fable")]

    async def test_model_settings_change_calls_update_model_settings_once(self):
        persona = _make_persona(
            PersonaAwarenessState(
                id="p1",
                model=ModelConfiguration(
                    settings=[
                        SettingConfiguration(id="a", current="1"),
                        SettingConfiguration(id="b", current="2"),
                    ]
                ),
            )
        )
        # Two changed settings, but update_model_settings is called only once
        # with the whole dict.
        spec = ModelSpec(settings={"a": "9", "b": "8"})
        await persona.apply_model_spec(spec)
        assert persona.calls == [("update_model_settings", {"a": "9", "b": "8"})]

    async def test_model_settings_none_value_skipped(self):
        persona = _make_persona(
            PersonaAwarenessState(
                id="p1",
                model=ModelConfiguration(
                    settings=[SettingConfiguration(id="a", current="1")]
                ),
            )
        )
        await persona.apply_model_spec(ModelSpec(settings={"a": None}))
        assert persona.calls == []

    async def test_model_settings_unchanged_skipped(self):
        persona = _make_persona(
            PersonaAwarenessState(
                id="p1",
                model=ModelConfiguration(
                    settings=[SettingConfiguration(id="a", current="1")]
                ),
            )
        )
        await persona.apply_model_spec(ModelSpec(settings={"a": "1"}))
        assert persona.calls == []

    async def test_model_and_settings_applied_in_order(self):
        persona = _make_persona(
            PersonaAwarenessState(
                id="p1",
                model=ModelConfiguration(
                    current="opus",
                    settings=[SettingConfiguration(id="a", current="1")],
                ),
            )
        )
        await persona.apply_model_spec(ModelSpec(id="fable", settings={"a": "2"}))
        assert persona.calls == [
            ("update_model", "fable"),
            ("update_model_settings", {"a": "2"}),
        ]


# ---------------------------------------------------------------------------
# apply_settings_spec
# ---------------------------------------------------------------------------

class TestApplySettingsSpec:
    async def test_none_value_skipped(self):
        persona = _make_persona(
            PersonaAwarenessState(
                id="p1", settings=[SettingConfiguration(id="mode", current="ask")]
            )
        )
        await persona.apply_settings_spec({"mode": None})
        assert persona.calls == []

    async def test_unchanged_skipped(self):
        persona = _make_persona(
            PersonaAwarenessState(
                id="p1", settings=[SettingConfiguration(id="mode", current="ask")]
            )
        )
        await persona.apply_settings_spec({"mode": "ask"})
        assert persona.calls == []

    async def test_change_calls_update_settings_once(self):
        persona = _make_persona(
            PersonaAwarenessState(
                id="p1",
                settings=[
                    SettingConfiguration(id="mode", current="ask"),
                    SettingConfiguration(id="effort", current="low"),
                ],
            )
        )
        spec = {"mode": "code", "effort": "high"}
        await persona.apply_settings_spec(spec)
        assert persona.calls == [("update_settings", spec)]


# ---------------------------------------------------------------------------
# apply_message_metadata
# ---------------------------------------------------------------------------

def _message(metadata):
    return Message(id="m1", body="hi", sender="user-1", time=0.0, metadata=metadata)


class TestApplyMessageMetadata:
    async def test_no_metadata_is_noop(self):
        persona = _make_persona()
        await persona.apply_message_metadata(_message(None))
        assert persona.calls == []

    async def test_applies_model_and_settings(self):
        persona = _make_persona(
            PersonaAwarenessState(
                id="p1",
                model=ModelConfiguration(current="opus"),
                settings=[SettingConfiguration(id="mode", current="ask")],
            )
        )
        msg = _message(
            {
                "model": {"id": "fable", "settings": {}},
                "settings": {"mode": "code"},
            }
        )
        await persona.apply_message_metadata(msg)
        assert persona.calls == [
            ("update_model", "fable"),
            ("update_settings", {"mode": "code"}),
        ]

    async def test_accepts_model_spec_instance(self):
        persona = _make_persona(
            PersonaAwarenessState(id="p1", model=ModelConfiguration(current="opus"))
        )
        msg = _message({"model": ModelSpec(id="fable")})
        await persona.apply_message_metadata(msg)
        assert persona.calls == [("update_model", "fable")]

    async def test_ignores_unrelated_metadata_keys(self):
        persona = _make_persona()
        msg = _message({"to_persona": "p1"})
        await persona.apply_message_metadata(msg)
        assert persona.calls == []
