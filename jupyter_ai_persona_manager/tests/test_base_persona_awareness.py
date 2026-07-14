"""
Tests for the awareness-channel API on BasePersona: the `get_*` readers, the
`report_*` methods that publish, apply_model_spec / apply_settings_spec, and
apply_specs_in_message.

The persona's session state lives in its `PersonaAwareness` slot, so these tests
back the persona with a real `PersonaAwareness` over an in-memory `YChat` — no
mocking of the awareness layer, so reads and writes go through the same
client-ID-scoped code paths as production.
"""

from unittest.mock import MagicMock

from jupyterlab_chat.models import Message, User
from jupyterlab_chat.ychat import YChat
from pycrdt import Awareness

from jupyter_ai_persona_manager.awareness_models import (
    CommandOption,
    ModelConfiguration,
    ModelOption,
    ModelSpec,
    SettingConfiguration,
    Usage,
)
from jupyter_ai_persona_manager.base_persona import BasePersona, PersonaDefaults
from jupyter_ai_persona_manager.persona_awareness import PersonaAwareness


def _awareness(id: str = "p1") -> PersonaAwareness:
    """A real PersonaAwareness over a fresh in-memory YChat. Constructed outside
    an event loop, so the heartbeat is skipped — everything else is real."""
    ychat = YChat()
    ychat.awareness = Awareness(ydoc=ychat._ydoc)
    user = User(username=id, name=id, display_name=id)
    return PersonaAwareness(ychat=ychat, log=MagicMock(), user=user, id=id)


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


def _make_persona(
    *,
    model: ModelConfiguration | None = None,
    settings: list[SettingConfiguration] | None = None,
    usage: Usage | None = None,
    slash_commands: list[CommandOption] | None = None,
) -> _ConcretePersona:
    """Create a persona bypassing __init__, wired to a dict-backed awareness slot
    seeded with the given session state."""
    persona = _ConcretePersona.__new__(_ConcretePersona)
    persona.log = MagicMock()
    persona.awareness = _awareness()
    persona.awareness.model = model or ModelConfiguration()
    persona.awareness.settings = settings or []
    persona.awareness.usage = usage or Usage()
    persona.awareness.slash_commands = slash_commands or []
    persona.calls = []
    return persona


# ---------------------------------------------------------------------------
# Getters
# ---------------------------------------------------------------------------

class TestGetters:
    def test_get_model_returns_current(self):
        persona = _make_persona(model=ModelConfiguration(current="opus"))
        assert persona.get_model() == "opus"

    def test_get_model_none_by_default(self):
        assert _make_persona().get_model() is None

    def test_get_model_settings_maps_id_to_current(self):
        persona = _make_persona(
            model=ModelConfiguration(
                settings=[
                    SettingConfiguration(id="context_size", current="200k"),
                    SettingConfiguration(id="temp", current=None),
                ]
            )
        )
        assert persona.get_model_settings() == {"context_size": "200k", "temp": None}

    def test_get_settings_maps_id_to_current(self):
        persona = _make_persona(
            settings=[SettingConfiguration(id="mode", current="ask")]
        )
        assert persona.get_settings() == {"mode": "ask"}

    def test_get_model_configuration_and_setting_configurations(self):
        model = ModelConfiguration(current="opus")
        settings = [SettingConfiguration(id="mode")]
        persona = _make_persona(model=model, settings=settings)
        assert persona.get_model_configuration() == model
        assert persona.get_setting_configurations() == settings

    def test_get_usage_and_slash_commands(self):
        usage = Usage(input_tokens=1)
        cmds = [CommandOption(name="/x")]
        persona = _make_persona(usage=usage, slash_commands=cmds)
        assert persona.get_usage() == usage
        assert persona.get_slash_commands() == cmds


# ---------------------------------------------------------------------------
# Reporters publish over awareness
# ---------------------------------------------------------------------------

class TestReporters:
    def test_report_model_configuration_publishes(self):
        persona = _make_persona()
        model = ModelConfiguration(current="opus", options=[ModelOption(id="opus")])
        persona.report_model_configuration(model)
        assert persona.get_model_configuration() == model
        # It is published as a top-level `model` entry of the slot.
        assert persona.awareness.get_local_state()["model"]["current"] == "opus"

    def test_report_settings_configuration_publishes(self):
        persona = _make_persona()
        settings = [SettingConfiguration(id="mode", current="ask")]
        persona.report_settings_configuration(settings)
        assert persona.get_setting_configurations() == settings
        assert persona.awareness.get_local_state()["settings"][0]["id"] == "mode"

    def test_slot_holds_flattened_fields_and_isWriting_stays_false(self):
        # Each field is a top-level entry of the slot; isWriting is its own key,
        # owned by the streaming path, and defaults to False.
        persona = _make_persona()
        persona.report_model_configuration(ModelConfiguration())
        slot = persona.awareness.get_local_state()
        assert {"id", "model", "settings", "usage", "slash_commands"} <= set(slot)
        assert slot["isWriting"] is False

    def test_report_slash_commands_replaces_and_publishes(self):
        persona = _make_persona()
        persona.report_slash_commands([CommandOption(name="/compact")])
        assert persona.get_slash_commands() == [CommandOption(name="/compact")]
        assert persona.awareness.get_local_state()["slash_commands"] == [
            {"name": "/compact", "description": None}
        ]


# ---------------------------------------------------------------------------
# report_usage merge/append semantics
# ---------------------------------------------------------------------------

class TestReportUsage:
    def test_replace_sets_provided_fields(self):
        persona = _make_persona()
        persona.report_usage(Usage(input_tokens=10, output_tokens=5))
        usage = persona.get_usage()
        assert usage.input_tokens == 10
        assert usage.output_tokens == 5

    def test_replace_only_touches_provided_fields(self):
        persona = _make_persona(usage=Usage(input_tokens=10, output_tokens=99))
        # Only input_tokens provided; output_tokens untouched.
        persona.report_usage(Usage(input_tokens=20))
        assert persona.get_usage().input_tokens == 20
        assert persona.get_usage().output_tokens == 99

    def test_replace_composes_context_and_tokens_across_calls(self):
        persona = _make_persona()
        persona.report_usage(Usage(context_tokens=100, context_size=1000))
        persona.report_usage(Usage(input_tokens=5, output_tokens=7))
        usage = persona.get_usage()
        assert usage.context_tokens == 100
        assert usage.context_size == 1000
        assert usage.input_tokens == 5
        assert usage.output_tokens == 7

    def test_append_adds_to_existing(self):
        persona = _make_persona(usage=Usage(input_tokens=10))
        persona.report_usage(Usage(input_tokens=5, output_tokens=3), append=True)
        assert persona.get_usage().input_tokens == 15
        assert persona.get_usage().output_tokens == 3

    def test_append_from_none_treats_existing_as_zero(self):
        persona = _make_persona()
        persona.report_usage(Usage(total_tokens=7), append=True)
        assert persona.get_usage().total_tokens == 7

    def test_report_usage_publishes(self):
        persona = _make_persona()
        persona.report_usage(Usage(input_tokens=1))
        assert persona.awareness.get_local_state()["usage"]["input_tokens"] == 1


# ---------------------------------------------------------------------------
# apply_model_spec
# ---------------------------------------------------------------------------

class TestApplyModelSpec:
    async def test_none_model_id_is_skipped(self):
        persona = _make_persona(model=ModelConfiguration(current="opus"))
        await persona.apply_model_spec(ModelSpec(id=None))
        assert persona.calls == []

    async def test_model_equal_to_current_is_skipped(self):
        persona = _make_persona(model=ModelConfiguration(current="opus"))
        await persona.apply_model_spec(ModelSpec(id="opus"))
        assert persona.calls == []

    async def test_model_change_calls_update_model(self):
        persona = _make_persona(model=ModelConfiguration(current="opus"))
        await persona.apply_model_spec(ModelSpec(id="fable"))
        assert persona.calls == [("update_model", "fable")]

    async def test_model_settings_change_calls_update_model_settings_once(self):
        persona = _make_persona(
            model=ModelConfiguration(
                settings=[
                    SettingConfiguration(id="a", current="1"),
                    SettingConfiguration(id="b", current="2"),
                ]
            )
        )
        # Two changed settings, but update_model_settings is called only once
        # with the changed values.
        await persona.apply_model_spec(ModelSpec(settings={"a": "9", "b": "8"}))
        assert persona.calls == [("update_model_settings", {"a": "9", "b": "8"})]

    async def test_passes_only_changed_model_settings(self):
        # Unchanged keys are filtered out before calling update_model_settings.
        persona = _make_persona(
            model=ModelConfiguration(
                settings=[
                    SettingConfiguration(id="a", current="1"),
                    SettingConfiguration(id="b", current="2"),
                ]
            )
        )
        await persona.apply_model_spec(ModelSpec(settings={"a": "9", "b": "2"}))
        assert persona.calls == [("update_model_settings", {"a": "9"})]

    async def test_records_new_current_values_and_publishes(self):
        persona = _make_persona(
            model=ModelConfiguration(
                current="opus",
                settings=[SettingConfiguration(id="a", current="1")],
            )
        )
        await persona.apply_model_spec(ModelSpec(id="fable", settings={"a": "2"}))
        # The awareness slot now reflects the applied selection.
        assert persona.get_model() == "fable"
        assert persona.get_model_settings() == {"a": "2"}
        assert persona.awareness.get_local_state()["model"]["current"] == "fable"

    async def test_model_settings_none_value_skipped(self):
        persona = _make_persona(
            model=ModelConfiguration(
                settings=[SettingConfiguration(id="a", current="1")]
            )
        )
        await persona.apply_model_spec(ModelSpec(settings={"a": None}))
        assert persona.calls == []

    async def test_model_settings_unchanged_skipped(self):
        persona = _make_persona(
            model=ModelConfiguration(
                settings=[SettingConfiguration(id="a", current="1")]
            )
        )
        await persona.apply_model_spec(ModelSpec(settings={"a": "1"}))
        assert persona.calls == []

    async def test_model_and_settings_applied_in_order(self):
        persona = _make_persona(
            model=ModelConfiguration(
                current="opus",
                settings=[SettingConfiguration(id="a", current="1")],
            )
        )
        await persona.apply_model_spec(ModelSpec(id="fable", settings={"a": "2"}))
        assert persona.calls == [
            ("update_model", "fable"),
            ("update_model_settings", {"a": "2"}),
        ]


class _NonConfigurablePersona(BasePersona):
    """A persona that never overrides update_model/settings — the defaults are
    no-ops, so its model and settings aren't configurable."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Fixed", description="d", avatar_path="", system_prompt=""
        )

    async def process_message(self, message):
        pass


def _make_nonconfigurable(model: ModelConfiguration) -> _NonConfigurablePersona:
    persona = _NonConfigurablePersona.__new__(_NonConfigurablePersona)
    persona.log = MagicMock()
    persona.awareness = _awareness()
    persona.awareness.model = model
    persona.awareness.settings = []
    return persona


class TestNonConfigurablePersona:
    """A persona that doesn't override update_* still instantiates and applies
    specs — the default update_* methods are no-ops (nothing to configure)."""

    def test_instantiates_without_update_overrides(self):
        # Would raise if update_* were abstract.
        persona = _make_nonconfigurable(ModelConfiguration(current="opus"))
        assert persona.get_model() == "opus"

    async def test_apply_model_spec_updates_awareness_without_override(self):
        persona = _make_nonconfigurable(ModelConfiguration(current="opus"))
        # No update_model override, but the applier still records the new current
        # value and publishes it.
        await persona.apply_model_spec(ModelSpec(id="fable"))
        assert persona.get_model() == "fable"


# ---------------------------------------------------------------------------
# apply_settings_spec
# ---------------------------------------------------------------------------

class TestApplySettingsSpec:
    async def test_none_value_skipped(self):
        persona = _make_persona(
            settings=[SettingConfiguration(id="mode", current="ask")]
        )
        await persona.apply_settings_spec({"mode": None})
        assert persona.calls == []

    async def test_unchanged_skipped(self):
        persona = _make_persona(
            settings=[SettingConfiguration(id="mode", current="ask")]
        )
        await persona.apply_settings_spec({"mode": "ask"})
        assert persona.calls == []

    async def test_change_calls_update_settings_once_with_changed_only(self):
        persona = _make_persona(
            settings=[
                SettingConfiguration(id="mode", current="ask"),
                SettingConfiguration(id="effort", current="low"),
            ]
        )
        # `effort` is unchanged, so only `mode` reaches update_settings.
        await persona.apply_settings_spec({"mode": "code", "effort": "low"})
        assert persona.calls == [("update_settings", {"mode": "code"})]

    async def test_records_new_current_value_and_publishes(self):
        persona = _make_persona(
            settings=[SettingConfiguration(id="mode", current="ask")]
        )
        await persona.apply_settings_spec({"mode": "code"})
        assert persona.get_settings() == {"mode": "code"}


# ---------------------------------------------------------------------------
# apply_specs_in_message
# ---------------------------------------------------------------------------

def _message(metadata):
    return Message(id="m1", body="hi", sender="user-1", time=0.0, metadata=metadata)


class TestApplySpecsInMessage:
    async def test_no_metadata_is_noop(self):
        persona = _make_persona()
        await persona.apply_specs_in_message(_message(None))
        assert persona.calls == []

    async def test_applies_model_and_settings(self):
        persona = _make_persona(
            model=ModelConfiguration(current="opus"),
            settings=[SettingConfiguration(id="mode", current="ask")],
        )
        msg = _message(
            {
                "model": {"id": "fable", "settings": {}},
                "settings": {"mode": "code"},
            }
        )
        await persona.apply_specs_in_message(msg)
        assert persona.calls == [
            ("update_model", "fable"),
            ("update_settings", {"mode": "code"}),
        ]

    async def test_accepts_model_spec_instance(self):
        persona = _make_persona(model=ModelConfiguration(current="opus"))
        msg = _message({"model": ModelSpec(id="fable")})
        await persona.apply_specs_in_message(msg)
        assert persona.calls == [("update_model", "fable")]

    async def test_ignores_unrelated_metadata_keys(self):
        persona = _make_persona()
        msg = _message({"to_persona": "p1"})
        await persona.apply_specs_in_message(msg)
        assert persona.calls == []
