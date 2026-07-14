"""Tests for the awareness-channel Pydantic models."""

from jupyter_ai_persona_manager.awareness_models import (
    ModelConfiguration,
    ModelOption,
    ModelSpec,
    PersonaOption,
    SettingConfiguration,
    SettingOption,
    Usage,
)


class TestPersonaOption:
    def test_round_trip(self):
        option = PersonaOption(
            id="jupyter-ai-personas::pkg::Bot",
            name="Bot",
            avatar_url="/avatars/bot",
            yjs_client_id=12345,
        )
        assert PersonaOption(**option.model_dump()) == option

    def test_serialized_shape(self):
        dumped = PersonaOption(id="p1", name="One", yjs_client_id=7).model_dump()
        assert dumped == {
            "id": "p1",
            "name": "One",
            "avatar_url": None,
            "yjs_client_id": 7,
        }


class TestModelConfiguration:
    def test_defaults(self):
        model = ModelConfiguration()
        assert model.current is None
        assert model.options == []
        assert model.settings == []

    def test_round_trip_full(self):
        model = ModelConfiguration(
            current="opus",
            options=[ModelOption(id="opus", name="Opus")],
            settings=[
                SettingConfiguration(
                    id="context_size",
                    current="200k",
                    options=[SettingOption(id="200k", name="200K")],
                )
            ],
        )
        assert ModelConfiguration(**model.model_dump()) == model


class TestSettingConfiguration:
    def test_round_trip_full(self):
        setting = SettingConfiguration(
            id="mode",
            current="ask",
            options=[SettingOption(id="ask"), SettingOption(id="code")],
        )
        assert SettingConfiguration(**setting.model_dump()) == setting


class TestUsage:
    def test_all_fields_default_none(self):
        usage = Usage()
        assert all(v is None for v in usage.model_dump().values())

    def test_exclude_none_only_keeps_set_fields(self):
        usage = Usage(input_tokens=3, context_size=1000)
        assert usage.model_dump(exclude_none=True) == {
            "input_tokens": 3,
            "context_size": 1000,
        }


class TestModelSpec:
    def test_defaults(self):
        spec = ModelSpec()
        assert spec.id is None
        assert spec.settings == {}

    def test_round_trip(self):
        spec = ModelSpec(id="opus", settings={"context_size": "200k", "x": None})
        restored = ModelSpec(**spec.model_dump())
        assert restored == spec

    def test_constructs_from_metadata_dict(self):
        # The shape stamped onto message metadata by the frontend.
        spec = ModelSpec(**{"id": None, "settings": {"agent_mode": "ask"}})
        assert spec.id is None
        assert spec.settings == {"agent_mode": "ask"}
