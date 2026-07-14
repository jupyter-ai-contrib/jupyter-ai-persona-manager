"""Tests for the awareness-channel Pydantic models."""

from jupyter_ai_persona_manager.awareness_models import (
    CommandOption,
    ModelConfiguration,
    ModelOption,
    ModelSpec,
    PersonaAwarenessState,
    PersonaManagerAwarenessState,
    PersonaOption,
    SettingConfiguration,
    SettingOption,
    Usage,
)


class TestPersonaManagerAwarenessState:
    def test_defaults_to_empty_persona_list(self):
        state = PersonaManagerAwarenessState()
        assert state.personas == []

    def test_round_trip(self):
        state = PersonaManagerAwarenessState(
            personas=[
                PersonaOption(
                    id="jupyter-ai-personas::pkg::Bot",
                    name="Bot",
                    avatar_url="/avatars/bot",
                    yjs_client_id=12345,
                )
            ]
        )
        restored = PersonaManagerAwarenessState(**state.model_dump())
        assert restored == state

    def test_serialized_shape(self):
        state = PersonaManagerAwarenessState(
            personas=[
                PersonaOption(id="p1", name="One", yjs_client_id=7),
            ]
        )
        dumped = state.model_dump()
        assert dumped == {
            "personas": [
                {
                    "id": "p1",
                    "name": "One",
                    "avatar_url": None,
                    "yjs_client_id": 7,
                }
            ]
        }


class TestPersonaAwarenessState:
    def test_defaults(self):
        state = PersonaAwarenessState(id="p1")
        assert state.model == ModelConfiguration()
        assert state.settings == []
        assert state.usage == Usage()
        assert state.slash_commands == []

    def test_round_trip_full(self):
        state = PersonaAwarenessState(
            id="p1",
            model=ModelConfiguration(
                current="opus",
                options=[ModelOption(id="opus", name="Opus")],
                settings=[
                    SettingConfiguration(
                        id="context_size",
                        current="200k",
                        options=[SettingOption(id="200k", name="200K")],
                    )
                ],
            ),
            settings=[
                SettingConfiguration(
                    id="mode",
                    current="ask",
                    options=[
                        SettingOption(id="ask"),
                        SettingOption(id="code"),
                    ],
                )
            ],
            usage=Usage(input_tokens=10, output_tokens=5),
            slash_commands=[CommandOption(name="/compact", description="Compact")],
        )
        restored = PersonaAwarenessState(**state.model_dump())
        assert restored == state

    def test_serialized_shape_defaults(self):
        dumped = PersonaAwarenessState(id="p1").model_dump()
        assert dumped["id"] == "p1"
        assert dumped["model"] == {"current": None, "options": [], "settings": []}
        assert dumped["settings"] == []
        assert dumped["slash_commands"] == []
        # Usage fields all default to None.
        assert set(dumped["usage"].values()) == {None}


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
