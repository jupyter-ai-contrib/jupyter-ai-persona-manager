"""
Fixture persona for E2E tests: one of a pair of near-identical echo-config
personas ("Switch A" / "Switch B") used to verify that switching the selected
persona does not carry the previous persona's model/settings selection over to
the new one (see tests/switch-personas.spec.ts, issue #62).

Like `echo-config`, it advertises a Model picker, a "Thinking" model setting, and
an "Effort" general setting, and echoes back what a selection *applied* (the
`update_*` hooks record it, so a non-default value is only echoed if the apply
path actually ran). The two personas are structurally identical apart from their
names and the model IDs they advertise, so a reply unambiguously identifies which
persona produced it and whether it inherited a selection it never should have.

Not part of the shipped package; see AGENTS.md.
"""

import os

from jupyter_ai_persona_manager import (
    BasePersona,
    ModelConfiguration,
    ModelOption,
    PersonaDefaults,
    SettingConfiguration,
    SettingOption,
)
from jupyterlab_chat.models import Message

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")

_MODELS = [
    ModelOption(id="a-one", name="A One"),
    ModelOption(id="a-two", name="A Two"),
]
_MODEL_CURRENT = "a-one"

_MODEL_SETTINGS = [
    SettingConfiguration(
        id="thinking",
        name="Thinking",
        current="medium",
        options=[
            SettingOption(id="low", name="Low"),
            SettingOption(id="medium", name="Medium"),
            SettingOption(id="high", name="High"),
        ],
    ),
]

_SETTINGS = [
    SettingConfiguration(
        id="effort",
        name="Effort",
        current="medium",
        options=[
            SettingOption(id="low", name="Low"),
            SettingOption(id="medium", name="Medium"),
            SettingOption(id="high", name="High"),
        ],
    ),
]


class SwitchAPersona(BasePersona):
    """Test-only persona (A) that echoes the model/settings a selection applied."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._applied_model: str | None = None
        self._applied_model_settings: dict[str, str | None] = {}
        self._applied_settings: dict[str, str | None] = {}
        self.report_model_configuration(
            ModelConfiguration(
                current=_MODEL_CURRENT,
                options=list(_MODELS),
                settings=[s.model_copy(deep=True) for s in _MODEL_SETTINGS],
            )
        )
        self.report_settings_configuration(
            [s.model_copy(deep=True) for s in _SETTINGS]
        )

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Switch A Persona",
            description="Test-only persona that echoes applied model/settings.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def update_model(self, model_id: str) -> None:
        self._applied_model = model_id

    async def update_model_settings(self, settings: dict[str, str | None]) -> None:
        self._applied_model_settings.update(settings)

    async def update_settings(self, settings: dict[str, str | None]) -> None:
        self._applied_settings.update(settings)

    async def process_message(self, message: Message) -> None:
        lines = [
            "persona: Switch A",
            f"applied model: {self._applied_model or '(default)'}",
        ]
        for setting in _MODEL_SETTINGS:
            value = self._applied_model_settings.get(setting.id) or "(default)"
            lines.append(f"applied {setting.id}: {value}")
        for setting in _SETTINGS:
            value = self._applied_settings.get(setting.id) or "(default)"
            lines.append(f"applied {setting.id}: {value}")
        self.send_message("\n".join(lines))
