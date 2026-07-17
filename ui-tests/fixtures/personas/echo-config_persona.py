"""
Fixture persona for E2E tests: a `BasePersona` that advertises a model list,
model settings, and general settings, and echoes back what a user's selection
applied — so a control change in the toolbar shows up, provably, in the reply.

How the round trip works (see BasePersona):

- `__init__` statically advertises the configuration via
  `report_model_configuration` (the model picker + model settings) and
  `report_settings_configuration` (general settings).
- The user picks a value in a toolbar control; the frontend stamps it onto the
  outgoing message's metadata.
- Before `process_message` runs, the PersonaManager calls
  `apply_specs_in_message`, which routes the selection through
  `update_model` / `update_model_settings` / `update_settings`. This persona
  *records* what each hook received, so a non-default selection is only echoed
  if the corresponding hook actually fired — proving the apply path, not just
  the awareness broadcast.
- `process_message` echoes those recorded values (or "(default)" when the user
  left a control on its default and the hook never fired).

Not part of the shipped package; see AGENTS.md and tests/test-helpers.ts for how
fixture personas are installed per suite. The persona class must be declared in
this module (the loader keeps only classes whose `__module__` is this file stem).
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

# The advertised model options; the first is the current/default. Fixed so the
# UI text is deterministic and directly assertable.
_MODELS = [
    ModelOption(id="claude-haiku", name="Claude Haiku"),
    ModelOption(id="claude-opus", name="Claude Opus"),
    ModelOption(id="claude-fable", name="Claude Fable"),
]
_MODEL_CURRENT = "claude-haiku"

# A model setting, rendered next to the model picker (ACP calls these
# `model_config`). Its list order controls the dropdown order.
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

# General settings, rendered as separate controls in this list order: a
# multi-value select and a boolean-style two-option select.
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
    SettingConfiguration(
        id="verbose",
        name="Verbose",
        current="off",
        options=[
            SettingOption(id="on", name="On"),
            SettingOption(id="off", name="Off"),
        ],
    ),
]


class EchoConfigPersona(BasePersona):
    """Test-only persona that echoes the model/settings a selection applied."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # What each update_* hook recorded. None / empty until a non-default
        # selection routes through the hook.
        self._applied_model: str | None = None
        self._applied_model_settings: dict[str, str | None] = {}
        self._applied_settings: dict[str, str | None] = {}
        # Advertise the configuration statically.
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
            name="Echo Config Persona",
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
        # apply_specs_in_message() has already run (the PersonaManager applies a
        # message's model & settings metadata before process_message), so the
        # update_* hooks above have recorded any non-default selection.
        lines = [
            f"applied model: {self._applied_model or '(default)'}",
        ]
        for setting in _MODEL_SETTINGS:
            value = self._applied_model_settings.get(setting.id) or "(default)"
            lines.append(f"applied {setting.id}: {value}")
        for setting in _SETTINGS:
            value = self._applied_settings.get(setting.id) or "(default)"
            lines.append(f"applied {setting.id}: {value}")
        self.send_message("\n".join(lines))
