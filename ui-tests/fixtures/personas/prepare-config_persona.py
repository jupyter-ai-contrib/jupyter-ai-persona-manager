"""
Fixture persona for E2E tests: a `BasePersona` that only learns its model &
settings during its one-time `prepare()` hook — mirroring an ACP persona that
must spawn its agent subprocess before it can advertise a model list.

Unlike `echo-config_persona.py`, which reports its configuration statically in
`__init__` (so its controls appear the instant it is selected), this persona
reports nothing until `prepare()` runs. It therefore proves issue #141: the
model & settings controls appear as soon as the persona is *selected* only if
the server eagerly runs `prepare()` on selection (via
`PersonaManager.prepare_persona`), rather than waiting for the first message.

`prepare()` sleeps briefly so a test can observe that the controls are absent
right after selection and then appear once preparation completes — without the
test ever sending a message.

Not part of the shipped package; see AGENTS.md and tests/test-helpers.ts for how
fixture personas are installed per suite. The persona class must be declared in
this module (the loader keeps only classes whose `__module__` is this file stem).
"""

import asyncio
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

# Advertised only after prepare(); the first model is the current/default.
_MODELS = [
    ModelOption(id="prepared-one", name="Prepared One"),
    ModelOption(id="prepared-two", name="Prepared Two"),
]
_MODEL_CURRENT = "prepared-one"

# A model setting, rendered next to the model picker once prepare() reports it.
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


class PrepareConfigPersona(BasePersona):
    """Test-only persona that advertises its config only from prepare()."""

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Prepare Config Persona",
            description="Test-only persona that reports config from prepare().",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def prepare(self) -> None:
        # A brief delay simulates the cost of an ACP persona spawning its agent
        # and querying it, and gives a test a window to observe the controls
        # being absent immediately after selection.
        await asyncio.sleep(0.5)
        self.report_model_configuration(
            ModelConfiguration(
                current=_MODEL_CURRENT,
                options=list(_MODELS),
                settings=[s.model_copy(deep=True) for s in _MODEL_SETTINGS],
            )
        )

    async def process_message(self, message: Message) -> None:
        self.send_message("prepared and ready")
