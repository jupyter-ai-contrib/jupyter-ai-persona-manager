"""
Fixture persona for E2E tests: a `BasePersona` that advertises several models
with a non-first current/default, so a test can assert the model picker's option
ordering and that the "Default (…)" row names the current model.

The model list order is fixed (Alpha, Beta, Gamma) and the current model is Beta
— deliberately not the first option — so the test can distinguish "renders in
advertised order" from "renders current first".

Advertised statically in `__init__`; echoes the applied model on reply. Not part
of the shipped package; see AGENTS.md.
"""

import os

from jupyter_ai_persona_manager import (
    BasePersona,
    ModelConfiguration,
    ModelOption,
    PersonaDefaults,
)
from jupyterlab_chat.models import Message

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")

_MODELS = [
    ModelOption(id="alpha", name="Model Alpha"),
    ModelOption(id="beta", name="Model Beta"),
    ModelOption(id="gamma", name="Model Gamma"),
]
_CURRENT = "beta"


class ModelsPersona(BasePersona):
    """Test-only persona advertising several models with a non-first default."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._applied_model: str | None = None
        self.report_model_configuration(
            ModelConfiguration(current=_CURRENT, options=list(_MODELS))
        )

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Models Persona",
            description="Test-only persona advertising several models.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def update_model(self, model_id: str) -> None:
        self._applied_model = model_id

    async def process_message(self, message: Message) -> None:
        self.send_message(f"applied model: {self._applied_model or '(default)'}")
