"""
Fixture persona for E2E tests: echoes back the metadata of the message it
receives, so a test can observe which metadata keys survived the frontend's
stamping.

This guards a specific regression. The persona-manager input-toolbar controls
stamp their own metadata (`to_persona`, model, settings) onto each outgoing
message. They must merge that onto whatever metadata other extensions have
already contributed to the shared chat input — never clear it. For example,
jupyterlab-commands-toolkit stamps a `web_client_id` onto the input so an AI
persona can route frontend commands back to the web client that triggered them.
This persona makes the metadata that actually reaches the message observable in
its reply, so a test can assert a third-party key is preserved.

It advertises a single general setting purely so a test has a control to change,
which forces the toolbar to re-stamp its metadata after a third-party key was
added — the exact point a clear-then-set would wipe that key.

Not part of the shipped package; see AGENTS.md and tests/test-helpers.ts for how
fixture personas are installed per suite. The persona class must be declared in
this module (the loader keeps only classes whose `__module__` is this file stem).
"""

import os

from jupyter_ai_persona_manager import (
    BasePersona,
    PersonaDefaults,
    SettingConfiguration,
    SettingOption,
)
from jupyterlab_chat.models import Message

_AVATAR_PATH = os.path.join(os.environ["JAI_TEST_ASSETS_DIR"], "persona.svg")

# The key a test stamps onto the input to stand in for a third-party extension's
# metadata (e.g. the commands-toolkit's web_client_id). Kept in sync with
# tests/metadata-preservation.spec.ts.
_THIRD_PARTY_KEY = "third_party_key"

# A single general setting, rendered as one toolbar control. Its value is
# unused; it exists only so a test can change a control to trigger a re-stamp.
_SETTINGS = [
    SettingConfiguration(
        id="mode",
        name="Mode",
        current="a",
        options=[
            SettingOption(id="a", name="A"),
            SettingOption(id="b", name="B"),
        ],
    ),
]


class MetadataEchoPersona(BasePersona):
    """Test-only persona that echoes the metadata of the message it receives."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.report_settings_configuration(
            [s.model_copy(deep=True) for s in _SETTINGS]
        )

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="Metadata Echo Persona",
            description="Test-only persona that echoes received message metadata.",
            avatar_path=_AVATAR_PATH,
            system_prompt="unused",
        )

    async def update_settings(self, settings: dict[str, str | None]) -> None:
        # The advertised setting exists only to give a test a control to change;
        # its applied value is irrelevant to what this persona echoes.
        pass

    async def process_message(self, message: Message) -> None:
        metadata = message.metadata or {}
        keys = ",".join(sorted(metadata))
        third_party = metadata.get(_THIRD_PARTY_KEY, "(absent)")
        self.send_message(
            f"metadata keys: {keys}\n{_THIRD_PARTY_KEY}: {third_party}"
        )
