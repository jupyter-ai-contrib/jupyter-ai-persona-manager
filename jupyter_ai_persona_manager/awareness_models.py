"""
Pydantic models for the persona-manager awareness channel.

Session information (the persona list, and each persona's model, model
settings, general settings, usage, and slash commands) is broadcast over the
chat's Yjs awareness channel rather than fetched through REST polling. The
awareness map is keyed by Yjs client IDs, with values being arbitrary dicts.
The two slots this project owns are managed by the awareness helpers in
`persona_awareness.py`:

- `PersonaManagerAwareness` (fixed client ID) publishes the list of personas
  (`PersonaOption`s) in the chat.
- `PersonaAwareness` (one per persona) publishes that persona's model
  configuration, general settings, usage, and slash commands.

Those helpers own the aggregate shape via typed properties, so this module only
defines the component models (the property types and the serialized shapes).

User selections are *not* stored here. They ride on outgoing message metadata
(see `ModelSpec`), so each user's choices are private and applied per message.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PersonaOption(BaseModel):
    """One persona in the chat, as advertised in the persona selector."""

    id: str
    name: str
    avatar_url: str | None = None
    # The Yjs client ID of this persona's awareness client. Persona client IDs
    # are dynamic (they change as personas load/reload), so the manager reports
    # each one here to let the browser look up a persona's awareness slot in O(1).
    yjs_client_id: int


class ModelOption(BaseModel):
    """One selectable model."""

    id: str
    name: str | None = None
    description: str | None = None


class SettingOption(BaseModel):
    """One selectable value for a setting."""

    id: str
    name: str | None = None
    description: str | None = None


class SettingConfiguration(BaseModel):
    """A single setting: its current value and all available options.

    Used both for model settings (rendered next to the model picker) and for
    general settings (rendered separately). The list order controls UI order.
    """

    # ID of the setting, e.g. "agent_mode".
    id: str
    # The current value, or None to indicate the persona's default.
    current: str | None = None
    name: str | None = None
    description: str | None = None
    options: list[SettingOption] = Field(default_factory=list)


class ModelConfiguration(BaseModel):
    """The persona's current model, its options, and its model settings."""

    # The current model ID, or None to indicate the persona's default.
    current: str | None = None
    options: list[ModelOption] = Field(default_factory=list)
    # Settings that should render near the model picker (ACP `model_config`).
    settings: list[SettingConfiguration] = Field(default_factory=list)


class Usage(BaseModel):
    """Token and cost usage reported by a persona for the current session."""

    # Live context-window snapshot. Unlike the counters below, these can
    # decrease during a session (e.g. after the agent compacts context).
    context_tokens: int | None = None  # tokens currently in the window
    context_size: int | None = None  # total window size

    # Cumulative token counts for the session.
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_read_tokens: int | None = None
    cached_write_tokens: int | None = None
    thought_tokens: int | None = None
    total_tokens: int | None = None

    # Cumulative session cost.
    cost_amount: float | None = None
    cost_currency: str | None = None  # ISO 4217, e.g. "USD"


class CommandOption(BaseModel):
    """One slash command advertised by a persona."""

    name: str  # includes leading "/", e.g. "/compact"
    description: str | None = None


################################################
# Message metadata (user selections)
################################################


class ModelSpec(BaseModel):
    """A user's model selection, carried on outgoing message metadata.

    `id` is the selected model ID, or None to keep the persona's current model.
    `settings` maps model-setting IDs to a selected option ID, or None to keep
    the current value for that setting.
    """

    id: str | None = None
    settings: dict[str, str | None] = Field(default_factory=dict)
