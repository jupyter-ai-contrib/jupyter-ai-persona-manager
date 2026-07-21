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
    """Stable unique identifier for the persona (e.g. its class-derived ID)."""
    name: str
    """Human-readable display name shown in the selector (e.g. "Jupyternaut")."""
    avatar_url: str | None = None
    """URL that serves the persona's avatar image, or None if it has no avatar."""
    yjs_client_id: int
    """
    The Yjs client ID of this persona's awareness client. Persona client IDs
    are dynamic (they change as personas load/reload), so the manager reports
    each one here to let the browser look up a persona's awareness slot in O(1).
    """


class ModelOption(BaseModel):
    """One selectable model."""

    id: str
    """The model's unique identifier, used when selecting it (e.g. via :class:`ModelSpec`)."""
    name: str | None = None
    """Human-readable label for the model; falls back to :attr:`id` when None."""
    description: str | None = None
    """Optional longer description shown alongside the model in the picker."""


class SettingOption(BaseModel):
    """One selectable value for a setting."""

    id: str
    """The option's unique identifier, used as the selected value for its setting."""
    name: str | None = None
    """Human-readable label for the option; falls back to :attr:`id` when None."""
    description: str | None = None
    """Optional longer description shown alongside the option in the picker."""


class SettingConfiguration(BaseModel):
    """A single setting: its current value and all available options.

    Used both for model settings (rendered next to the model picker) and for
    general settings (rendered separately). The list order controls UI order.
    """

    id: str
    """ID of the setting, e.g. "agent_mode"."""
    current: str | None = None
    """The current value, or None to indicate the persona's default."""
    name: str | None = None
    """Human-readable label for the setting; falls back to :attr:`id` when None."""
    description: str | None = None
    """Optional longer description explaining what the setting controls."""
    options: list[SettingOption] = Field(default_factory=list)
    """The selectable values for this setting, in display order."""


class ModelConfiguration(BaseModel):
    """The persona's current model, its options, and its model settings."""

    current: str | None = None
    """The current model ID, or None to indicate the persona's default."""
    options: list[ModelOption] = Field(default_factory=list)
    """The models the persona offers, in display order."""
    settings: list[SettingConfiguration] = Field(default_factory=list)
    """Settings that should render near the model picker (ACP ``model_config``)."""


class Usage(BaseModel):
    """Token and cost usage reported by a persona for the current session."""

    context_tokens: int | None = None
    """
    Tokens currently in the context window. Part of the live context-window
    snapshot: unlike the cumulative counters below, ``context_tokens`` and
    ``context_size`` can decrease during a session (e.g. after the agent
    compacts context).
    """
    context_size: int | None = None
    """Total context-window size. See :attr:`context_tokens`."""
    context_percent: float | None = None
    """
    Context fill as a bare percentage (0-100), the fallback for agents that
    report only a percentage with no token counts (e.g. kiro-cli). Precedence
    contract for consumers: when ``context_tokens``/``context_size`` are
    present, derive the percentage from them and ignore this field; read this
    field only when they are absent.
    """

    input_tokens: int | None = None
    """Cumulative count of input tokens for the session."""
    output_tokens: int | None = None
    """Cumulative count of output tokens for the session."""
    cached_read_tokens: int | None = None
    """Cumulative count of tokens read from the prompt cache this session."""
    cached_write_tokens: int | None = None
    """Cumulative count of tokens written to the prompt cache this session."""
    thought_tokens: int | None = None
    """Cumulative count of reasoning ("thinking") tokens for the session."""
    total_tokens: int | None = None
    """Cumulative count of all tokens for the session."""

    cost_amount: float | None = None
    """Cumulative session cost, expressed in :attr:`cost_currency`."""
    cost_currency: str | None = None
    """
    The currency of :attr:`cost_amount`: an ISO 4217 code (e.g. "USD") or, for
    agents that meter in their own unit, that unit's plural name (e.g.
    "credits").
    """


class CommandOption(BaseModel):
    """One slash command advertised by a persona."""

    name: str
    """The command name, including the leading "/", e.g. "/compact"."""
    description: str | None = None
    """Optional short description of what the command does, shown in the UI."""


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
    """The selected model ID, or None to keep the persona's current model."""
    settings: dict[str, str | None] = Field(default_factory=dict)
    """
    Maps each model-setting ID to its selected option ID, or to None to keep
    that setting's current value.
    """
