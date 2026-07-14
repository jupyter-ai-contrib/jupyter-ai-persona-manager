"""
Pydantic models for the persona-manager awareness channel.

Session information (the persona list, and each persona's model, model
settings, general settings, usage, and slash commands) is broadcast over the
chat's Yjs awareness channel rather than fetched through REST polling. The
awareness map is keyed by Yjs client IDs, with values being arbitrary dicts.
This module defines the subset of that map written by the `PersonaManager` and
its personas:

- The `PersonaManager` registers one awareness client (a fixed client ID) whose
  state is a `PersonaManagerAwarenessState` — the list of personas in the chat.
- Each persona's awareness client holds a `PersonaAwarenessState` — that
  persona's model configuration, settings, usage, and slash commands.

User selections are *not* stored here. They ride on outgoing message metadata
(see `ModelSpec`), so each user's choices are private and applied per message.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

################################################
# PersonaManager awareness state
################################################


class PersonaOption(BaseModel):
    """One persona in the chat, as advertised in the persona selector."""

    id: str
    name: str
    avatar_url: str | None = None
    # The Yjs client ID of this persona's awareness client. Persona client IDs
    # are dynamic (they change as personas load/reload), so the manager reports
    # each one here to let the browser look up a persona's `PersonaAwarenessState`
    # in O(1).
    yjs_client_id: int


class PersonaManagerAwarenessState(BaseModel):
    """The `PersonaManager`'s awareness state: the list of available personas."""

    personas: list[PersonaOption] = Field(default_factory=list)


################################################
# Persona awareness state
################################################


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


class PersonaAwarenessState(BaseModel):
    """
    A single persona's awareness state. This *is* the dict published under the
    persona's Yjs client ID (each field is a top-level entry of the awareness
    slot), so this model documents the exact shape clients read.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str
    model: ModelConfiguration = Field(default_factory=ModelConfiguration)
    settings: list[SettingConfiguration] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    slash_commands: list[CommandOption] = Field(default_factory=list)

    # Whether this persona is currently writing a reply. `False` when idle;
    # while streaming, the ID of the message being written (jupyter-chat reads
    # this to render the typing indicator and enable the stop button). Written
    # directly to awareness on the hot path (see `BasePersona.stream_message`),
    # not through the config broadcast — this annotation just types the slot.
    # The awareness key is `isWriting` (camelCase) for jupyter-chat.
    is_writing: bool | str = Field(default=False, alias="isWriting")


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
