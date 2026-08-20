/**
 * Shared persona session data types.
 *
 * These mirror the Pydantic models in the persona-manager Python package and are
 * the payloads carried by the `personas` / `persona_state` Jupyter Events. The
 * live, per-chat state built from those events lives in
 * `PersonaManagerSessionState` / `PersonaSessionState` (see `./persona-events`).
 */

/** A selectable model. Mirrors `ModelOption` in the Python package. */
export type ModelOption = {
  id: string;
  name: string | null;
  description: string | null;
};

/** A selectable value for a setting. Mirrors `SettingOption`. */
export type SettingOption = {
  id: string;
  name: string | null;
  description: string | null;
};

/**
 * A single setting: its current value and all options. Used both for model
 * settings (rendered near the model picker) and general settings. Mirrors
 * `SettingConfiguration`. `current` is null when the persona's default applies.
 */
export type SettingConfiguration = {
  id: string;
  current: string | null;
  name: string | null;
  description: string | null;
  options: SettingOption[];
};

/**
 * The persona's current model, its options, and its model settings. Mirrors
 * `ModelConfiguration`.
 */
export type ModelConfiguration = {
  current: string | null;
  options: ModelOption[];
  settings: SettingConfiguration[];
};

/**
 * Token and cost usage reported by a persona. Mirrors `Usage`. Every field is
 * null until the persona reports it.
 */
export type Usage = {
  context_tokens: number | null;
  context_size: number | null;
  /**
   * Context fill as a bare percentage (0-100), the fallback for agents that
   * report only a percentage with no token counts. Precedence contract: when
   * `context_tokens`/`context_size` are present, derive the percentage from
   * them and ignore this field; read this field only when they are absent.
   */
  context_percent: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  cached_read_tokens: number | null;
  cached_write_tokens: number | null;
  thought_tokens: number | null;
  total_tokens: number | null;
  cost_amount: number | null;
  /**
   * An ISO 4217 code (e.g. "USD") or, for agents that meter in their own
   * unit, that unit's plural name (e.g. "credits").
   */
  cost_currency: string | null;
};

/** One slash command advertised by a persona. Mirrors `CommandOption`. */
export type CommandOption = {
  name: string;
  description: string | null;
};

/** One persona in the chat, as advertised by the manager. Mirrors `PersonaOption`. */
export type PersonaOption = {
  id: string;
  name: string;
  avatar_url: string | null;
  /** Deprecated: legacy Yjs client id, unused now that state rides events. */
  yjs_client_id?: number;
};

export const EMPTY_USAGE: Usage = {
  context_tokens: null,
  context_size: null,
  context_percent: null,
  input_tokens: null,
  output_tokens: null,
  cached_read_tokens: null,
  cached_write_tokens: null,
  thought_tokens: null,
  total_tokens: null,
  cost_amount: null,
  cost_currency: null
};

export const EMPTY_MODEL_CONFIGURATION: ModelConfiguration = {
  current: null,
  options: [],
  settings: []
};
