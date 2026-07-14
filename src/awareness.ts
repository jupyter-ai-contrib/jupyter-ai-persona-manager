/**
 * Reading persona session information from a chat's Yjs awareness channel.
 *
 * The persona-manager server extension broadcasts session info over awareness
 * instead of REST. This module gives frontends a typed, read-only view of it:
 *
 * - `PersonaManagerAwareness` reads the persona list the `PersonaManager`
 *   publishes under a fixed, hardcoded client ID.
 * - `PersonaAwareness` reads one persona's slot (model configuration, settings,
 *   usage, slash commands, writing status), keyed by that persona's Yjs client
 *   ID (reported in the persona list).
 *
 * These mirror the Python awareness helpers of the same names. They are
 * read-only: user selections ride on message metadata, not awareness.
 */

import { IAwareness } from '@jupyter/ydoc';

/**
 * The fixed Yjs client ID the `PersonaManager` publishes its persona list under.
 * A chosen 53-bit constant, hardcoded on both server and client so the browser
 * can find the manager's slot without a discovery request. Must match
 * `PERSONA_MANAGER_AWARENESS_CLIENT_ID` in the persona-manager Python package.
 */
export const PERSONA_MANAGER_AWARENESS_CLIENT_ID = 7133713371337;

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
  input_tokens: number | null;
  output_tokens: number | null;
  cached_read_tokens: number | null;
  cached_write_tokens: number | null;
  thought_tokens: number | null;
  total_tokens: number | null;
  cost_amount: number | null;
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
  /** The Yjs client ID of this persona's awareness slot. */
  yjs_client_id: number;
};

export const EMPTY_USAGE: Usage = {
  context_tokens: null,
  context_size: null,
  input_tokens: null,
  output_tokens: null,
  cached_read_tokens: null,
  cached_write_tokens: null,
  thought_tokens: null,
  total_tokens: null,
  cost_amount: null,
  cost_currency: null
};

// Interval and attempt cap for `PersonaManagerAwareness.from()` polling. The
// manager registers a moment after a chat opens, and a persona's agent session
// can take 20s+ to initialize, so poll generously before giving up.
const POLL_MS = 500;
const MAX_POLLS = 120;

const delay = (ms: number): Promise<void> =>
  new Promise(resolve => setTimeout(resolve, ms));

/**
 * A typed, read-only view of the `PersonaManager`'s awareness slot: the list of
 * personas in the chat. Construct it with `PersonaManagerAwareness.from()`,
 * which resolves once the manager has published its slot.
 */
export class PersonaManagerAwareness {
  private constructor(private _awareness: IAwareness) {}

  /**
   * Resolve once the `PersonaManager` has published its slot under the fixed
   * client ID, polling the awareness map until it appears (or rejecting after a
   * bounded wait). Pass the chat's Yjs awareness object (e.g. from the chat
   * model's shared model).
   *
   * Safe to `await` in any context: if the slot is already present it resolves
   * on the first check, effectively immediately.
   */
  static async from(awareness: IAwareness): Promise<PersonaManagerAwareness> {
    for (let i = 0; i < MAX_POLLS; i++) {
      if (awareness.getStates().has(PERSONA_MANAGER_AWARENESS_CLIENT_ID)) {
        return new PersonaManagerAwareness(awareness);
      }
      await delay(POLL_MS);
    }
    throw new Error(
      'Timed out waiting for the PersonaManager awareness slot. Is the ' +
        'jupyter-ai-persona-manager server extension installed and enabled?'
    );
  }

  /**
   * The personas available in this chat. Empty if none are published. Pass an
   * entry to `PersonaAwareness.from()` to read that persona's own slot.
   */
  get personas(): PersonaOption[] {
    const state = this._awareness
      .getStates()
      .get(PERSONA_MANAGER_AWARENESS_CLIENT_ID);
    const personas = state?.personas;
    return Array.isArray(personas) ? (personas as PersonaOption[]) : [];
  }
}

/**
 * A typed, read-only view of one persona's awareness slot. Construct it with
 * `PersonaAwareness.from(awareness, personaOption)` — the `PersonaOption` comes
 * from `PersonaManagerAwareness.personas` and carries the persona's client ID.
 */
export class PersonaAwareness {
  private constructor(
    private _awareness: IAwareness,
    private _id: string,
    private _clientId: number
  ) {}

  /** Build a view of the persona named by `option` (from the persona list). */
  static from(awareness: IAwareness, option: PersonaOption): PersonaAwareness {
    return new PersonaAwareness(awareness, option.id, option.yjs_client_id);
  }

  /** The persona's stable ID (from the persona list). */
  get id(): string {
    return this._id;
  }

  private get _state(): { [key: string]: any } | undefined {
    return this._awareness.getStates().get(this._clientId);
  }

  /** The persona's model configuration (current model, options, model settings). */
  get model(): ModelConfiguration {
    return (
      (this._state?.model as ModelConfiguration) ?? {
        current: null,
        options: [],
        settings: []
      }
    );
  }

  /** The persona's general (non-model) setting configurations. */
  get settings(): SettingConfiguration[] {
    return (this._state?.settings as SettingConfiguration[]) ?? [];
  }

  /** The token and cost usage the persona reports for the session. */
  get usage(): Usage {
    return (this._state?.usage as Usage) ?? EMPTY_USAGE;
  }

  /** The slash commands the persona advertises. */
  get slash_commands(): CommandOption[] {
    return (this._state?.slash_commands as CommandOption[]) ?? [];
  }

  /**
   * Whether the persona is currently writing: `false` when idle, or the ID of
   * the message being written while streaming.
   */
  get isWriting(): boolean | string {
    return this._state?.isWriting ?? false;
  }

  /** Whether this persona has published its state yet. */
  get isReady(): boolean {
    return this._state?.model !== undefined;
  }
}
