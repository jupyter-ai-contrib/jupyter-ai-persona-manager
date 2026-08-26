/**
 * Frontend persona session state, delivered over Jupyter Events (via
 * `jupyterlab-eventlistener`), replacing the previous Yjs-awareness channel.
 *
 * Flow: the server emits `personas` / `persona_state` events (each carrying the
 * chat's stable `chat_id`); the `PersonaSessionRegistry` routes them to the
 * per-chat `PersonaManagerSessionState`, which holds the persona list and a
 * `PersonaSessionState` per persona and fires a Lumino `changed` signal. React
 * components listen to that signal. When a chat closes, its state is discarded.
 *
 * Names mirror the Python package (`PersonaManagerSessionState`,
 * `PersonaSessionState`).
 */
import { Event } from '@jupyterlab/services';
import { IEventListener } from 'jupyterlab-eventlistener';
import { Token } from '@lumino/coreutils';
import { IDisposable } from '@lumino/disposable';
import { ISignal, Signal } from '@lumino/signaling';

import {
  CommandOption,
  EMPTY_MODEL_CONFIGURATION,
  EMPTY_USAGE,
  ModelConfiguration,
  PersonaOption,
  SettingConfiguration,
  Usage
} from './awareness';

export const PERSONAS_EVENT_SCHEMA_ID =
  'https://schema.jupyter.org/jupyter_ai_persona_manager/personas/v1';
export const PERSONA_STATE_EVENT_SCHEMA_ID =
  'https://schema.jupyter.org/jupyter_ai_persona_manager/persona_state/v1';
export const PERSONA_SELECTED_EVENT_SCHEMA_ID =
  'https://schema.jupyter.org/jupyter_ai_persona_manager/persona_selected/v1';

/** The wire shape of a `persona_state` event. */
type PersonaStatePayload = {
  chat_id?: string;
  persona_id?: string;
  model?: ModelConfiguration;
  settings?: SettingConfiguration[];
  usage?: Usage;
  slash_commands?: CommandOption[];
};

/** The wire shape of a `personas` event. */
type PersonasPayload = {
  chat_id?: string;
  personas?: PersonaOption[];
};

/**
 * One persona's live session state, built from a `persona_state` event.
 * Immutable: each update produces a new instance, so React consumers see a new
 * reference and re-render.
 */
export class PersonaSessionState {
  constructor(
    public readonly id: string,
    payload: PersonaStatePayload = {}
  ) {
    this.model = payload.model ?? EMPTY_MODEL_CONFIGURATION;
    this.settings = payload.settings ?? [];
    this.usage = { ...EMPTY_USAGE, ...(payload.usage ?? {}) };
    this.slash_commands = payload.slash_commands ?? [];
  }

  readonly model: ModelConfiguration;
  readonly settings: SettingConfiguration[];
  readonly usage: Usage;
  readonly slash_commands: CommandOption[];
}

/**
 * The per-chat manager session state: the persona list plus a
 * `PersonaSessionState` per persona. Fires `changed` whenever the persona list
 * or any persona's state updates, so React components re-render.
 */
export class PersonaManagerSessionState implements IDisposable {
  constructor(public readonly chatId: string) {}

  /** Emits whenever the persona list or a persona's state changes. */
  get changed(): ISignal<this, void> {
    return this._changed;
  }

  /** Whether a `personas` event has been received for this chat yet. */
  get ready(): boolean {
    return this._personasReceived;
  }

  /** The personas advertised in this chat. */
  get personas(): PersonaOption[] {
    return this._personas;
  }

  /** A persona's session state, or undefined if it has not published yet. */
  getPersona(id: string): PersonaSessionState | undefined {
    return this._states.get(id);
  }

  /** Apply a `personas` event payload. */
  updatePersonas(personas: PersonaOption[]): void {
    this._personasReceived = true;
    this._personas = personas;
    this._changed.emit();
  }

  /** Apply a `persona_state` event payload for one persona. */
  updatePersonaState(personaId: string, payload: PersonaStatePayload): void {
    this._states.set(personaId, new PersonaSessionState(personaId, payload));
    this._changed.emit();
  }

  get isDisposed(): boolean {
    return this._isDisposed;
  }

  dispose(): void {
    if (this._isDisposed) {
      return;
    }
    this._isDisposed = true;
    this._states.clear();
    Signal.clearData(this);
  }

  private _personasReceived = false;
  private _personas: PersonaOption[] = [];
  private _states = new Map<string, PersonaSessionState>();
  private _isDisposed = false;
  private _changed = new Signal<this, void>(this);
}

/**
 * Routes persona events to the correct per-chat `PersonaManagerSessionState`,
 * creating one on demand and discarding it when the chat closes.
 *
 * A single instance is created by the plugin and shared with the toolbar
 * controls and the slash-command provider.
 */
export class PersonaSessionRegistry {
  constructor(eventListener: IEventListener) {
    eventListener.addListener(
      PERSONAS_EVENT_SCHEMA_ID,
      this._onPersonasEvent as any
    );
    eventListener.addListener(
      PERSONA_STATE_EVENT_SCHEMA_ID,
      this._onPersonaStateEvent as any
    );
  }

  /**
   * Get (or create) the manager session state for a chat. Components call this
   * with their chat's stable id (`IChatModel.id` / `IChatContext.id`).
   */
  get(chatId: string): PersonaManagerSessionState {
    let state = this._byChatId.get(chatId);
    if (!state) {
      state = new PersonaManagerSessionState(chatId);
      this._byChatId.set(chatId, state);
    }
    return state;
  }

  /** Whether a manager session state exists for `chatId` (without creating one). */
  has(chatId: string): boolean {
    return this._byChatId.has(chatId);
  }

  /**
   * Discard a chat's session state and free its memory. Called when the client
   * closes the chat (wired to the chat model's `disposed` signal).
   */
  discard(chatId: string): void {
    const state = this._byChatId.get(chatId);
    if (state) {
      state.dispose();
      this._byChatId.delete(chatId);
    }
  }

  private _onPersonasEvent = async (
    _manager: unknown,
    _schemaId: string,
    event: Event.Emission
  ): Promise<void> => {
    const data = event as PersonasPayload;
    if (!data.chat_id) {
      return;
    }
    this.get(data.chat_id).updatePersonas(
      Array.isArray(data.personas) ? data.personas : []
    );
  };

  private _onPersonaStateEvent = async (
    _manager: unknown,
    _schemaId: string,
    event: Event.Emission
  ): Promise<void> => {
    const data = event as PersonaStatePayload;
    if (!data.chat_id || !data.persona_id) {
      return;
    }
    this.get(data.chat_id).updatePersonaState(data.persona_id, data);
  };

  private _byChatId = new Map<string, PersonaManagerSessionState>();
}

/**
 * Plugin token for the shared `PersonaSessionRegistry`. Consumed by the toolbar
 * controls and the slash-command provider.
 */
export const IPersonaSessionRegistry = new Token<PersonaSessionRegistry>(
  '@jupyter-ai/persona-manager:IPersonaSessionRegistry'
);
