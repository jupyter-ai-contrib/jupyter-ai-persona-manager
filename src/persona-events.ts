/**
 * Frontend persona session state, delivered over Jupyter Events (via the
 * `ServiceManager` event bus), replacing the previous Yjs-awareness channel.
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
export type PersonaStatePayload = {
  chat_id?: string;
  persona_id?: string;
  model?: ModelConfiguration;
  settings?: SettingConfiguration[];
  usage?: Usage;
  slash_commands?: CommandOption[];
  processing?: boolean;
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
    this.processing = payload.processing ?? false;
  }

  readonly model: ModelConfiguration;
  readonly settings: SettingConfiguration[];
  readonly usage: Usage;
  readonly slash_commands: CommandOption[];
  /** Whether the persona is currently processing a message. */
  readonly processing: boolean;

  /**
   * Merge a `persona_state` event payload onto this state, returning a new
   * instance. Attributes the event omits are carried forward unchanged, so a
   * partial event (e.g. usage-only) replaces only what it carries. A new
   * reference is returned so React consumers re-render.
   */
  withUpdate(payload: PersonaStatePayload): PersonaSessionState {
    return new PersonaSessionState(this.id, {
      model: payload.model ?? this.model,
      settings: payload.settings ?? this.settings,
      usage: payload.usage ?? this.usage,
      slash_commands: payload.slash_commands ?? this.slash_commands,
      processing: payload.processing ?? this.processing
    });
  }
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

  /** Whether one or several personas have been registered. */
  get ready(): boolean {
    return this._personasReceived;
  }

  /** The personas advertised in this chat (backend + frontend). */
  get personas(): PersonaOption[] {
    const personas = [
      ...this._backendPersonas,
      ...this._frontendPersonas.values()
    ];
    personas.sort((a, b) => (a.name >= b.name ? 1 : -1));
    return personas;
  }

  /** A persona's session state, or undefined if it has not published yet. */
  getPersona(id: string): PersonaSessionState | undefined {
    return this._states.get(id);
  }

  /**
   * Whether any persona in this chat is currently processing a message. The
   * stop button uses this to enable itself, since a persona can be processing
   * (thinking, running tools, awaiting an agent turn) without actively writing.
   */
  get processing(): boolean {
    for (const state of this._states.values()) {
      if (state.processing) {
        return true;
      }
    }
    return false;
  }

  /** Apply a `personas` event payload, keeping any registered frontend personas. */
  updateBackendPersonas(personas: PersonaOption[]): void {
    this._personasReceived = true;
    this._backendPersonas = personas;
    this._changed.emit();
  }

  /**
   * Register a frontend-only persona (one with no backend counterpart). It
   * survives subsequent `updatePersonas` calls and immediately marks the list
   * as ready, so the toolbar never shows the loading placeholder when only
   * frontend personas are available.
   */
  registerFrontendPersona(persona: PersonaOption): void {
    this._personasReceived = true;
    this._frontendPersonas.set(persona.id, persona);
    this._changed.emit();
  }

  /**
   * Unregister a frontend persona from its ID.
   */
  unregisterFrontendPersona(personaId: string): void {
    this._frontendPersonas.delete(personaId);
    this._changed.emit();
  }

  /**
   * Apply a `persona_state` event payload for one persona, merging it onto the
   * persona's existing state (absent attributes are left unchanged).
   */
  updatePersonaState(personaId: string, payload: PersonaStatePayload): void {
    const prev = this._states.get(personaId);
    const next = prev
      ? prev.withUpdate(payload)
      : new PersonaSessionState(personaId, payload);
    this._states.set(personaId, next);
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
    this._frontendPersonas.clear();
    Signal.clearData(this);
  }

  private _personasReceived = false;
  private _backendPersonas: PersonaOption[] = [];
  private _frontendPersonas = new Map<string, PersonaOption>();
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
  constructor(events: Event.IManager) {
    // The ServiceManager event bus (JupyterLab >= 4.0) exposes a single shared
    // stream of all Jupyter Events; filter it by schema id to route the two
    // persona event types. This supersedes the former `jupyterlab-eventlistener`
    // dependency.
    events.stream.connect((_, emission) => {
      if (emission.schema_id === PERSONAS_EVENT_SCHEMA_ID) {
        void this._onPersonasEvent(emission);
      } else if (emission.schema_id === PERSONA_STATE_EVENT_SCHEMA_ID) {
        void this._onPersonaStateEvent(emission);
      }
    });
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
   * Register a frontend-only persona for a chat. Shorthand for
   * `registry.get(chatId).registerFrontendPersona(persona)`.
   */
  registerFrontendPersona(chatId: string, persona: PersonaOption): void {
    this.get(chatId).registerFrontendPersona(persona);
  }

  /**
   * Update a persona's state for a chat. Shorthand for
   * `registry.get(chatId).updatePersonaState(personaId, payload)`.
   */
  updatePersonaState(
    chatId: string,
    personaId: string,
    payload: PersonaStatePayload
  ): void {
    this.get(chatId).updatePersonaState(personaId, payload);
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

  private _onPersonasEvent = async (event: Event.Emission): Promise<void> => {
    const data = event as PersonasPayload;
    if (!data.chat_id) {
      return;
    }
    this.get(data.chat_id).updateBackendPersonas(
      Array.isArray(data.personas) ? data.personas : []
    );
  };

  private _onPersonaStateEvent = async (
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
