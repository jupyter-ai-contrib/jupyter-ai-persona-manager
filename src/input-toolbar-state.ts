import { ISignal, Signal } from '@lumino/signaling';

import { PersonaSettings } from './metadata';
import { PersonaSessionRegistry, PersonaSessionState } from './persona-events';

/**
 * Observable model for the persona-related state of one chat input toolbar.
 * Lives outside React so toolbar remounts (between inline and overflow popup)
 * don't reset it, and so the toolbar reconciler can read it without
 * being inside a React component.
 */
export class InputToolbarState {
  constructor(sessionRegistry: PersonaSessionRegistry) {
    this._sessionRegistry = sessionRegistry;
    this.changed.connect(this._updatePersonaState, this);
    this.chatIdChanged.connect(this._resubscribeManager, this);
    this.chatIdChanged.connect(this._updatePersonaState, this);
  }

  // The currently selected persona.
  get personaId(): string | null {
    return this._id;
  }

  set personaId(value: string | null) {
    if (this._id === value) {
      return;
    }
    this._id = value;
    this._changed.emit(value);
  }

  get changed(): ISignal<this, string | null> {
    return this._changed;
  }

  // Per-persona cached settings; restores picks when switching personas.
  get settings(): Record<string, PersonaSettings> {
    return this._settings;
  }

  setSettings(personaId: string, next: PersonaSettings): void {
    if (this._settings[personaId] === next) {
      return;
    }
    this._settings = { ...this._settings, [personaId]: next };
    this._settingsChanged.emit();
  }

  get settingsChanged(): ISignal<this, void> {
    return this._settingsChanged;
  }

  // Stable chat id, published by PersonaControls once chatModel.ready resolves.
  get chatId(): string | null {
    return this._chatId;
  }

  set chatId(value: string | null) {
    if (this._chatId === value) {
      return;
    }
    this._chatId = value;
    this._chatIdChanged.emit(value);
  }

  get chatIdChanged(): ISignal<this, string | null> {
    return this._chatIdChanged;
  }

  // Live session state of the selected persona, derived from chatId + personaId.
  get personaState(): PersonaSessionState | null {
    return this._personaState;
  }

  get personaStateChanged(): ISignal<this, PersonaSessionState | null> {
    return this._personaStateChanged;
  }

  // Guards the sole-persona auto-pick in reconcileSelection.
  userPicked = false;

  private _updatePersonaState(): void {
    const managerState = this._chatId
      ? this._sessionRegistry.get(this._chatId)
      : null;
    const next =
      managerState && this._id
        ? (managerState.getPersona(this._id) ?? null)
        : null;
    if (this._personaState === next) {
      return;
    }
    this._personaState = next;
    this._personaStateChanged.emit(next);
  }

  private _resubscribeManager(): void {
    this._detachManager?.();
    this._detachManager = null;
    const managerState = this._chatId
      ? this._sessionRegistry.get(this._chatId)
      : null;
    if (managerState) {
      const handler = () => this._updatePersonaState();
      managerState.changed.connect(handler);
      this._detachManager = () => managerState.changed.disconnect(handler);
    }
  }

  private _sessionRegistry: PersonaSessionRegistry;
  private _detachManager: (() => void) | null = null;
  private _id: string | null = null;
  private _changed = new Signal<this, string | null>(this);
  private _settings: Record<string, PersonaSettings> = {};
  private _settingsChanged = new Signal<this, void>(this);
  private _chatId: string | null = null;
  private _chatIdChanged = new Signal<this, string | null>(this);
  private _personaState: PersonaSessionState | null = null;
  private _personaStateChanged = new Signal<this, PersonaSessionState | null>(
    this
  );
}
