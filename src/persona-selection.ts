import { ISignal, Signal } from '@lumino/signaling';

import { PersonaSettings } from './metadata';

/**
 * Per-chat persistent state for the input toolbar's persona controls. Lives
 * outside React so that moves between the inline toolbar and the overflow
 * popup — which unmount and remount `PersonaControls` — do not reset the
 * user's selection, cached per-persona settings, or sole-persona guard.
 */
export class PersonaSelection {
  get id(): string | null {
    return this._id;
  }

  set id(value: string | null) {
    if (this._id === value) {
      return;
    }
    this._id = value;
    this._changed.emit(value);
  }

  get changed(): ISignal<this, string | null> {
    return this._changed;
  }

  // Whether the user has made an explicit choice at least once.
  // Used to select the unique persona if there is only one and the user did not
  // already explicitly choose no-one.
  userPicked = false;
  // Per-persona cached settings.
  settings: Record<string, PersonaSettings> = {};

  private _id: string | null = null;
  private _changed = new Signal<this, string | null>(this);
}
