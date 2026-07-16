import { Token } from '@lumino/coreutils';
import { IChatModel, IInputModel } from '@jupyter/chat';
import * as React from 'react';

/**
 * Props passed to a contributed persona control. They give the control the
 * context it needs to act: which persona is selected, and the chat and input
 * models of the toolbar it is rendered in.
 */
export interface IPersonaControlProps {
  /** The ID of the persona currently selected in the persona picker. */
  personaId: string;
  /** The chat model, for reading chat state / awareness. */
  chatModel?: IChatModel;
  /** The input model, for reading or stamping message metadata. */
  model: IInputModel;
}

/**
 * A control contributed to the persona controls in the chat input toolbar.
 *
 * This is the extension point that lets a persona's frontend plugin add its own
 * control (e.g. a settings button) next to the built-in persona picker, model
 * selector, and usage chip — without the persona-manager hard-coding anything
 * persona-specific.
 */
export interface IPersonaControl {
  /** A unique ID for this control (used as the React key and for dedup). */
  id: string;
  /**
   * The React component rendered for this control. It receives
   * `IPersonaControlProps`.
   */
  component: React.FunctionComponent<IPersonaControlProps>;
  /**
   * The ID of the persona this control belongs to. The control is only rendered
   * when that persona is selected. Omit to render the control for every persona.
   */
  personaId?: string;
  /**
   * Sort order among contributed controls (ascending). Controls with the same
   * rank keep registration order. Defaults to 0.
   */
  rank?: number;
}

/**
 * A registry of controls contributed to the persona controls toolbar. Extensions
 * obtain it via the `IPersonaControlRegistry` token and call `addControl` to add
 * their own controls; the persona controls render the ones that apply to the
 * selected persona.
 */
export interface IPersonaControlRegistry {
  /** Register a control. */
  addControl(control: IPersonaControl): void;
  /**
   * Return the controls to render for the given selected persona, in rank order:
   * controls scoped to that persona plus controls scoped to all personas.
   */
  getControls(personaId: string): IPersonaControl[];
}

/**
 * The token used to require/provide the persona control registry.
 */
export const IPersonaControlRegistry = new Token<IPersonaControlRegistry>(
  '@jupyter-ai/persona-manager:IPersonaControlRegistry',
  'A registry for contributing controls to the persona controls in the chat input toolbar.'
);

/**
 * Default `IPersonaControlRegistry` implementation.
 */
export class PersonaControlRegistry implements IPersonaControlRegistry {
  private _controls: IPersonaControl[] = [];

  addControl(control: IPersonaControl): void {
    // Replace any existing control with the same ID so re-registration (e.g. on
    // a plugin reload) doesn't duplicate it.
    this._controls = this._controls.filter(c => c.id !== control.id);
    this._controls.push(control);
  }

  getControls(personaId: string): IPersonaControl[] {
    return this._controls
      .filter(c => c.personaId === undefined || c.personaId === personaId)
      .sort((a, b) => (a.rank ?? 0) - (b.rank ?? 0));
  }
}
