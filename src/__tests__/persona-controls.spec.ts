/**
 * Tests that the toolbar's controls are built from a persona's awareness state
 * (model, model settings, general settings) and reflect the user's current
 * per-message selection.
 */

import { PersonaOption } from '../awareness';

import {
  buildControls,
  filterChoices,
  reconcilePersonas,
  reconcileSelection,
  showLoadingPlaceholder
} from '../persona-controls';
import { emptyPersonaSettings, PersonaSettings } from '../metadata';
import { personaAwareness } from './awareness-fixture';

function personaOption(id: string): PersonaOption {
  return { id, name: id, avatar_url: null, yjs_client_id: 1 };
}

const withControls = personaAwareness({
  model: {
    current: 'opus-48',
    options: [
      { id: 'opus-48', name: 'Opus 4.8', description: null },
      { id: 'fable-5', name: 'Fable 5', description: null }
    ],
    settings: [
      {
        id: 'context_size',
        current: '200k',
        name: 'Context size',
        description: null,
        options: [{ id: '200k', name: '200K', description: null }]
      }
    ]
  },
  settings: [
    {
      id: '__mode__',
      current: 'ask',
      name: 'Mode',
      description: null,
      options: [
        { id: 'ask', name: 'Ask', description: null },
        { id: 'code', name: 'Code', description: null }
      ]
    }
  ]
});

describe('buildControls', () => {
  it('returns nothing when there is no persona state', () => {
    expect(buildControls(null, emptyPersonaSettings())).toEqual([]);
  });

  it('builds a model control, its model settings, then general settings, in order', () => {
    const controls = buildControls(withControls, emptyPersonaSettings());
    expect(controls.map(p => [p.id, p.kind])).toEqual([
      ['__model__', 'model'],
      ['context_size', 'model_setting'],
      ['__mode__', 'setting']
    ]);
  });

  it('omits the model control when the persona advertises no models', () => {
    const controls = buildControls(
      personaAwareness({
        settings: [
          {
            id: '__mode__',
            current: 'ask',
            name: 'Mode',
            description: null,
            options: []
          }
        ]
      }),
      emptyPersonaSettings()
    );
    expect(controls.map(p => p.id)).toEqual(['__mode__']);
  });

  it('carries the persona current value from awareness onto each control', () => {
    const controls = buildControls(withControls, emptyPersonaSettings());
    const model = controls.find(p => p.id === '__model__')!;
    expect(model.current).toBe('opus-48');
    const mode = controls.find(p => p.id === '__mode__')!;
    expect(mode.current).toBe('ask');
  });

  it('reflects the user selection on each control', () => {
    const settings: PersonaSettings = {
      modelId: 'fable-5',
      modelSettings: { context_size: null },
      settings: { __mode__: 'code' }
    };
    const controls = buildControls(withControls, settings);
    expect(controls.find(p => p.id === '__model__')!.selection).toBe('fable-5');
    // Left at default → null selection (renders as the persona's current value).
    expect(controls.find(p => p.id === 'context_size')!.selection).toBeNull();
    expect(controls.find(p => p.id === '__mode__')!.selection).toBe('code');
  });

  it('maps each model option into the control choices', () => {
    const controls = buildControls(withControls, emptyPersonaSettings());
    const model = controls.find(p => p.id === '__model__')!;
    expect(model.options).toEqual([
      { id: 'opus-48', name: 'Opus 4.8', description: null },
      { id: 'fable-5', name: 'Fable 5', description: null }
    ]);
  });
});

describe('reconcilePersonas', () => {
  it('accepts a fresh non-empty list', () => {
    const previous = [personaOption('a')];
    const next = [personaOption('a'), personaOption('b')];
    expect(reconcilePersonas(previous, next)).toBe(next);
  });

  it('keeps the previous list on a transient empty read', () => {
    const previous = [personaOption('a'), personaOption('b')];
    expect(reconcilePersonas(previous, [])).toBe(previous);
  });

  it('stays empty when nothing has ever loaded', () => {
    expect(reconcilePersonas([], [])).toEqual([]);
  });
});

describe('reconcileSelection', () => {
  it('keeps a valid selection', () => {
    expect(
      reconcileSelection([personaOption('a'), personaOption('b')], 'a', true)
    ).toBeUndefined();
  });

  it('selects the sole persona before any explicit choice', () => {
    expect(reconcileSelection([personaOption('a')], null, false)).toBe('a');
  });

  it('keeps "No one" once the user picked it, even with a sole persona', () => {
    expect(
      reconcileSelection([personaOption('a')], null, true)
    ).toBeUndefined();
  });

  it('replaces an invalid selection with the sole persona before any choice', () => {
    expect(reconcileSelection([personaOption('a')], 'missing', false)).toBe(
      'a'
    );
  });

  it('clears an invalid selection once the user has picked', () => {
    expect(
      reconcileSelection([personaOption('a')], 'missing', true)
    ).toBeNull();
  });

  it('clears an invalid selection among several personas', () => {
    expect(
      reconcileSelection(
        [personaOption('a'), personaOption('b')],
        'missing',
        false
      )
    ).toBeNull();
  });

  it('makes no decision before personas load', () => {
    expect(reconcileSelection([], 'a', false)).toBeUndefined();
  });
});

describe('showLoadingPlaceholder', () => {
  it('shows while the manager is still resolving', () => {
    expect(showLoadingPlaceholder(true, false, false, false)).toBe(true);
  });

  it('shows until the first persona-list read lands', () => {
    expect(showLoadingPlaceholder(true, true, false, false)).toBe(true);
  });

  it('hides once the list has been read (an empty chat is empty, not loading)', () => {
    expect(showLoadingPlaceholder(true, true, false, true)).toBe(false);
  });

  it('hides when resolution failed', () => {
    expect(showLoadingPlaceholder(true, false, true, false)).toBe(false);
  });

  it('hides without an awareness channel to wait on', () => {
    expect(showLoadingPlaceholder(false, false, false, false)).toBe(false);
  });
});

describe('filterChoices', () => {
  const opus = { id: 'opus-48', primary: 'Opus 4.8', description: null };
  const fable = { id: 'fable-5', primary: 'Fable 5', description: null };
  const defaultChoice = {
    id: null,
    primary: 'Default (Opus 4.8)',
    description: null
  };
  const choices = [defaultChoice, opus, fable];

  it('returns everything for an empty query', () => {
    expect(filterChoices(choices, '')).toEqual(choices);
  });

  it('returns everything for a whitespace-only query', () => {
    expect(filterChoices(choices, '   ')).toEqual(choices);
  });

  it('matches case-insensitively as a substring', () => {
    expect(filterChoices(choices, 'fable')).toEqual([defaultChoice, fable]);
    expect(filterChoices(choices, 'FABLE')).toEqual([defaultChoice, fable]);
    expect(filterChoices(choices, 'abl')).toEqual([defaultChoice, fable]);
  });

  it('always keeps the Default row first, even when it would not match', () => {
    expect(filterChoices(choices, 'fable')).toEqual([defaultChoice, fable]);
  });

  it('drops non-Default choices with no match', () => {
    expect(filterChoices(choices, 'nonexistent')).toEqual([defaultChoice]);
  });

  it('handles a Default-only list (no options advertised)', () => {
    expect(filterChoices([defaultChoice], 'anything')).toEqual([defaultChoice]);
  });
});
