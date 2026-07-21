/**
 * Tests that the toolbar's controls are built from a persona's awareness state
 * (model, model settings, general settings) and reflect the user's current
 * per-message selection.
 */

import { PersonaOption } from '../awareness';

import {
  buildControls,
  navigateIndex,
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

describe('navigateIndex', () => {
  it('steps forward on ArrowDown / ArrowRight', () => {
    expect(navigateIndex(3, 0, 'ArrowDown')).toBe(1);
    expect(navigateIndex(3, 0, 'ArrowRight')).toBe(1);
  });

  it('steps backward on ArrowUp / ArrowLeft', () => {
    expect(navigateIndex(3, 2, 'ArrowUp')).toBe(1);
    expect(navigateIndex(3, 2, 'ArrowLeft')).toBe(1);
  });

  it('clamps at the ends rather than wrapping', () => {
    expect(navigateIndex(3, 2, 'ArrowDown')).toBe(2);
    expect(navigateIndex(3, 0, 'ArrowUp')).toBe(0);
  });

  it('jumps to the ends on Home / End', () => {
    expect(navigateIndex(3, 1, 'Home')).toBe(0);
    expect(navigateIndex(3, 1, 'End')).toBe(2);
  });

  it('enters the list from an unknown current value (index -1)', () => {
    // ArrowDown lands on the first value; ArrowUp on the last.
    expect(navigateIndex(3, -1, 'ArrowDown')).toBe(0);
    expect(navigateIndex(3, -1, 'ArrowUp')).toBe(2);
  });

  it('ignores non-navigation keys, so they act normally', () => {
    expect(navigateIndex(3, 0, 'Tab')).toBeNull();
    expect(navigateIndex(3, 0, 'Enter')).toBeNull();
    expect(navigateIndex(3, 0, ' ')).toBeNull();
    expect(navigateIndex(3, 0, 'a')).toBeNull();
  });

  it('makes no decision for an empty list', () => {
    expect(navigateIndex(0, -1, 'ArrowDown')).toBeNull();
  });
});
