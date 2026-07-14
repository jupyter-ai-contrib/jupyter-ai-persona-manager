/**
 * Tests for the read-only awareness views: `PersonaManagerAwareness` (the
 * persona list under the fixed client ID) and `PersonaAwareness` (one persona's
 * slot). Both read from a plain in-memory stand-in for a Yjs awareness object.
 */

import { IAwareness } from '@jupyter/ydoc';
import {
  PERSONA_MANAGER_AWARENESS_CLIENT_ID,
  PersonaAwareness,
  PersonaManagerAwareness,
  PersonaOption
} from '../awareness';

/**
 * A minimal stand-in for a Yjs awareness object implementing only `getStates()`,
 * keyed by client ID as the real one is.
 */
function fakeAwareness(
  states: Record<number, Record<string, unknown>>
): IAwareness {
  const map = new Map<number, Record<string, unknown>>(
    Object.entries(states).map(([k, v]) => [Number(k), v])
  );
  return { getStates: () => map } as unknown as IAwareness;
}

function personaSlot(partial: Record<string, unknown> = {}) {
  return {
    id: 'kiro',
    model: { current: null, options: [], settings: [] },
    settings: [],
    usage: {},
    slash_commands: [],
    isWriting: false,
    ...partial
  };
}

const KIRO: PersonaOption = {
  id: 'kiro',
  name: 'Kiro',
  avatar_url: '/k',
  yjs_client_id: 42
};

describe('PersonaManagerAwareness.from', () => {
  it('resolves immediately when the manager slot is already present', async () => {
    const awareness = fakeAwareness({
      [PERSONA_MANAGER_AWARENESS_CLIENT_ID]: { personas: [KIRO] }
    });
    const manager = await PersonaManagerAwareness.from(awareness);
    expect(manager.personas).toEqual([KIRO]);
  });
});

describe('PersonaManagerAwareness.personas', () => {
  it('reads the persona list from the manager slot', () => {
    const awareness = fakeAwareness({
      [PERSONA_MANAGER_AWARENESS_CLIENT_ID]: { personas: [KIRO] }
    });
    // Access the getter without awaiting from() by constructing via from() on an
    // already-present slot.
    return PersonaManagerAwareness.from(awareness).then(m => {
      expect(m.personas).toEqual([KIRO]);
    });
  });

  it('is empty when the slot has no personas', async () => {
    const awareness = fakeAwareness({
      [PERSONA_MANAGER_AWARENESS_CLIENT_ID]: {}
    });
    const manager = await PersonaManagerAwareness.from(awareness);
    expect(manager.personas).toEqual([]);
  });
});

describe('PersonaAwareness.from', () => {
  it("reads a persona's slot via the option's client id", () => {
    const awareness = fakeAwareness({
      42: personaSlot({
        model: {
          current: 'opus-48',
          options: [{ id: 'opus-48', name: 'Opus', description: null }],
          settings: [
            {
              id: 'context_size',
              current: '200k',
              name: 'Context',
              description: null,
              options: [{ id: '200k', name: '200K', description: null }]
            }
          ]
        },
        settings: [
          { id: '__mode__', current: 'ask', name: 'Mode', description: null, options: [] }
        ],
        usage: { context_tokens: 1000, context_size: 200000, total_tokens: 4200 },
        slash_commands: [{ name: '/compact', description: 'Compact context' }]
      })
    });

    const persona = PersonaAwareness.from(awareness, KIRO);
    expect(persona.model.current).toBe('opus-48');
    expect(persona.model.settings[0].id).toBe('context_size');
    expect(persona.settings[0].id).toBe('__mode__');
    expect(persona.usage.context_tokens).toBe(1000);
    expect(persona.slash_commands).toEqual([
      { name: '/compact', description: 'Compact context' }
    ]);
    expect(persona.isWriting).toBe(false);
  });

  it('exposes the persona id from the option, even before the slot exists', () => {
    // The id comes from the PersonaOption, so it's available immediately, not
    // gated on the persona publishing its slot.
    expect(PersonaAwareness.from(fakeAwareness({}), KIRO).id).toBe('kiro');
  });

  it('falls back to safe defaults when the slot is absent', () => {
    const persona = PersonaAwareness.from(fakeAwareness({}), KIRO);
    expect(persona.model).toEqual({ current: null, options: [], settings: [] });
    expect(persona.settings).toEqual([]);
    expect(persona.slash_commands).toEqual([]);
    expect(persona.usage.total_tokens).toBeNull();
    expect(persona.isWriting).toBe(false);
  });

  it('reads isWriting (message id) while the persona is streaming', () => {
    const awareness = fakeAwareness({ 42: personaSlot({ isWriting: 'msg-1' }) });
    expect(PersonaAwareness.from(awareness, KIRO).isWriting).toBe('msg-1');
  });
});

describe('the manager list + PersonaAwareness.from together', () => {
  it('resolves a persona by walking the published list', async () => {
    const awareness = fakeAwareness({
      [PERSONA_MANAGER_AWARENESS_CLIENT_ID]: { personas: [KIRO] },
      42: personaSlot({ slash_commands: [{ name: '/login', description: null }] })
    });
    const manager = await PersonaManagerAwareness.from(awareness);
    const option = manager.personas.find(p => p.id === 'kiro')!;
    const persona = PersonaAwareness.from(awareness, option);
    expect(persona.slash_commands).toEqual([{ name: '/login', description: null }]);
  });
});
