/**
 * Tests for the frontend persona manager fed by Jupyter Events:
 * per-chat routing, the `changed` signal, and discarding a chat's state when it
 * closes.
 */
import { Event } from '@jupyterlab/services';
import { Signal } from '@lumino/signaling';

import {
  PERSONAS_EVENT_SCHEMA_ID,
  PERSONA_STATE_EVENT_SCHEMA_ID,
  PersonaSessionRegistry
} from '../persona-events';

/**
 * A stand-in for the ServiceManager event bus (`Event.IManager`) that lets
 * tests push events onto the shared stream the registry subscribes to.
 */
class FakeEventManager {
  readonly stream = new Signal<this, Event.Emission>(this);

  /** Push an event onto the stream, mirroring the jupyter_server event bus. */
  async emit(schemaId: string, data: any): Promise<void> {
    this.stream.emit({ schema_id: schemaId, ...data });
  }
}

function makeRegistry(): {
  registry: PersonaSessionRegistry;
  events: FakeEventManager;
} {
  const events = new FakeEventManager();
  const registry = new PersonaSessionRegistry(
    events as unknown as Event.IManager
  );
  return { registry, events };
}

describe('PersonaSessionRegistry', () => {
  it('routes a personas event to the matching chat by id', async () => {
    const { registry, events } = makeRegistry();
    await events.emit(PERSONAS_EVENT_SCHEMA_ID, {
      chat_id: 'a.chat',
      personas: [{ id: 'p1', name: 'One', avatar_url: null }]
    });
    expect(registry.get('a.chat').personas.map(p => p.id)).toEqual(['p1']);
    expect(registry.get('a.chat').ready).toBe(true);
    // A different chat is unaffected.
    expect(registry.has('b.chat')).toBe(false);
  });

  it('routes persona_state events and builds PersonaSessionState', async () => {
    const { registry, events } = makeRegistry();
    await events.emit(PERSONA_STATE_EVENT_SCHEMA_ID, {
      chat_id: 'a.chat',
      persona_id: 'p1',
      model: { current: 'm1', options: [], settings: [] },
      usage: { input_tokens: 5 }
    });
    const state = registry.get('a.chat').getPersona('p1');
    expect(state?.model.current).toBe('m1');
    expect(state?.usage.input_tokens).toBe(5);
  });

  it('merges partial events, leaving unset attributes unchanged', async () => {
    const { registry, events } = makeRegistry();
    // First event sets the model.
    await events.emit(PERSONA_STATE_EVENT_SCHEMA_ID, {
      chat_id: 'a.chat',
      persona_id: 'p1',
      model: { current: 'm1', options: [], settings: [] }
    });
    // A usage-only event must not drop the previously-set model.
    await events.emit(PERSONA_STATE_EVENT_SCHEMA_ID, {
      chat_id: 'a.chat',
      persona_id: 'p1',
      usage: { input_tokens: 5 }
    });
    const state = registry.get('a.chat').getPersona('p1');
    expect(state?.model.current).toBe('m1');
    expect(state?.usage.input_tokens).toBe(5);

    // A processing-only event likewise preserves model and usage.
    await events.emit(PERSONA_STATE_EVENT_SCHEMA_ID, {
      chat_id: 'a.chat',
      persona_id: 'p1',
      processing: true
    });
    const after = registry.get('a.chat').getPersona('p1');
    expect(after?.model.current).toBe('m1');
    expect(after?.usage.input_tokens).toBe(5);
    expect(after?.processing).toBe(true);
    // A new instance is produced on each merge, so React consumers re-render.
    expect(after).not.toBe(state);
  });

  it('fires the changed signal on updates', async () => {
    const { registry, events } = makeRegistry();
    const personaManager = registry.get('a.chat');
    let fired = 0;
    personaManager.changed.connect(() => {
      fired += 1;
    });
    await events.emit(PERSONAS_EVENT_SCHEMA_ID, {
      chat_id: 'a.chat',
      personas: []
    });
    await events.emit(PERSONA_STATE_EVENT_SCHEMA_ID, {
      chat_id: 'a.chat',
      persona_id: 'p1'
    });
    expect(fired).toBe(2);
  });

  it('tracks per-persona processing and exposes chat-level processing', async () => {
    const { registry, events } = makeRegistry();
    const personaManager = registry.get('a.chat');
    expect(personaManager.processing).toBe(false);

    // A persona reports it started processing.
    await events.emit(PERSONA_STATE_EVENT_SCHEMA_ID, {
      chat_id: 'a.chat',
      persona_id: 'p1',
      processing: true
    });
    expect(personaManager.getPersona('p1')?.processing).toBe(true);
    expect(personaManager.processing).toBe(true);

    // A second, idle persona doesn't flip the chat back to idle.
    await events.emit(PERSONA_STATE_EVENT_SCHEMA_ID, {
      chat_id: 'a.chat',
      persona_id: 'p2',
      processing: false
    });
    expect(personaManager.processing).toBe(true);

    // The processing persona reports it finished; chat is now idle.
    await events.emit(PERSONA_STATE_EVENT_SCHEMA_ID, {
      chat_id: 'a.chat',
      persona_id: 'p1',
      processing: false
    });
    expect(personaManager.processing).toBe(false);
  });

  it('defaults persona processing to false when the field is absent', async () => {
    const { registry, events } = makeRegistry();
    await events.emit(PERSONA_STATE_EVENT_SCHEMA_ID, {
      chat_id: 'a.chat',
      persona_id: 'p1'
    });
    expect(registry.get('a.chat').getPersona('p1')?.processing).toBe(false);
    expect(registry.get('a.chat').processing).toBe(false);
  });

  it('discards a chat persona manager on close, freeing memory', async () => {
    const { registry, events } = makeRegistry();
    await events.emit(PERSONAS_EVENT_SCHEMA_ID, {
      chat_id: 'a.chat',
      personas: [{ id: 'p1', name: 'One', avatar_url: null }]
    });
    const manager = registry.get('a.chat');
    expect(registry.has('a.chat')).toBe(true);

    registry.discard('a.chat');

    expect(registry.has('a.chat')).toBe(false);
    expect(manager.isDisposed).toBe(true);
    // A fresh get() creates a new, empty manager (the old one is gone).
    const fresh = registry.get('a.chat');
    expect(fresh).not.toBe(manager);
    expect(fresh.personas).toEqual([]);
    expect(fresh.ready).toBe(false);
  });

  describe('registerFrontendPersona', () => {
    it('marks the list as ready immediately, without a backend event', () => {
      const { registry } = makeRegistry();
      const manager = registry.get('a.chat');
      expect(manager.ready).toBe(false);
      registry.registerFrontendPersona('a.chat', {
        id: 'fp1',
        name: 'Frontend',
        avatar_url: null
      });
      expect(manager.ready).toBe(true);
    });

    it('adds the persona to the list', () => {
      const { registry } = makeRegistry();
      registry.registerFrontendPersona('a.chat', {
        id: 'fp1',
        name: 'Frontend',
        avatar_url: null
      });
      expect(registry.get('a.chat').personas.map(p => p.id)).toContain('fp1');
    });

    it('fires the changed signal', () => {
      const { registry } = makeRegistry();
      const manager = registry.get('a.chat');
      let fired = 0;
      manager.changed.connect(() => {
        fired += 1;
      });
      registry.registerFrontendPersona('a.chat', {
        id: 'fp1',
        name: 'Frontend',
        avatar_url: null
      });
      expect(fired).toBe(1);
    });

    it('survives subsequent backend personas events', async () => {
      const { registry, events } = makeRegistry();
      registry.registerFrontendPersona('a.chat', {
        id: 'fp1',
        name: 'Frontend',
        avatar_url: null
      });
      await events.emit(PERSONAS_EVENT_SCHEMA_ID, {
        chat_id: 'a.chat',
        personas: [{ id: 'bp1', name: 'Backend', avatar_url: null }]
      });
      const ids = registry.get('a.chat').personas.map(p => p.id);
      expect(ids).toContain('fp1');
      expect(ids).toContain('bp1');
    });

    it('merges with backend personas, sorted by name', async () => {
      const { registry, events } = makeRegistry();
      registry.registerFrontendPersona('a.chat', {
        id: 'fp1',
        name: 'Zara',
        avatar_url: null
      });
      await events.emit(PERSONAS_EVENT_SCHEMA_ID, {
        chat_id: 'a.chat',
        personas: [{ id: 'bp1', name: 'Alice', avatar_url: null }]
      });
      const names = registry.get('a.chat').personas.map(p => p.name);
      expect(names).toEqual(['Alice', 'Zara']);
    });
  });

  describe('unregisterFrontendPersona', () => {
    it('removes the persona from the list', () => {
      const { registry } = makeRegistry();
      registry.registerFrontendPersona('a.chat', {
        id: 'fp1',
        name: 'Frontend',
        avatar_url: null
      });
      registry.unregisterFrontendPersona('a.chat', 'fp1');
      expect(registry.get('a.chat').personas.map(p => p.id)).not.toContain(
        'fp1'
      );
    });

    it('fires the changed signal', () => {
      const { registry } = makeRegistry();
      registry.registerFrontendPersona('a.chat', {
        id: 'fp1',
        name: 'Frontend',
        avatar_url: null
      });
      const manager = registry.get('a.chat');
      let fired = 0;
      manager.changed.connect(() => {
        fired += 1;
      });
      manager.unregisterFrontendPersona('fp1');
      expect(fired).toBe(1);
    });
  });

  describe('registry-level updatePersonaState', () => {
    it('directly updates state without a backend event', () => {
      const { registry } = makeRegistry();
      registry.updatePersonaState('a.chat', 'p1', {
        model: { current: 'm1', options: [], settings: [] },
        processing: true
      });
      const state = registry.get('a.chat').getPersona('p1');
      expect(state?.model.current).toBe('m1');
      expect(state?.processing).toBe(true);
    });

    it('merges partial updates, preserving existing state', () => {
      const { registry } = makeRegistry();
      registry.updatePersonaState('a.chat', 'p1', {
        model: { current: 'm1', options: [], settings: [] }
      });
      registry.updatePersonaState('a.chat', 'p1', { processing: true });
      const state = registry.get('a.chat').getPersona('p1');
      expect(state?.model.current).toBe('m1');
      expect(state?.processing).toBe(true);
    });

    it('fires the changed signal', () => {
      const { registry } = makeRegistry();
      const manager = registry.get('a.chat');
      let fired = 0;
      manager.changed.connect(() => {
        fired += 1;
      });
      registry.updatePersonaState('a.chat', 'p1', { processing: false });
      expect(fired).toBe(1);
    });
  });
});
