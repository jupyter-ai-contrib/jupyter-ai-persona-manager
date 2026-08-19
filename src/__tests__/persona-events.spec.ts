/**
 * Tests for the frontend persona session state fed by Jupyter Events:
 * per-chat routing, the `changed` signal, and discarding a chat's state when it
 * closes.
 */
import { IEventListener } from 'jupyterlab-eventlistener';

import {
  PERSONAS_EVENT_SCHEMA_ID,
  PERSONA_STATE_EVENT_SCHEMA_ID,
  PersonaSessionRegistry
} from '../persona-events';

type Listener = (m: unknown, s: string, e: any) => Promise<void>;

/** A stand-in for IEventListener that lets tests emit events to listeners. */
class FakeEventListener {
  private _bySchema = new Map<string, Listener[]>();

  addListener(schemaId: string, listener: Listener): void {
    const list = this._bySchema.get(schemaId) ?? [];
    list.push(listener);
    this._bySchema.set(schemaId, list);
  }
  async emit(schemaId: string, data: any): Promise<void> {
    for (const l of this._bySchema.get(schemaId) ?? []) {
      await l(null, schemaId, data);
    }
  }
}

function makeRegistry(): {
  registry: PersonaSessionRegistry;
  events: FakeEventListener;
} {
  const events = new FakeEventListener();
  const registry = new PersonaSessionRegistry(
    events as unknown as IEventListener
  );
  return { registry, events };
}

describe('PersonaSessionRegistry', () => {
  it('routes a personas event to the matching chat by path', async () => {
    const { registry, events } = makeRegistry();
    await events.emit(PERSONAS_EVENT_SCHEMA_ID, {
      path: 'a.chat',
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
      path: 'a.chat',
      persona_id: 'p1',
      model: { current: 'm1', options: [], settings: [] },
      usage: { input_tokens: 5 }
    });
    const state = registry.get('a.chat').getPersona('p1');
    expect(state?.model.current).toBe('m1');
    expect(state?.usage.input_tokens).toBe(5);
  });

  it('fires the changed signal on updates', async () => {
    const { registry, events } = makeRegistry();
    const managerState = registry.get('a.chat');
    let fired = 0;
    managerState.changed.connect(() => {
      fired += 1;
    });
    await events.emit(PERSONAS_EVENT_SCHEMA_ID, {
      path: 'a.chat',
      personas: []
    });
    await events.emit(PERSONA_STATE_EVENT_SCHEMA_ID, {
      path: 'a.chat',
      persona_id: 'p1'
    });
    expect(fired).toBe(2);
  });

  it('discards a chat session state on close, freeing memory', async () => {
    const { registry, events } = makeRegistry();
    await events.emit(PERSONAS_EVENT_SCHEMA_ID, {
      path: 'a.chat',
      personas: [{ id: 'p1', name: 'One', avatar_url: null }]
    });
    const state = registry.get('a.chat');
    expect(registry.has('a.chat')).toBe(true);

    registry.discard('a.chat');

    expect(registry.has('a.chat')).toBe(false);
    expect(state.isDisposed).toBe(true);
    // A fresh get() creates a new, empty state (the old one is gone).
    const fresh = registry.get('a.chat');
    expect(fresh).not.toBe(state);
    expect(fresh.personas).toEqual([]);
    expect(fresh.ready).toBe(false);
  });
});
