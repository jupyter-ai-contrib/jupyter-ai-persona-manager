/**
 * Shared test helper: build a real `PersonaSessionState` from a partial payload,
 * so specs exercise the actual model (its fields) rather than a hand-rolled
 * shape.
 */
import { PersonaSessionState } from '../persona-events';

/**
 * A `PersonaSessionState` whose fields are the given partial (merged over empty
 * defaults). Pass `null` for "no persona state".
 */
export function personaAwareness(
  slot: Record<string, any> | null
): PersonaSessionState | null {
  if (slot === null) {
    return null;
  }
  return new PersonaSessionState('kiro', {
    model: slot.model ?? { current: null, options: [], settings: [] },
    settings: slot.settings ?? [],
    usage: slot.usage ?? {},
    slash_commands: slot.slash_commands ?? []
  });
}
