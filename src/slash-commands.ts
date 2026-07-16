import { ChatCommand, IChatCommandProvider, IInputModel } from '@jupyter/chat';

import { PersonaAwareness, PersonaManagerAwareness } from './awareness';

export const SLASH_COMMAND_PROVIDER_ID =
  '@jupyter-ai/persona-manager:slash-command-provider';

/**
 * A command provider that provides completions for slash commands and handles
 * slash command calls.
 *
 * - Slash commands are intended for "chat-level" operations that aren't
 * specific to any persona.
 *
 * - Slash commands may only appear one-at-a-time, and only do something if the
 * first word of the input specifies a slash command `/{slash-command-id}`.
 *
 * - Note: In v2, slash commands were reserved for specific tasks like
 * 'generate' or 'learn'. But because tasks are handled by AI personas via agent
 * tools in v3, slash commands in v3 are reserved for "chat-level" operations
 * that are not specific to an AI persona.
 */
export class SlashCommandProvider implements IChatCommandProvider {
  public id: string = SLASH_COMMAND_PROVIDER_ID;

  /**
   * Returns slash command completions for the current input.
   *
   * Slash commands are read from the selected persona's awareness slot — the
   * same source the toolbar reads — with no REST call. The target persona is the
   * one the picker stamped onto the input metadata as `to_persona`.
   */
  async listCommandCompletions(
    inputModel: IInputModel
  ): Promise<ChatCommand[]> {
    const currentWord = inputModel.currentWord || '';

    // return early if current word doesn't start with '/'.
    if (!currentWord.startsWith('/')) {
      return [];
    }

    const awareness = inputModel.chatContext?.awareness ?? null;
    if (!awareness) {
      return [];
    }

    const personaId = inputModel.getMetadata?.().to_persona ?? null;
    if (!personaId) {
      return [];
    }

    // `from()` resolves immediately when the manager is already registered
    // (the normal case). If it never registers, this rejects after a bounded
    // wait; a late/failed completion query is harmless.
    let manager: PersonaManagerAwareness;
    try {
      manager = await PersonaManagerAwareness.from(awareness);
    } catch {
      return [];
    }
    const option = manager.personas.find(p => p.id === personaId);
    if (!option) {
      return [];
    }
    const persona = PersonaAwareness.from(awareness, option);

    const commandSuggestions: ChatCommand[] = [];
    for (const cmd of persona.slash_commands) {
      const name = cmd.name.startsWith('/') ? cmd.name : `/${cmd.name}`;
      // continue if command does not match current word
      if (!name.startsWith(currentWord)) {
        continue;
      }
      commandSuggestions.push({
        name,
        providerId: this.id,
        description: cmd.description ?? undefined,
        spaceOnAccept: true
      });
    }

    return commandSuggestions;
  }

  async onSubmit(inputModel: IInputModel): Promise<void> {
    // no-op. Persona slash commands are handled by the persona's backend.
    return;
  }
}
