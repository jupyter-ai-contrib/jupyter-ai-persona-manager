// Required: makes this a module file so `declare module` below augments
// @jupyter/chat rather than replacing it with an ambient module declaration.
export {};

declare module '@jupyter/chat' {
  export interface IMessageMetadata {
    /**
     * ID of the persona this message is directed to. Read by the
     * PersonaManager to route the message.
     */
    to_persona?: string | null;
    /**
     * The user's model selection for the addressed persona. `id` is the chosen
     * model ID (null = use the persona's current model); `settings` maps model
     * setting IDs to a chosen option ID (null = use the current value).
     * Mirrors the Python `ModelSpec`.
     */
    model?: {
      id: string | null;
      settings: { [id: string]: string | null };
    };
    /**
     * The user's general (non-model) setting selections, keyed by setting ID.
     * A null value means "use the persona's current value".
     */
    settings?: { [id: string]: string | null };
  }
}
