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
    /**
     * Permission requests raised by a persona, awaiting the user's decision.
     * Written by the server (`BasePersona.request_permission`) as a list keyed
     * by `request_id`, so one message can host several without clobbering. The
     * frontend renders each block and emits a `permission_response` event on
     * click.
     */
    permission_requests?: {
      request_id: string;
      persona_id: string;
      chat_id: string;
      title: string;
      detail?: string | null;
      diffs?:
        | { path: string; new_text: string; old_text?: string | null }[]
        | null;
      correlation_id?: string | null;
      options: { option_id: string; name: string; kind?: string | null }[];
      status: 'pending' | 'resolved';
      selected_option_id?: string | null;
    }[];
    /**
     * Tool calls reported by a persona (`BasePersona.report_tool_call`), in
     * creation order. The frontend renders each with a status glyph, an
     * optional detail line, file diffs, and expandable output. A tool call may
     * also carry an attached permission request (the approve/deny buttons then
     * render on its row and emit a `permission_response` event).
     */
    tool_calls?: {
      tool_call_id: string;
      title: string;
      kind?: string | null;
      status?: 'in_progress' | 'completed' | 'failed' | (string & {}) | null;
      locations?: string[] | null;
      raw_input?: unknown;
      raw_output?: unknown;
      diffs?:
        | { path: string; new_text: string; old_text?: string | null }[]
        | null;
      // Attached permission (present only when a request targets this call):
      permission_options?:
        | { option_id: string; name: string; kind?: string | null }[]
        | null;
      permission_status?: 'pending' | 'resolved' | null;
      selected_option_id?: string | null;
      request_id?: string | null;
      chat_id?: string | null;
      persona_id?: string | null;
    }[];
  }
}
