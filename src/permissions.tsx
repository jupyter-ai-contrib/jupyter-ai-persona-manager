/**
 * Frontend for the general persona permission API.
 *
 * A persona's permission request is reflected in the chat as a
 * `permission_request` block on a message's metadata (written by the server's
 * `BasePersona.request_permission`). This module renders that block as buttons
 * in the message preamble and, on click, sends the user's decision back to the
 * server by emitting a `permission_response` Jupyter Event (client -> server,
 * `POST /api/events`) — the same events plane the persona-manager already uses
 * to stream state to the frontend, just in the opposite direction.
 */
import React from 'react';
import { MessagePreambleProps } from '@jupyter/chat';
import { Event } from '@jupyterlab/services';

export const PERMISSION_RESPONSE_EVENT_SCHEMA_ID =
  'https://schema.jupyter.org/jupyter_ai_persona_manager/permission_response/v1';

/**
 * Emit the user's permission decision back to the server over the events plane.
 * `optionId === null` cancels/denies without a selection.
 */
export async function submitPermissionDecision(
  events: Event.IManager,
  block: {
    room_id: string;
    persona_id: string;
    request_id: string;
  },
  optionId: string | null
): Promise<void> {
  await events.emit({
    schema_id: PERMISSION_RESPONSE_EVENT_SCHEMA_ID,
    version: '1',
    data: {
      room_id: block.room_id,
      persona_id: block.persona_id,
      request_id: block.request_id,
      option_id: optionId
    }
  });
}

/**
 * Build the message-preamble component that renders permission requests. The
 * event manager is injected by the plugin (preamble props don't carry it).
 */
export function createPermissionPreamble(
  events: Event.IManager
): (props: MessagePreambleProps) => JSX.Element | null {
  return function PermissionPreamble(
    props: MessagePreambleProps
  ): JSX.Element | null {
    const block = props.message.metadata?.permission_request;
    if (!block) {
      return null;
    }
    return <PermissionRequestView events={events} block={block} />;
  };
}

function PermissionRequestView({
  events,
  block
}: {
  events: Event.IManager;
  block: NonNullable<
    MessagePreambleProps['message']['metadata']
  >['permission_request'];
}): JSX.Element | null {
  const [submitting, setSubmitting] = React.useState(false);
  if (!block) {
    return null;
  }

  const resolved = block.status === 'resolved';
  const selected = block.options.find(
    o => o.option_id === block.selected_option_id
  );

  const handleClick = async (optionId: string) => {
    setSubmitting(true);
    try {
      await submitPermissionDecision(events, block, optionId);
    } catch (err) {
      console.error('Failed to submit permission decision:', err);
      setSubmitting(false);
    }
  };

  return (
    <div className="jp-jupyter-ai-permission-request">
      <div className="jp-jupyter-ai-permission-title">{block.title}</div>
      {block.detail ? (
        <div className="jp-jupyter-ai-permission-detail">{block.detail}</div>
      ) : null}
      {resolved ? (
        selected ? (
          <div className="jp-jupyter-ai-permission-resolved">
            — {selected.name}
          </div>
        ) : (
          <div className="jp-jupyter-ai-permission-resolved">— Cancelled</div>
        )
      ) : (
        <div className="jp-jupyter-ai-permission-buttons">
          <span>Allow?</span>
          {block.options.map(opt => (
            <button
              key={opt.option_id}
              className={
                'jp-jupyter-ai-permission-btn' +
                (opt.kind
                  ? ` jp-jupyter-ai-permission-btn-${opt.kind.replace(/_/g, '-')}`
                  : '')
              }
              onClick={() => handleClick(opt.option_id)}
              disabled={submitting}
              title={opt.kind ?? undefined}
            >
              {opt.name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
