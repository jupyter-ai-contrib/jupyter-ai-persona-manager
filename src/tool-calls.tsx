/**
 * Generic renderer for persona tool calls (`BasePersona.report_tool_call`),
 * ported from jupyter-ai-acp-client so the tool-call UI is available to any
 * persona. Renders each tool call in a message's `tool_calls` metadata with a
 * status glyph, an optional detail line, file diffs, and expandable output. A
 * tool call carrying an attached permission renders approve/deny buttons that
 * emit a `permission_response` event (the same decision transport as standalone
 * permission requests).
 */
import React from 'react';
import { MessagePreambleProps } from '@jupyter/chat';
import { Event } from '@jupyterlab/services';
import { PageConfig, PathExt } from '@jupyterlab/coreutils';

import { DiffView, IDiff } from './diff-view';
import {
  createPermissionPreamble,
  submitPermissionDecision
} from './permissions';

type ToolCall = NonNullable<
  NonNullable<MessagePreambleProps['message']['metadata']>['tool_calls']
>[number];

/** Convert an absolute filesystem path to a server-relative path for display. */
function toServerRelativePath(absolutePath: string): string {
  const rootUri = PageConfig.getOption('rootUri');
  const serverRoot = rootUri
    ? new URL(rootUri).pathname
    : PageConfig.getOption('serverRoot');
  if (!serverRoot) {
    return absolutePath;
  }
  const relativePath = PathExt.relative(serverRoot, absolutePath);
  if (relativePath.startsWith('..')) {
    return absolutePath;
  }
  return relativePath;
}

function formatOutput(rawOutput: unknown): string {
  if (typeof rawOutput === 'string') {
    return rawOutput;
  }
  if (Array.isArray(rawOutput) && rawOutput.every(i => (i as any)?.text)) {
    return (rawOutput as any[]).map(i => i.text).join('\n');
  }
  return JSON.stringify(rawOutput, null, 2);
}

function formatToolInput(input: unknown): string {
  if (typeof input === 'string') {
    return input;
  }
  if (typeof input !== 'object' || input === null || Array.isArray(input)) {
    return JSON.stringify(input, null, 2);
  }
  const entries = Object.entries(input as Record<string, unknown>);
  const isFlat = entries.every(
    ([, v]) =>
      typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean'
  );
  if (isFlat) {
    return entries.map(([k, v]) => `${k}: ${v}`).join('\n');
  }
  return JSON.stringify(input, null, 2);
}

/** Pre-permission detail text for a tool call (command, paths, or params). */
function buildPermissionDetail(toolCall: ToolCall): string | null {
  const { kind, title, locations, raw_input } = toolCall;

  if (kind === 'execute') {
    const rawObj =
      typeof raw_input === 'object' && raw_input !== null
        ? (raw_input as Record<string, unknown>)
        : null;
    const cmd =
      rawObj && typeof rawObj.command === 'string'
        ? rawObj.command
        : title
            ?.replace(/^Running:\s*/i, '')
            .replace(/\.\.\.$/, '')
            .trim() || null;
    if (!cmd || cmd === title) {
      return null;
    }
    return '$ ' + cmd;
  }

  if (
    (kind === 'delete' || kind === 'move' || kind === 'read') &&
    locations?.length
  ) {
    return kind === 'move' && locations.length >= 2
      ? toServerRelativePath(locations[0]) +
          '  \u2192  ' +
          toServerRelativePath(locations[1])
      : locations.map(toServerRelativePath).join('\n');
  }

  if (
    raw_input !== null &&
    typeof raw_input === 'object' &&
    !Array.isArray(raw_input)
  ) {
    const obj = raw_input as Record<string, unknown>;
    const purpose =
      typeof obj.__tool_use_purpose === 'string'
        ? obj.__tool_use_purpose
        : null;
    const paramEntries = Object.entries(obj).filter(
      ([k]) => !k.startsWith('__')
    );
    const params =
      paramEntries.length > 0
        ? formatToolInput(Object.fromEntries(paramEntries))
        : null;
    if (purpose && params) {
      return purpose + '\n' + params;
    }
    return purpose || params;
  }

  if (raw_input !== null && raw_input !== undefined) {
    return formatToolInput(raw_input);
  }
  return null;
}

const FILE_KINDS = new Set(['read', 'edit', 'delete', 'move']);
const OUTPUT_KINDS = new Set(['search', 'execute', 'think', 'fetch']);

/** Expandable details for a completed/failed tool call. */
function buildDetailsLines(toolCall: ToolCall): string[] {
  const lines: string[] = [];
  const kind = toolCall.kind ?? undefined;
  if (kind && FILE_KINDS.has(kind) && toolCall.locations?.length) {
    for (const loc of toolCall.locations) {
      lines.push(toServerRelativePath(loc));
    }
  } else if (kind && OUTPUT_KINDS.has(kind) && toolCall.raw_output) {
    lines.push(formatOutput(toolCall.raw_output));
  } else if (toolCall.raw_output && typeof toolCall.raw_output === 'string') {
    lines.push(toolCall.raw_output);
  }
  return lines;
}

/** The user's resolved permission selection, shown after a decision. */
function PermissionLabel({
  toolCall
}: {
  toolCall: ToolCall;
}): JSX.Element | null {
  if (toolCall.permission_status !== 'resolved') {
    return null;
  }
  const selectedName = toolCall.permission_options?.find(
    opt => opt.option_id === toolCall.selected_option_id
  )?.name;
  return (
    <span className="jp-jai-permission-label">
      {' '}
      — {selectedName ?? 'Cancelled'}
    </span>
  );
}

/** Approve/deny buttons for a tool call with a pending permission. */
function PermissionButtons({
  events,
  toolCall
}: {
  events: Event.IManager;
  toolCall: ToolCall;
}): JSX.Element | null {
  const [submitting, setSubmitting] = React.useState(false);
  if (
    !toolCall.permission_options?.length ||
    toolCall.permission_status !== 'pending' ||
    !toolCall.request_id ||
    !toolCall.chat_id ||
    !toolCall.persona_id
  ) {
    return null;
  }
  const handleClick = async (optionId: string) => {
    setSubmitting(true);
    try {
      await submitPermissionDecision(
        events,
        {
          chat_id: toolCall.chat_id!,
          persona_id: toolCall.persona_id!,
          request_id: toolCall.request_id!
        },
        optionId
      );
    } catch (err) {
      console.error('Failed to submit permission decision:', err);
      setSubmitting(false);
    }
  };
  return (
    <div className="jp-jai-permission-buttons">
      <span>Allow?</span>
      {toolCall.permission_options.map(opt => (
        <button
          key={opt.option_id}
          className={
            'jp-jai-permission-btn' +
            (opt.kind
              ? ` jp-jai-permission-btn-${opt.kind.replace(/_/g, '-')}`
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
  );
}

function ToolCallLine({
  events,
  toolCall,
  onOpenFile
}: {
  events: Event.IManager;
  toolCall: ToolCall;
  onOpenFile?: (path: string) => void;
}): JSX.Element {
  const { title, status, kind } = toolCall;
  const displayTitle =
    title ||
    (kind
      ? `${kind.charAt(0).toUpperCase()}${kind.slice(1)}...`
      : 'Working...');
  const selectedOpt = toolCall.permission_options?.find(
    opt => opt.option_id === toolCall.selected_option_id
  );
  const isRejected =
    toolCall.permission_status === 'resolved' &&
    !!selectedOpt?.kind?.includes('reject');
  const hasPendingPermission = toolCall.permission_status === 'pending';
  const isInProgress =
    !isRejected &&
    (status === 'in_progress' || status === 'pending' || hasPendingPermission);
  const isCompleted = status === 'completed';
  const isFailed = status === 'failed' || isRejected;

  const icon = isInProgress
    ? '\u2022'
    : isCompleted
      ? '\u2713'
      : isFailed
        ? '\u2717'
        : '\u2022';
  const effectiveStatus = isRejected ? 'failed' : status || 'in_progress';
  const cssClass = `jp-jai-tool-call jp-jai-tool-call-${effectiveStatus}`;
  const hasDiffs = !!toolCall.diffs?.length;

  // Pending permission with diffs.
  if (hasDiffs && hasPendingPermission) {
    return (
      <div className={cssClass}>
        <details open>
          <summary>
            <span className="jp-jai-tool-call-icon">{icon}</span>{' '}
            <em>{displayTitle}</em>
          </summary>
          <DiffView
            diffs={toolCall.diffs as IDiff[]}
            onOpenFile={kind === 'edit' ? onOpenFile : undefined}
            toDisplayPath={toServerRelativePath}
            pendingPermission
          />
        </details>
        <PermissionButtons events={events} toolCall={toolCall} />
      </div>
    );
  }

  // Pending permission without diffs: show kind-specific detail.
  if (!hasDiffs && hasPendingPermission) {
    const detail = buildPermissionDetail(toolCall);
    if (detail !== null) {
      return (
        <div className={cssClass}>
          <details open>
            <summary>
              <span className="jp-jai-tool-call-icon">{icon}</span>{' '}
              <em>{displayTitle}</em>
            </summary>
            <div className="jp-jai-tool-call-detail">{detail}</div>
          </details>
          <PermissionButtons events={events} toolCall={toolCall} />
        </div>
      );
    }
  }

  const detailsLines =
    !hasDiffs && (isCompleted || isFailed) ? buildDetailsLines(toolCall) : [];
  const hasExpandable = hasDiffs || detailsLines.length > 0;

  if ((isCompleted || isFailed) && hasExpandable) {
    return (
      <details className={cssClass}>
        <summary>
          <span className="jp-jai-tool-call-icon">{icon}</span> {displayTitle}
          <PermissionLabel toolCall={toolCall} />
        </summary>
        {hasDiffs ? (
          <DiffView
            diffs={toolCall.diffs as IDiff[]}
            onOpenFile={onOpenFile}
            toDisplayPath={toServerRelativePath}
          />
        ) : (
          <div className="jp-jai-tool-call-detail">
            {detailsLines.join('\n')}
          </div>
        )}
      </details>
    );
  }

  if (isInProgress) {
    return (
      <div className={cssClass}>
        <span className="jp-jai-tool-call-icon">{icon}</span>{' '}
        <em>{displayTitle}</em>
        <PermissionButtons events={events} toolCall={toolCall} />
      </div>
    );
  }

  return (
    <div className={cssClass}>
      <span className="jp-jai-tool-call-icon">{icon}</span> {displayTitle}
      <PermissionLabel toolCall={toolCall} />
    </div>
  );
}

/** Build the message-preamble component that renders tool calls. */
export function createToolCallsPreamble(
  events: Event.IManager
): (props: MessagePreambleProps) => JSX.Element | null {
  return function ToolCallsPreamble(
    props: MessagePreambleProps
  ): JSX.Element | null {
    const toolCalls = props.message.metadata?.tool_calls;
    if (!toolCalls || toolCalls.length === 0) {
      return null;
    }
    const onOpenFile = (path: string) => {
      (props.model as any).documentManager?.openOrReveal(
        toServerRelativePath(path)
      );
    };
    return (
      <div className="jp-jai-tool-calls">
        {toolCalls.map(tc => (
          <ToolCallLine
            key={tc.tool_call_id}
            events={events}
            toolCall={tc}
            onOpenFile={onOpenFile}
          />
        ))}
      </div>
    );
  };
}

/**
 * Single message-preamble component that renders standalone permission requests
 * (if any) followed by tool calls (if any). Registering one component avoids
 * double-rendering seen when multiple preamble components are registered.
 */
export function createPersonaPreamble(
  events: Event.IManager
): (props: MessagePreambleProps) => JSX.Element | null {
  const permission = createPermissionPreamble(events);
  const toolCalls = createToolCallsPreamble(events);
  return function PersonaPreamble(
    props: MessagePreambleProps
  ): JSX.Element | null {
    const p = permission(props);
    const t = toolCalls(props);
    if (!p && !t) {
      return null;
    }
    return (
      <>
        {p}
        {t}
      </>
    );
  };
}
