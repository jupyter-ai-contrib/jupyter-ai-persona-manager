import React from 'react';
import { PathExt } from '@jupyterlab/coreutils';
import { structuredPatch } from 'diff';

/** A single file diff, as carried in tool-call / permission metadata. */
export interface IDiff {
  path: string;
  new_text: string;
  old_text?: string | null;
}

/** Maximum number of diff lines shown before truncation. */
const MAX_DIFF_LINES = 20;

interface IDiffLineInfo {
  cls: string;
  prefix: string;
  text: string;
  key: string;
}

/**
 * Renders a single file diff block: filename header, line-level highlighting,
 * and click-to-expand truncation. Ported from jupyter-ai-acp-client's DiffView
 * so the generic persona-manager renderer has the same diff fidelity.
 */
function DiffBlock({
  diff,
  onOpenFile,
  toDisplayPath,
  pendingPermission
}: {
  diff: IDiff;
  onOpenFile?: (path: string) => void;
  toDisplayPath?: (path: string) => string;
  pendingPermission?: boolean;
}): JSX.Element {
  const patch = structuredPatch(
    diff.path,
    diff.path,
    diff.old_text ?? '',
    diff.new_text,
    undefined,
    undefined,
    { context: Infinity }
  );
  const displayPath = toDisplayPath
    ? toDisplayPath(diff.path)
    : PathExt.basename(diff.path);
  const isOutsideRoot = displayPath.startsWith('/');
  const isClickable =
    !!onOpenFile &&
    !isOutsideRoot &&
    !(pendingPermission && (diff.old_text ?? undefined) === undefined);
  const [expanded, setExpanded] = React.useState(false);

  const allLines: IDiffLineInfo[] = [];
  for (const hunk of patch.hunks) {
    hunk.lines
      .filter(line => !line.startsWith('\\'))
      .forEach((line, j) => {
        const prefix = line[0];
        const text = line.slice(1);
        const isAdded = prefix === '+';
        const isRemoved = prefix === '-';
        allLines.push({
          cls: isAdded
            ? 'jp-jai-diff-added'
            : isRemoved
              ? 'jp-jai-diff-removed'
              : 'jp-jai-diff-context',
          prefix,
          text,
          key: `${hunk.oldStart}-${j}`
        });
      });
  }

  const canTruncate = allLines.length > MAX_DIFF_LINES;
  const visible =
    canTruncate && !expanded ? allLines.slice(0, MAX_DIFF_LINES) : allLines;
  const hiddenCount = allLines.length - MAX_DIFF_LINES;

  return (
    <div className="jp-jai-diff-block">
      <div
        className={
          'jp-jai-diff-header' +
          (isClickable ? ' jp-jai-diff-header-clickable' : '')
        }
        onClick={isClickable ? () => onOpenFile!(diff.path) : undefined}
        title={diff.path}
      >
        {displayPath}
      </div>
      <div className="jp-jai-diff-content">
        {visible.map((line: IDiffLineInfo) => (
          <div key={line.key} className={`jp-jai-diff-line ${line.cls}`}>
            <span className="jp-jai-diff-line-text">
              {line.prefix} {line.text}
            </span>
          </div>
        ))}
        {canTruncate && !expanded && (
          <div className="jp-jai-diff-toggle" onClick={() => setExpanded(true)}>
            ... {hiddenCount} more lines
          </div>
        )}
        {canTruncate && expanded && (
          <div
            className="jp-jai-diff-toggle"
            onClick={() => setExpanded(false)}
          >
            show less
          </div>
        )}
      </div>
    </div>
  );
}

/** Renders one or more file diffs. */
export function DiffView({
  diffs,
  onOpenFile,
  toDisplayPath,
  pendingPermission
}: {
  diffs: IDiff[];
  onOpenFile?: (path: string) => void;
  toDisplayPath?: (path: string) => string;
  pendingPermission?: boolean;
}): JSX.Element {
  return (
    <div className="jp-jai-diff-container">
      {diffs.map((d, i) => (
        <DiffBlock
          key={i}
          diff={d}
          onOpenFile={onOpenFile}
          toDisplayPath={toDisplayPath}
          pendingPermission={pendingPermission}
        />
      ))}
    </div>
  );
}
