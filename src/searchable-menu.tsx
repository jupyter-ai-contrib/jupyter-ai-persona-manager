import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState
} from 'react';
import { ClickAwayListener, Paper, Popper, TextField } from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import { fuzzyFilter, fuzzyMatch } from './fuzzy';

const MENU_CLASS = 'jp-jai-controlMenu';
const SEARCH_CLASS = 'jp-jai-searchMenu';

/**
 * One selectable row in a searchable menu. `value` is what the caller commits;
 * `label` is shown and searched. `secondary` is optional supporting text (e.g.
 * a model description). `icon` renders before the label (e.g. a persona avatar).
 */
export type SearchableOption = {
  value: string | null;
  label: string;
  secondary?: string | null;
  icon?: React.ReactNode;
  /** The search text, if it should differ from `label`. Defaults to `label`. */
  searchText?: string;
};

/** How a commit should move focus afterwards. */
export type CommitMode = 'stay' | 'next' | 'prev';

/**
 * The focusable controls in the chat input, in tab order, that a menu can
 * advance to. Restricting to the input container keeps `Tab` moving through the
 * toolbar (persona picker → model → … → send) and out to the send button, not
 * to unrelated page chrome.
 */
function tabbableControls(from: HTMLElement): HTMLElement[] {
  const container = from.closest('.jp-chat-input-container');
  if (!container) {
    return [];
  }
  const candidates = Array.from(
    container.querySelectorAll<HTMLElement>(
      'button, input, textarea, select, a[href], [tabindex]'
    )
  );
  return candidates.filter(el => {
    if (el.getAttribute('tabindex') === '-1') {
      return false;
    }
    if (
      el.hasAttribute('disabled') ||
      el.getAttribute('aria-disabled') === 'true'
    ) {
      return false;
    }
    // Skip the aria-hidden, `inert` measurement copy of the controls row.
    if (el.closest('[inert]') || el.closest('[aria-hidden="true"]')) {
      return false;
    }
    // Skip elements not laid out (display:none, detached).
    return el.offsetParent !== null || el === document.activeElement;
  });
}

/**
 * Move focus to the control adjacent to `from` (the just-closed menu's trigger)
 * in the input's tab order. `dir` is +1 for the next control, -1 for the
 * previous. Deferred to the next frame so any controls that appear or disappear
 * as a result of the commit (e.g. selecting a persona reveals its model control)
 * are in the DOM before we compute adjacency.
 */
export function focusAdjacentControl(
  from: HTMLElement | null,
  dir: 1 | -1
): void {
  if (!from) {
    return;
  }
  requestAnimationFrame(() => {
    const controls = tabbableControls(from);
    const index = controls.indexOf(from);
    if (index === -1) {
      return;
    }
    controls[index + dir]?.focus();
  });
}

/**
 * State + handlers for a control that opens a `SearchableMenu`. The menu opens
 * when the trigger gains focus (Tab into it) or is clicked, and closes on
 * commit or cancel. Committing with `Tab` advances focus to the neighbouring
 * control; committing with `Enter`/click, or cancelling with `Escape`, returns
 * focus to the trigger without reopening (a plain `focus()` would retrigger
 * open-on-focus, so a one-shot suppression flag guards it).
 */
export function useSearchableTrigger(): {
  open: boolean;
  triggerRef: React.RefObject<HTMLButtonElement>;
  triggerProps: {
    ref: React.RefObject<HTMLButtonElement>;
    onFocus: () => void;
    onClick: () => void;
    onKeyDown: (event: React.KeyboardEvent) => void;
    'aria-haspopup': 'listbox';
    'aria-expanded': boolean;
  };
  choose: (mode: CommitMode) => void;
  close: () => void;
  cancel: () => void;
  clickAway: () => void;
} {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const suppressOpen = useRef(false);

  const returnFocusNoReopen = useCallback(() => {
    suppressOpen.current = true;
    triggerRef.current?.focus();
    // Safety net: clear the flag next frame in case focus() fired no `focus`
    // event (e.g. the element was already focused).
    requestAnimationFrame(() => {
      suppressOpen.current = false;
    });
  }, []);

  const onFocus = useCallback(() => {
    if (suppressOpen.current) {
      suppressOpen.current = false;
      return;
    }
    setOpen(true);
  }, []);

  const onClick = useCallback(() => setOpen(true), []);

  const onKeyDown = useCallback((event: React.KeyboardEvent) => {
    // Reopen a closed control from the keyboard, matching native <select> /
    // combobox behaviour. (While open, focus is in the menu's search field, so
    // this only fires when closed.)
    if (['Enter', ' ', 'ArrowDown', 'ArrowUp'].includes(event.key)) {
      event.preventDefault();
      setOpen(true);
    }
  }, []);

  const close = useCallback(() => setOpen(false), []);

  const cancel = useCallback(() => {
    setOpen(false);
    returnFocusNoReopen();
  }, [returnFocusNoReopen]);

  const clickAway = useCallback(() => setOpen(false), []);

  const choose = useCallback(
    (mode: CommitMode) => {
      setOpen(false);
      if (mode === 'stay') {
        returnFocusNoReopen();
      } else {
        focusAdjacentControl(triggerRef.current, mode === 'next' ? 1 : -1);
      }
    },
    [returnFocusNoReopen]
  );

  return {
    open,
    triggerRef,
    triggerProps: {
      ref: triggerRef,
      onFocus,
      onClick,
      onKeyDown,
      'aria-haspopup': 'listbox',
      'aria-expanded': open
    },
    choose,
    close,
    cancel,
    clickAway
  };
}

/**
 * Embolden the fuzzy-matched characters of `label` given the current `query`,
 * so a searcher sees why each option matched. With an empty query the label is
 * returned unchanged.
 */
function highlight(label: string, query: string): React.ReactNode {
  const { matched, indices } = fuzzyMatch(label, query);
  if (!matched || indices.length === 0) {
    return label;
  }
  const set = new Set(indices);
  const parts: React.ReactNode[] = [];
  let run = '';
  let runBold = false;
  const flush = (key: number) => {
    if (!run) {
      return;
    }
    parts.push(runBold ? <strong key={key}>{run}</strong> : run);
    run = '';
  };
  for (let i = 0; i < label.length; i++) {
    const bold = set.has(i);
    if (bold !== runBold) {
      flush(i);
      runBold = bold;
    }
    run += label[i];
  }
  flush(label.length);
  return <>{parts}</>;
}

/**
 * The index a searchable menu should highlight for a given `query` result over
 * `options`: the option matching `selectedValue` if it's still in the list,
 * otherwise the first option (the best fuzzy match once filtered). Returns -1
 * for an empty list. Pure, so the seeding rule is unit-testable.
 */
export function seedHighlight(
  options: SearchableOption[],
  selectedValue: string | null
): number {
  if (options.length === 0) {
    return -1;
  }
  const current = options.findIndex(o => o.value === selectedValue);
  return current >= 0 ? current : 0;
}

/**
 * A keyboard-first, searchable dropdown for the persona controls. It opens
 * anchored to a trigger button and takes focus into its search field, so the
 * flow is: open → type to filter (fuzzy) and/or arrow to move the highlight →
 * commit. Committing happens on Enter, on click, or on Tab — and Tab also
 * advances focus to the next control, which is what makes tabbing through the
 * toolbar confirm each control in turn and eventually land on the send button.
 * Escape closes without committing.
 *
 * Adapted from jupyter-ai-jupyternaut's `SimpleAutocomplete`, specialized for
 * committing option values (not free text) and for Tab-to-confirm-and-advance.
 */
export function SearchableMenu(props: {
  anchorEl: HTMLElement | null;
  open: boolean;
  options: SearchableOption[];
  /** The currently committed value, used to seed the highlight on open. */
  selectedValue: string | null;
  /** Accessible label for the search field. */
  searchLabel: string;
  /** Apply a chosen value and move focus per `mode`. */
  onChoose: (value: string | null, mode: CommitMode) => void;
  /** Close without committing, returning focus to the trigger. */
  onCancel: () => void;
  /** Close without committing when focus/click leaves the menu. */
  onClickAway: () => void;
}): JSX.Element | null {
  const {
    anchorEl,
    open,
    options,
    selectedValue,
    searchLabel,
    onChoose,
    onCancel,
    onClickAway
  } = props;
  const [query, setQuery] = useState('');
  const [highlighted, setHighlighted] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const filtered = useMemo(
    () => fuzzyFilter(options, query, o => o.searchText ?? o.label),
    [options, query]
  );

  // On open, clear the query and seed the highlight to the committed value (or
  // the first option), then focus the search field so typing filters and arrows
  // move the highlight right away. Intentionally keyed only on `open`: this runs
  // when the menu opens, not on every options/selectedValue change (typing then
  // manages the highlight itself, below).
  useEffect(() => {
    if (!open) {
      return;
    }
    setQuery('');
    setHighlighted(seedHighlight(options, selectedValue));
    const id = requestAnimationFrame(() => inputRef.current?.focus());
    return () => cancelAnimationFrame(id);
  }, [open]);

  // Keep the highlighted row scrolled into view as it moves.
  useLayoutEffect(() => {
    if (!open) {
      return;
    }
    const row = listRef.current?.children[highlighted] as
      | HTMLElement
      | undefined;
    row?.scrollIntoView({ block: 'nearest' });
  }, [highlighted, open]);

  // The value a commit would apply: the highlighted row, or the committed value
  // when nothing is highlighted (e.g. the filter emptied the list).
  const currentValue = (): string | null =>
    filtered.length && highlighted >= 0 && highlighted < filtered.length
      ? filtered[highlighted].value
      : selectedValue;

  const handleKeyDown = (event: React.KeyboardEvent) => {
    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        setHighlighted(prev =>
          filtered.length ? (prev + 1) % filtered.length : 0
        );
        break;
      case 'ArrowUp':
        event.preventDefault();
        setHighlighted(prev =>
          filtered.length ? (prev - 1 + filtered.length) % filtered.length : 0
        );
        break;
      case 'Enter':
        event.preventDefault();
        if (filtered.length && highlighted >= 0) {
          onChoose(filtered[highlighted].value, 'stay');
        }
        break;
      case 'Tab':
        event.preventDefault();
        onChoose(currentValue(), event.shiftKey ? 'prev' : 'next');
        break;
      case 'Escape':
        event.preventDefault();
        event.stopPropagation();
        onCancel();
        break;
      default:
        break;
    }
  };

  if (!open) {
    return null;
  }

  return (
    <Popper
      open={open}
      anchorEl={anchorEl}
      placement="top-start"
      className={`${SEARCH_CLASS}-popper`}
      modifiers={[{ name: 'offset', options: { offset: [0, 4] } }]}
    >
      <ClickAwayListener onClickAway={onClickAway}>
        <Paper className={`${MENU_CLASS}-paper ${SEARCH_CLASS}-paper`}>
          <div className={`${SEARCH_CLASS}-search`}>
            <TextField
              inputRef={inputRef}
              value={query}
              onChange={e => {
                setQuery(e.target.value);
                setHighlighted(0);
              }}
              onKeyDown={handleKeyDown}
              placeholder="Search…"
              variant="standard"
              fullWidth
              size="small"
              inputProps={{
                'aria-label': searchLabel,
                role: 'combobox',
                'aria-expanded': true,
                'aria-controls': `${SEARCH_CLASS}-listbox`
              }}
              InputProps={{ disableUnderline: true }}
            />
          </div>
          <div
            className={`${SEARCH_CLASS}-list`}
            id={`${SEARCH_CLASS}-listbox`}
            role="listbox"
            ref={listRef}
          >
            {filtered.length === 0 ? (
              <div className={`${SEARCH_CLASS}-empty`}>No matches</div>
            ) : (
              filtered.map((option, index) => (
                <div
                  key={option.value ?? '__null__'}
                  role="option"
                  aria-selected={index === highlighted}
                  className={`${SEARCH_CLASS}-option${
                    index === highlighted ? ' jp-mod-highlighted' : ''
                  }`}
                  onMouseEnter={() => setHighlighted(index)}
                  // Keep the search input focused through the click so the
                  // option's onClick fires before any blur handling.
                  onMouseDown={e => e.preventDefault()}
                  onClick={() => onChoose(option.value, 'stay')}
                >
                  {option.icon ? (
                    <span className={`${SEARCH_CLASS}-option-icon`}>
                      {option.icon}
                    </span>
                  ) : null}
                  <span className={`${SEARCH_CLASS}-option-text`}>
                    <span className={`${MENU_CLASS}-name`}>
                      {highlight(option.label, query)}
                    </span>
                    {option.secondary ? (
                      <span className={`${MENU_CLASS}-desc`}>
                        {option.secondary}
                      </span>
                    ) : null}
                  </span>
                  {option.value === selectedValue ? (
                    <CheckIcon
                      className={`${MENU_CLASS}-check`}
                      fontSize="small"
                    />
                  ) : null}
                </div>
              ))
            )}
          </div>
        </Paper>
      </ClickAwayListener>
    </Popper>
  );
}
