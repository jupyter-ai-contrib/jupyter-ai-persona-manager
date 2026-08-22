import React, {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState
} from 'react';
import {
  Button,
  ListItemText,
  ListSubheader,
  Menu,
  MenuItem,
  Popover,
  Skeleton,
  TextField
} from '@mui/material';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import CheckIcon from '@mui/icons-material/Check';
import MoreHorizIcon from '@mui/icons-material/MoreHoriz';
import { PageConfig } from '@jupyterlab/coreutils';
import { InputToolbarRegistry } from '@jupyter/chat';
import {
  EMPTY_USAGE,
  PersonaOption,
  SettingConfiguration,
  Usage
} from './awareness';
import { PersonaSessionRegistry, PersonaSessionState } from './persona-events';
import {
  PersonaSettings,
  buildMessageMetadata,
  emptyPersonaSettings
} from './metadata';
import { IPersonaControlRegistry } from './persona-control-registry';

const SELECTOR_CLASS = 'jp-jai-personaControls';
const MENU_CLASS = 'jp-jai-controlMenu';
const USAGE_CLASS = 'jp-jai-usage';
const NO_ONE_LABEL = 'No one';

// Stable control ID for the model selector (setting IDs are used verbatim).
const MODEL_CONTROL_ID = '__model__';

// Context-fill fractions at which the chip starts demanding attention: the
// ring and percent turn warn, then error, colored.
const USAGE_WARN_AT = 0.7;
const USAGE_ERROR_AT = 0.9;

/**
 * The chat's default persona ID, advertised by the persona-manager server
 * extension via PageConfig. Used as the initial selection for a chat where the
 * user hasn't picked a persona yet. Empty string if none is configured.
 */
const DEFAULT_PERSONA_ID =
  PageConfig.getOption('jupyter_ai_default_persona') || null;

// Width (px) reserved for the overflow ("...") button when not every control
// fits inline.
const OVERFLOW_BTN_WIDTH = 36;

const menuAnchorProps = {
  anchorOrigin: { vertical: 'top', horizontal: 'left' } as const,
  transformOrigin: { vertical: 'bottom', horizontal: 'left' } as const,
  PaperProps: { className: `${MENU_CLASS}-paper` }
};

/**
 * A UI control for one control (the model, a model setting, or a general
 * setting). It carries the persona's current value (from awareness) and the
 * user's per-message selection (null = use the persona's default).
 */
export type Control = {
  id: string;
  kind: 'model' | 'model_setting' | 'setting';
  label: string;
  /** The persona's current value, from awareness. Null when on its default. */
  current: string | null;
  /** The user's selection for this message. Null means "use the default". */
  selection: string | null;
  options: { id: string; name: string; description: string | null }[];
};

/**
 * Convert a persona's awareness `SettingConfiguration` into a `Control` of the
 * given kind, seeding the user selection from the current per-persona
 * selection (defaulting to null = default).
 */
function settingToControl(
  setting: SettingConfiguration,
  kind: 'model_setting' | 'setting',
  selection: string | null
): Control {
  return {
    id: setting.id,
    kind,
    label: setting.name ?? setting.id,
    current: setting.current,
    selection,
    options: setting.options.map(o => ({
      id: o.id,
      name: o.name ?? o.id,
      description: o.description
    }))
  };
}

/**
 * Build the list of controls to render for a persona: the model control (when the
 * persona advertises models), its model settings, then its general settings.
 * The user's current selection seeds each control's `selection`.
 */
export function buildControls(
  persona: PersonaSessionState | null,
  settings: PersonaSettings
): Control[] {
  if (!persona) {
    return [];
  }
  const controls: Control[] = [];
  if (persona.model.options.length) {
    controls.push({
      id: MODEL_CONTROL_ID,
      kind: 'model',
      label: 'Model',
      current: persona.model.current,
      selection: settings.modelId,
      options: persona.model.options.map(o => ({
        id: o.id,
        name: o.name ?? o.id,
        description: o.description
      }))
    });
  }
  for (const setting of persona.model.settings) {
    controls.push(
      settingToControl(
        setting,
        'model_setting',
        settings.modelSettings[setting.id] ?? null
      )
    );
  }
  for (const setting of persona.settings) {
    controls.push(
      settingToControl(
        setting,
        'setting',
        settings.settings[setting.id] ?? null
      )
    );
  }
  return controls;
}

/**
 * Decide which persona list the toolbar should display: a freshly read empty
 * list is treated as a transient blip (e.g. a Yjs awareness sync hiccup
 * during a persona reload) rather than "no personas", so the toolbar keeps
 * showing the previous list instead of unmounting - `PersonaControls` hides
 * itself entirely when `personas.length === 0`, so accepting every empty
 * read verbatim flashes the whole toolbar (persona name, model picker,
 * everything) to nothing and back on each blip. A genuinely personas-less
 * chat never reads a non-empty list in the first place, so this doesn't mask
 * that case - it only guards against reverting an already-populated list.
 */
export function reconcilePersonas(
  previous: PersonaOption[],
  next: PersonaOption[]
): PersonaOption[] {
  return next.length ? next : previous;
}

/**
 * Decide which selected-persona state the toolbar should track: a fresh
 * null read is treated as a transient blip (the same class of awareness
 * sync hiccup `reconcilePersonas` guards against, e.g. during a persona
 * reload) rather than "nothing selected", so the toolbar keeps the last
 * known state instead of dropping it. This one is not just cosmetic:
 * `buildControls` returns `[]` for a null persona state, so `controls`
 * reads empty and `PersonaControls` conditionally unmounts `ControlsRow`
 * entirely - destroying any open `ControlMenu`'s own state (its search
 * query, its open popover) mid-interaction, not merely blanking a label.
 *
 * Only for the *recurring* awareness-driven read, not the initial one: the
 * initial read (on mount, or when `selectedId` itself changes to a
 * different persona) must apply unconditionally, including a genuine null
 * result, or switching personas could show the previous persona's stale
 * state under the new selection.
 */
export function reconcilePersonaState(
  previous: PersonaSessionState | null,
  next: PersonaSessionState | null
): PersonaSessionState | null {
  return next ?? previous;
}

/**
 * Decide how to reconcile the current selection with a freshly read persona
 * list: the new selection to apply, or `undefined` to keep the current one.
 *
 * A selection pointing at a persona the chat doesn't have resolves to the sole
 * persona (as a convenience) or to no one. The sole-persona convenience also
 * seeds an empty initial selection, but only until the user has made an
 * explicit choice: after that their choice, including "No one" (`null`),
 * sticks.
 */
export function reconcileSelection(
  personas: PersonaOption[],
  selectedId: string | null,
  userPicked: boolean
): string | null | undefined {
  if (!personas.length) {
    return undefined;
  }
  if (selectedId && personas.some(p => p.id === selectedId)) {
    return undefined;
  }
  if (personas.length === 1 && !userPicked) {
    return personas[0].id;
  }
  // An invalid selection clears to "No one"; an already-empty one stands.
  return selectedId ? null : undefined;
}

/**
 * Whether the toolbar, knowing no personas yet, should show the loading
 * placeholder rather than nothing: only while the manager's slot can still
 * resolve (awareness exists, resolution hasn't failed) and the manager or its
 * first persona-list read is still pending. Without an awareness channel there
 * is nothing to wait on, so nothing renders.
 */
export function showLoadingPlaceholder(
  hasAwareness: boolean,
  managerResolved: boolean,
  managerFailed: boolean,
  listRead: boolean
): boolean {
  if (!hasAwareness || managerFailed) {
    return false;
  }
  return !managerResolved || !listRead;
}

/**
 * Fold a changed control value into the user's `PersonaSettings`, keyed by the
 * control's kind. A null value resets that control to the persona's default.
 */
export function applyControlChange(
  settings: PersonaSettings,
  control: Control,
  value: string | null
): PersonaSettings {
  const next: PersonaSettings = {
    modelId: settings.modelId,
    modelSettings: { ...settings.modelSettings },
    settings: { ...settings.settings }
  };
  if (control.kind === 'model') {
    next.modelId = value;
  } else if (control.kind === 'model_setting') {
    next.modelSettings[control.id] = value;
  } else {
    next.settings[control.id] = value;
  }
  return next;
}

/**
 * The value a control currently reflects: the user's selection if they picked
 * one, otherwise the persona's current value (the default).
 */
function effectiveValue(control: Control): string | null {
  return control.selection ?? control.current;
}

/**
 * A small round avatar image, or a same-sized spacer to keep labels aligned.
 */
function Avatar(props: { url: string | null | undefined }): JSX.Element {
  if (!props.url) {
    return <span className={`${SELECTOR_CLASS}-avatar-spacer`} />;
  }
  return <img className={`${SELECTOR_CLASS}-avatar`} src={props.url} alt="" />;
}

/**
 * Placeholder for the toolbar while the persona list is being resolved over
 * awareness: a circle where the picker's avatar sits and a bar where its label
 * sits, so a slow network reads as loading rather than a missing toolbar.
 */
function LoadingPlaceholder(): JSX.Element {
  return (
    <div
      className={`${SELECTOR_CLASS}-group ${SELECTOR_CLASS}-skeleton`}
      title="Loading personas"
    >
      <Skeleton variant="circular" width={18} height={18} />
      <Skeleton variant="rounded" width={90} height={12} />
    </div>
  );
}

/**
 * The label shown on a control's button: the name of its effective value, or the
 * control's own label when nothing resolves (no options, no current value).
 */
function currentControlLabel(control: Control): string {
  const value = effectiveValue(control);
  const option = control.options.find(o => o.id === value);
  return option?.name ?? value ?? control.label;
}

/**
 * One choice row in a control dropdown. Shows the choice name, and a secondary
 * description only when it adds information (some agents repeat the name as the
 * description, which is just noise). The full description is available on hover.
 */
function ChoiceMenuItem(props: {
  primary: string;
  description: string | null;
  selected: boolean;
  onSelect: () => void;
  /** Extra class name, used by ControlMenu to show keyboard focus. */
  className?: string;
}): JSX.Element {
  // MenuList clones the row it picks for initial focus with extra props
  // (tabIndex, autoFocus); forward them to the MenuItem, or no row is ever
  // focused and the menu's arrow-key and type-ahead handling never engages.
  // Shared by OverflowControlsMenu (still a real Menu/MenuList, still
  // depends on this) and ControlMenu (no longer rendered inside a
  // MenuList - see the comment there - so this spread is a no-op for it,
  // and it leaves `role` at MenuItem's own default rather than overriding
  // it, to avoid disturbing OverflowControlsMenu's ARIA contract).
  const {
    primary,
    description: rawDescription,
    selected,
    onSelect,
    ...menuItemProps
  } = props;
  const description =
    rawDescription &&
    rawDescription.trim().toLowerCase() !== primary.trim().toLowerCase()
      ? rawDescription
      : null;
  return (
    <MenuItem
      {...menuItemProps}
      selected={selected}
      onClick={onSelect}
      title={description ?? undefined}
    >
      <ListItemText
        primary={primary}
        secondary={description}
        classes={{
          primary: `${MENU_CLASS}-name`,
          secondary: `${MENU_CLASS}-desc`
        }}
      />
      {selected ? (
        <CheckIcon className={`${MENU_CLASS}-check`} fontSize="small" />
      ) : null}
    </MenuItem>
  );
}

/** One selectable row in a control's dropdown: an option, or the leading
 * "Default" row (`id: null`). */
type Choice = {
  id: string | null;
  primary: string;
  description: string | null;
};

/**
 * Filter a control's choices by a search query: a case-insensitive substring
 * match against each choice's name. An empty (or whitespace-only) query
 * matches everything, so a freshly opened menu shows the full list. The
 * leading "Default" choice (`id: null`) is excluded from filtering entirely -
 * always kept, and always first - since it's a fixed, load-bearing option
 * (what "no explicit selection" resolves to), not just another item to
 * search among.
 */
export function filterChoices(choices: Choice[], query: string): Choice[] {
  const defaultChoice = choices.filter(c => c.id === null);
  const rest = choices.filter(c => c.id !== null);
  const q = query.trim().toLowerCase();
  const filteredRest = q
    ? rest.filter(c => c.primary.toLowerCase().includes(q))
    : rest;
  return [...defaultChoice, ...filteredRest];
}

/**
 * The "Default" row shown at the top of every control. Selecting it sets the
 * user's value to null, i.e. "use the persona's current value". Its label shows
 * that current value so the user sees what the default points to.
 */
function defaultChoiceLabel(control: Control): string {
  const current = control.options.find(o => o.id === control.current);
  const name = current?.name ?? control.current;
  return name ? `Default (${name})` : 'Default';
}

/**
 * The uppercase group label used in control menus: it titles a control's own
 * dropdown and labels each control's section in the overflow menu. Rendered
 * with MUI's `ListSubheader`, which has no tabindex, so arrow-key focus skips
 * it and the menu stays keyboard-navigable. Sticky: when the menu scrolls, the
 * label pins to the top, so the group the visible rows belong to stays
 * readable; in the overflow menu the next section's label paints over it on
 * arrival.
 */
function ControlMenuSubheader(props: {
  label: string;
  id?: string;
}): JSX.Element {
  return (
    <ListSubheader id={props.id} className={`${MENU_CLASS}-subheader`}>
      {props.label}
    </ListSubheader>
  );
}

// MUI's MenuList skips initial focus for children whose type carries this
// static; ListSubheader's own copy is hidden behind the wrapper.
ControlMenuSubheader.muiSkipListHighlight = true;

/**
 * A searchable dropdown for a control, titled with the control's label. The
 * first choice row is always "Default" (selection = null, never filtered
 * out); the rest are the persona's advertised options (selection = that
 * option's id), filtered by the search box as the user types. Exported for
 * tests.
 *
 * Built on `Popover` rather than `Menu`/`MenuList`: `MenuList` owns its own
 * keyboard handling (arrow keys, type-ahead-by-letter) which would fight a
 * text input for keystrokes, so with a search box in the picture this
 * component manages its own keyboard navigation (`focusedIndex`) instead of
 * relying on `MenuList`'s.
 */
export function ControlMenu(props: {
  control: Control;
  onSelect: (value: string | null) => void;
}): JSX.Element {
  const { control, onSelect } = props;
  const [anchor, setAnchor] = useState<HTMLElement | null>(null);
  const [query, setQuery] = useState('');
  // Index into the *filtered* choice list (Default row included at 0), for
  // arrow-key navigation. Reset whenever the query changes, since the
  // previously-focused row may no longer be in the filtered list.
  const [focusedIndex, setFocusedIndex] = useState(0);
  // The heading names the menu for assistive tech: the subheader itself is a
  // roleless, never-focused list row, so without this wiring the popup has no
  // accessible name at all.
  const headingId = useId();

  const choices: Choice[] = [
    { id: null, primary: defaultChoiceLabel(control), description: null },
    ...control.options.map(o => ({
      id: o.id,
      primary: o.name,
      description: o.description
    }))
  ];
  const filtered = filterChoices(choices, query);

  const close = (): void => {
    setAnchor(null);
    setQuery('');
    setFocusedIndex(0);
  };

  const select = (id: string | null): void => {
    close();
    onSelect(id);
  };

  const handleQueryChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ): void => {
    setQuery(event.target.value);
    setFocusedIndex(0);
  };

  const handleKeyDown = (event: React.KeyboardEvent): void => {
    if (!filtered.length) {
      if (event.key === 'Escape') {
        close();
      }
      return;
    }
    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        setFocusedIndex(i => (i + 1) % filtered.length);
        break;
      case 'ArrowUp':
        event.preventDefault();
        setFocusedIndex(i => (i - 1 + filtered.length) % filtered.length);
        break;
      case 'Enter': {
        event.preventDefault();
        const choice = filtered[focusedIndex];
        if (choice) {
          select(choice.id);
        }
        break;
      }
      case 'Escape':
        event.preventDefault();
        close();
        break;
    }
  };

  // Open focused on the currently selected row (falling back to Default,
  // index 0, when the selection is stale/absent) - computed fresh on each
  // open rather than via a lazy useState initializer, since this component
  // instance persists across opens/closes (only `anchor` toggles), so a
  // one-time initializer would never re-run for a later open.
  const openMenu = (event: React.MouseEvent<HTMLElement>): void => {
    setAnchor(event.currentTarget);
    const idx = choices.findIndex(c => c.id === control.selection);
    setFocusedIndex(idx >= 0 ? idx : 0);
  };

  return (
    <>
      <Button
        className={`${SELECTOR_CLASS} ${SELECTOR_CLASS}-control-btn`}
        size="small"
        variant="text"
        disableRipple
        endIcon={<ArrowDropDownIcon className={`${SELECTOR_CLASS}-arrow`} />}
        onClick={openMenu}
        title={control.label}
      >
        <span className={`${SELECTOR_CLASS}-control-value`}>
          {currentControlLabel(control)}
        </span>
      </Button>
      <Popover
        anchorEl={anchor}
        open={!!anchor}
        onClose={close}
        {...menuAnchorProps}
      >
        <ControlMenuSubheader id={headingId} label={control.label} />
        {control.options.length > 0 ? (
          <TextField
            autoFocus
            fullWidth
            size="small"
            variant="standard"
            placeholder={`Search ${control.label.toLowerCase()}`}
            value={query}
            onChange={handleQueryChange}
            onKeyDown={handleKeyDown}
            className={`${MENU_CLASS}-search-input`}
          />
        ) : null}
        {/* role="group", not "listbox": MenuItem's own default role
            (menuitem) would mismatch a listbox's expected "option"
            children, and changing MenuItem's role risks disturbing
            OverflowControlsMenu, which shares ChoiceMenuItem and still
            relies on the real Menu/MenuList's own ARIA menu semantics. */}
        <div
          role="group"
          aria-labelledby={headingId}
          className={`${MENU_CLASS}-search-list`}
        >
          {filtered.length ? (
            filtered.map((choice, index) => (
              <ChoiceMenuItem
                key={choice.id ?? '__default__'}
                primary={choice.primary}
                description={choice.description}
                selected={control.selection === choice.id}
                onSelect={() => select(choice.id)}
                className={
                  index === focusedIndex
                    ? `${MENU_CLASS}-kbd-focused`
                    : undefined
                }
              />
            ))
          ) : (
            <MenuItem disabled>No matches</MenuItem>
          )}
        </div>
      </Popover>
    </>
  );
}

/**
 * The overflow popover: controls that did not fit inline, shown as a single flat
 * menu (no nested dropdowns). Each control renders as a group label followed by
 * its Default row and choices. Exported for tests.
 */
export function OverflowControlsMenu(props: {
  controls: Control[];
  anchor: HTMLElement | null;
  onClose: () => void;
  onChange: (control: Control, value: string | null) => void;
}): JSX.Element {
  const { controls, anchor, onClose, onChange } = props;
  return (
    <Menu
      anchorEl={anchor}
      open={!!anchor}
      onClose={onClose}
      MenuListProps={{ 'aria-label': 'More controls' }}
      {...menuAnchorProps}
    >
      {controls.flatMap(control => [
        <ControlMenuSubheader
          key={`${control.id}-label`}
          label={control.label}
        />,
        <ChoiceMenuItem
          key={`${control.id}-default`}
          primary={defaultChoiceLabel(control)}
          description={null}
          selected={control.selection === null}
          onSelect={() => {
            onClose();
            onChange(control, null);
          }}
        />,
        ...control.options.map(option => (
          <ChoiceMenuItem
            key={`${control.id}-${option.id}`}
            primary={option.name}
            description={option.description}
            selected={control.selection === option.id}
            onSelect={() => {
              onClose();
              onChange(control, option.id);
            }}
          />
        ))
      ])}
    </Menu>
  );
}

/**
 * A single-row, width-aware list of controls. Shows as many as fit inline and
 * collapses the rest into an overflow ("...") popover, recomputing on resize.
 */
function ControlsRow(props: {
  controls: Control[];
  onChange: (control: Control, value: string | null) => void;
}): JSX.Element {
  const { controls, onChange } = props;
  const rowRef = useRef<HTMLDivElement>(null);
  const measureRef = useRef<HTMLDivElement>(null);
  const overflowBtnRef = useRef<HTMLButtonElement>(null);
  const [visibleCount, setVisibleCount] = useState(controls.length);
  const [overflowAnchor, setOverflowAnchor] = useState<HTMLElement | null>(
    null
  );

  // Re-measure only when a control's displayed width could change (its set of
  // ids or effective values), not on every re-render.
  const controlsKey = controls
    .map(p => `${p.id}:${effectiveValue(p)}`)
    .join('|');

  useLayoutEffect(() => {
    const row = rowRef.current;
    const measure = measureRef.current;
    if (!row || !measure) {
      return;
    }
    // The measurement copy exists only to size controls; keep its buttons out of
    // the tab order and the accessibility tree.
    measure.inert = true;
    const GAP = 2;
    let frame = 0;
    const compute = () => {
      const avail = row.clientWidth;
      const widths = (Array.from(measure.children) as HTMLElement[]).map(
        el => el.offsetWidth
      );
      const total = widths.reduce((a, w, i) => a + w + (i ? GAP : 0), 0);
      if (total <= avail) {
        setVisibleCount(widths.length);
        return;
      }
      const reserve =
        (overflowBtnRef.current?.offsetWidth ?? OVERFLOW_BTN_WIDTH) + GAP;
      let used = 0;
      let count = 0;
      for (let i = 0; i < widths.length; i++) {
        const w = widths[i] + (i ? GAP : 0);
        if (used + w + reserve <= avail) {
          used += w;
          count++;
        } else {
          break;
        }
      }
      setVisibleCount(count);
    };
    // A ResizeObserver can fire many times during a drag; coalesce the work to
    // one measurement per animation frame.
    const schedule = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(compute);
    };
    compute();
    const observer = new ResizeObserver(schedule);
    observer.observe(row);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [controlsKey]);

  const visible = controls.slice(0, visibleCount);
  const overflow = controls.slice(visibleCount);

  return (
    <div className={`${SELECTOR_CLASS}-controls`} ref={rowRef}>
      {/* Hidden full-width copy used only to measure each control's width. */}
      <div
        className={`${SELECTOR_CLASS}-controls-measure`}
        ref={measureRef}
        aria-hidden="true"
      >
        {controls.map(control => (
          <ControlMenu
            key={control.id}
            control={control}
            onSelect={v => onChange(control, v)}
          />
        ))}
      </div>

      {visible.map(control => (
        <ControlMenu
          key={control.id}
          control={control}
          onSelect={v => onChange(control, v)}
        />
      ))}

      {overflow.length ? (
        <>
          <button
            type="button"
            ref={overflowBtnRef}
            className={`${SELECTOR_CLASS} ${SELECTOR_CLASS}-overflow-btn`}
            onClick={event => setOverflowAnchor(event.currentTarget)}
            title="More controls"
            aria-label="More controls"
          >
            <MoreHorizIcon fontSize="small" />
          </button>
          <OverflowControlsMenu
            controls={overflow}
            anchor={overflowAnchor}
            onClose={() => setOverflowAnchor(null)}
            onChange={onChange}
          />
        </>
      ) : null}
    </div>
  );
}

// All formatters pin the `en` locale so numbers agree with each other and
// with the surrounding English labels.
const exactNumber = new Intl.NumberFormat('en');
const compactNumber = new Intl.NumberFormat('en', {
  notation: 'compact',
  maximumSignificantDigits: 3
});
const costNumber = new Intl.NumberFormat('en', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2
});

/**
 * Format a token count compactly: 950 stays as-is, 41500 becomes "41.5k",
 * 1240000 becomes "1.24M". `Intl.NumberFormat` picks the tier after rounding,
 * so boundary values like 999500 become "1M" rather than an exponential form.
 * Token values render compactly everywhere (magnitude is what a status surface
 * communicates); the exact count rides on the element's hover title.
 */
export function formatTokens(n: number): string {
  return compactNumber.format(n).replace('K', 'k');
}

/**
 * Format a token count exactly, with thousands separators, for hover titles.
 */
export function formatTokensExact(n: number): string {
  return `${exactNumber.format(n)} tokens`;
}

/**
 * Format a cost amount with its currency code or unit name (e.g. "credits").
 */
export function formatCost(amount: number, currency: string): string {
  const value = costNumber.format(amount);
  return currency === 'USD' ? `$${value}` : `${value} ${currency}`;
}

/**
 * A ring gauge showing how full the context window is. The track is a muted
 * full circle; the fill arc grows clockwise from 12 o'clock and takes the
 * chip's current color, so the warn/error classes color it via `currentColor`.
 */
function UsageRing(props: { fraction: number }): JSX.Element {
  const radius = 6;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.min(Math.max(props.fraction, 0), 1);
  return (
    <svg
      className={`${USAGE_CLASS}-ring`}
      viewBox="0 0 16 16"
      width="16"
      height="16"
      aria-hidden="true"
    >
      <circle
        className={`${USAGE_CLASS}-ring-track`}
        cx="8"
        cy="8"
        r={radius}
        fill="none"
        strokeWidth="2"
      />
      <circle
        className={`${USAGE_CLASS}-ring-fill`}
        cx="8"
        cy="8"
        r={radius}
        fill="none"
        strokeWidth="2"
        strokeDasharray={circumference}
        strokeDashoffset={circumference * (1 - clamped)}
        transform="rotate(-90 8 8)"
      />
    </svg>
  );
}

/**
 * A group header in the usage popover: an uppercase label with the group's
 * headline value. Detail rows, when the group has any, follow beneath.
 */
function UsageSection(props: {
  label: string;
  value: string;
  title?: string;
}): JSX.Element {
  return (
    <div className={`${USAGE_CLASS}-section`} title={props.title}>
      <span>{props.label}</span>
      <span className={`${USAGE_CLASS}-section-value`}>{props.value}</span>
    </div>
  );
}

/**
 * One "label: value" detail row in the usage popover. `title` carries the
 * exact value behind a compact one.
 */
function UsageRow(props: {
  label: string;
  value: string;
  title?: string;
}): JSX.Element {
  return (
    <div className={`${USAGE_CLASS}-row`} title={props.title}>
      <span className={`${USAGE_CLASS}-row-label`}>{props.label}</span>
      <span className={`${USAGE_CLASS}-row-value`}>{props.value}</span>
    </div>
  );
}

/**
 * The usage chip for the input toolbar: a ring gauge and percent of the
 * persona's context-window fill, colored once fill crosses the warn threshold.
 * Hover shows a one-line summary; click opens a popover with the full breakdown
 * (context, session token totals, cost). Renders nothing when the persona has
 * reported no usage at all, so absence reads as unknown rather than empty.
 */
export function UsageChip(props: { usage: Usage }): JSX.Element | null {
  const usage = props.usage;
  const [anchor, setAnchor] = useState<HTMLElement | null>(null);

  const hasContext =
    usage.context_tokens !== null && usage.context_size !== null;
  // Precedence: a token-derived percentage always wins; `context_percent` is
  // read only when the agent reported no token counts (e.g. kiro-cli).
  const hasPercentOnly = !hasContext && usage.context_percent !== null;
  const showContext = hasContext || hasPercentOnly;
  const hasTokens = usage.total_tokens !== null;
  const hasCost = usage.cost_amount !== null && usage.cost_currency !== null;

  if (!showContext && !hasTokens && !hasCost) {
    return null;
  }

  const fraction =
    hasContext && (usage.context_size as number) > 0
      ? (usage.context_tokens as number) / (usage.context_size as number)
      : hasPercentOnly
        ? (usage.context_percent as number) / 100
        : 0;
  const percent = Math.round(fraction * 100);
  const level =
    fraction >= USAGE_ERROR_AT
      ? 'error'
      : fraction >= USAGE_WARN_AT
        ? 'warn'
        : 'ok';

  const summary = [
    hasContext &&
      `Context: ${formatTokens(usage.context_tokens as number)} of ${formatTokens(usage.context_size as number)} tokens (${percent}%)`,
    hasPercentOnly && `Context: ${percent}% used`,
    hasTokens &&
      `Session tokens: ${formatTokens(usage.total_tokens as number)}`,
    hasCost &&
      `Cost: ${formatCost(usage.cost_amount as number, usage.cost_currency as string)}`
  ]
    .filter(Boolean)
    .join('\n');

  return (
    <>
      <button
        type="button"
        className={`${USAGE_CLASS}-chip ${USAGE_CLASS}-${level}`}
        onClick={event => setAnchor(event.currentTarget)}
        title={summary}
        aria-label={showContext ? `Context ${percent}% used` : 'Usage'}
      >
        {showContext ? (
          <>
            <UsageRing fraction={fraction} />
            <span className={`${USAGE_CLASS}-pct`}>{percent}%</span>
          </>
        ) : null}
        {!showContext && hasTokens ? (
          <span className={`${USAGE_CLASS}-pct`}>
            {formatTokens(usage.total_tokens as number)}
          </span>
        ) : null}
      </button>
      <Popover
        anchorEl={anchor}
        open={!!anchor}
        onClose={() => setAnchor(null)}
        {...menuAnchorProps}
      >
        <div className={`${USAGE_CLASS}-card`}>
          {hasContext ? (
            <UsageSection
              label="Context"
              value={`${formatTokens(usage.context_tokens as number)} of ${formatTokens(usage.context_size as number)} (${percent}%)`}
              title={`${exactNumber.format(usage.context_tokens as number)} of ${exactNumber.format(usage.context_size as number)} tokens`}
            />
          ) : null}
          {hasPercentOnly ? (
            <UsageSection label="Context" value={`${percent}%`} />
          ) : null}
          {hasTokens ? (
            <>
              <UsageSection
                label="Session tokens"
                value={formatTokens(usage.total_tokens as number)}
                title={formatTokensExact(usage.total_tokens as number)}
              />
              {usage.input_tokens !== null ? (
                <UsageRow
                  label="Input"
                  value={formatTokens(usage.input_tokens)}
                  title={formatTokensExact(usage.input_tokens)}
                />
              ) : null}
              {usage.output_tokens !== null ? (
                <UsageRow
                  label="Output"
                  value={formatTokens(usage.output_tokens)}
                  title={formatTokensExact(usage.output_tokens)}
                />
              ) : null}
              {usage.cached_read_tokens !== null ? (
                <UsageRow
                  label="Cache read"
                  value={formatTokens(usage.cached_read_tokens)}
                  title={formatTokensExact(usage.cached_read_tokens)}
                />
              ) : null}
              {usage.cached_write_tokens !== null ? (
                <UsageRow
                  label="Cache write"
                  value={formatTokens(usage.cached_write_tokens)}
                  title={formatTokensExact(usage.cached_write_tokens)}
                />
              ) : null}
              {usage.thought_tokens !== null ? (
                <UsageRow
                  label="Thinking"
                  value={formatTokens(usage.thought_tokens)}
                  title={formatTokensExact(usage.thought_tokens)}
                />
              ) : null}
            </>
          ) : null}
          {hasCost ? (
            <UsageSection
              // API list prices are quoted in USD; for any other unit (e.g.
              // metered credits) the amount is the agent's own accounting, so
              // neither the estimate suffix nor the list-price note applies.
              label={
                usage.cost_currency === 'USD'
                  ? 'Session cost (est.)'
                  : 'Session cost'
              }
              value={formatCost(
                usage.cost_amount as number,
                usage.cost_currency as string
              )}
              title={
                usage.cost_currency === 'USD'
                  ? 'Estimated at API list prices'
                  : undefined
              }
            />
          ) : null}
        </div>
      </Popover>
    </>
  );
}

/**
 * The persona control for the chat input toolbar. Shows which persona a message
 * will be directed to (with its avatar), lets the user switch it, and, when the
 * selected persona advertises model/settings, renders those controls next to it.
 * Hides itself when the chat has no personas.
 *
 * All session information (the persona list, each persona's model/settings
 * configuration, usage, and slash commands) is read from the chat's Yjs
 * awareness channel. The selection is owned by the frontend and stamped onto
 * each message's metadata (there is no server-side "active persona" and no REST
 * polling). It's seeded from the default persona advertised over PageConfig.
 */
export function PersonaControls(
  props: InputToolbarRegistry.IToolbarItemProps & {
    /**
     * Registry of controls contributed by other extensions (e.g. a persona's
     * settings button). Rendered for the selected persona after the usage chip.
     * Optional so the component still works without a registry.
     */
    controlRegistry?: IPersonaControlRegistry;
    /**
     * The per-chat persona session-state registry (fed by Jupyter Events).
     * Optional so the component still works without it (renders nothing).
     */
    sessionRegistry?: PersonaSessionRegistry;
  }
): JSX.Element | null {
  const { chatModel, model, controlRegistry, sessionRegistry } = props;
  // The chat's stable id scopes persona events to this chat. It is assigned
  // asynchronously (once the model is `ready`: the WS connection frame arrives,
  // or the RTC document syncs), so track it in state and update it when the
  // model becomes ready. Reading it synchronously would capture `undefined`
  // forever, since `id` is a plain accessor with no change signal.
  const [chatId, setChatId] = useState<string | null>(chatModel?.id ?? null);
  useEffect(() => {
    if (!chatModel) {
      setChatId(null);
      return;
    }
    let cancelled = false;
    void chatModel.ready.then(id => {
      if (!cancelled) {
        setChatId(id ?? null);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [chatModel]);

  // The per-chat persona session state, built from persona events and shared
  // via the registry. Created on demand; discarded when the chat closes.
  const managerState = useMemo(
    () => (sessionRegistry && chatId ? sessionRegistry.get(chatId) : null),
    [sessionRegistry, chatId]
  );

  const [personas, setPersonas] = useState<PersonaOption[]>([]);
  // Whether a persona list has been received for this chat yet. Before that, an
  // empty list means "still loading", not "this chat has no personas".
  const [ready, setReady] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(
    DEFAULT_PERSONA_ID
  );
  const [personaState, setPersonaState] = useState<PersonaSessionState | null>(
    null
  );
  // Per-persona settings the user has chosen, indexed by persona ID. Remembers
  // each persona's picks so switching away and back restores them, rather than
  // resetting to defaults. Component-lifetime only (not persisted across
  // reloads). A persona absent from the cache has made no changes yet.
  const [settingsCache, setSettingsCache] = useState<
    Record<string, PersonaSettings>
  >({});
  const [personaAnchor, setPersonaAnchor] = useState<HTMLElement | null>(null);
  // Whether the user has explicitly picked a persona (or "No one") in this
  // chat. Guards the sole-persona convenience in reconcileSelection.
  const userPicked = useRef(false);

  // The selected persona's settings: its cache entry, or empty (all defaults)
  // when it has none yet.
  const settings = selectedId
    ? (settingsCache[selectedId] ?? emptyPersonaSettings())
    : emptyPersonaSettings();

  // Re-read the persona list from the session state and reconcile the
  // selection (see reconcileSelection for the decision rules).
  const readManager = useCallback(() => {
    if (!managerState) {
      return;
    }
    const list = managerState.personas;
    setPersonas(prev => reconcilePersonas(prev, list));
    setSelectedId(current => {
      const next = reconcileSelection(list, current, userPicked.current);
      return next === undefined ? current : next;
    });
  }, [managerState]);

  // React to the session state's `changed` signal: re-read the persona list and
  // readiness. This replaces the awareness `change` subscription — a persona
  // publishing or updating its state fires `changed`.
  useEffect(() => {
    if (!managerState) {
      return;
    }
    const sync = () => {
      readManager();
      setReady(managerState.ready);
    };
    sync();
    managerState.changed.connect(sync);
    return () => {
      managerState.changed.disconnect(sync);
    };
  }, [managerState, readManager]);

  // Track the selected persona's state, re-reading on every change so the
  // toolbar reflects the latest published model/usage/commands. The first read
  // applies unconditionally (it runs whenever `selectedId` changes and must
  // reflect the newly selected persona, even an absent one); later reads go
  // through reconcilePersonaState so a transient absence doesn't blank it out.
  useEffect(() => {
    if (!managerState || !selectedId) {
      setPersonaState(null);
      return;
    }
    setPersonaState(managerState.getPersona(selectedId) ?? null);
    const read = () =>
      setPersonaState(prev =>
        reconcilePersonaState(prev, managerState.getPersona(selectedId) ?? null)
      );
    managerState.changed.connect(read);
    return () => {
      managerState.changed.disconnect(read);
    };
  }, [managerState, selectedId]);

  // Discard this chat's session state when the chat model is disposed (chat
  // closed), freeing its memory.
  useEffect(() => {
    if (!sessionRegistry || !chatId || !chatModel) {
      return;
    }
    const onDisposed = () => sessionRegistry.discard(chatId);
    chatModel.disposed.connect(onDisposed);
    return () => {
      chatModel.disposed.disconnect(onDisposed);
    };
  }, [sessionRegistry, chatId, chatModel]);

  // Stamp the current persona + its settings onto the input model's metadata,
  // so it rides out with the next message and the PersonaManager routes and
  // applies it. Keyed on a signature so we only write when it changes.
  const metadataSignature = JSON.stringify({ selectedId, settings });
  useEffect(() => {
    model.clearMetadata();
    model.updateMetadata(buildMessageMetadata(selectedId, settings));
  }, [model, metadataSignature]);

  // No personas yet. While the manager's slot or its first list read is still
  // pending, show a loading placeholder (on slow networks this takes seconds);
  // once ready with no personas, show nothing.
  if (!personas.length) {
    // Still loading while the session state exists but no persona list has
    // arrived yet (events are in flight). Once ready with an empty list, or
    // with no registry at all, render nothing.
    if (managerState && !ready) {
      return <LoadingPlaceholder />;
    }
    return null;
  }

  const selectedPersona = personas.find(p => p.id === selectedId) ?? null;
  const personaLabel = selectedPersona?.name ?? NO_ONE_LABEL;
  const activeAvatar = selectedPersona?.avatar_url ?? null;
  const usage = personaState?.usage ?? EMPTY_USAGE;
  const controls = buildControls(personaState, settings);

  const handlePersona = (personaId: string | null) => {
    setPersonaAnchor(null);
    userPicked.current = true;
    setSelectedId(personaId);
  };

  const handleControl = (control: Control, value: string | null) => {
    if (!selectedId) {
      return;
    }
    // Fold the change into this persona's cached settings, remembering it for
    // when the user switches away and back.
    setSettingsCache(prev => ({
      ...prev,
      [selectedId]: applyControlChange(
        prev[selectedId] ?? emptyPersonaSettings(),
        control,
        value
      )
    }));
  };

  return (
    <div className={`${SELECTOR_CLASS}-group`}>
      <Button
        className={`${SELECTOR_CLASS} ${SELECTOR_CLASS}-persona-btn`}
        size="small"
        variant="text"
        disableRipple
        startIcon={<Avatar url={activeAvatar} />}
        endIcon={<ArrowDropDownIcon className={`${SELECTOR_CLASS}-arrow`} />}
        onClick={event => setPersonaAnchor(event.currentTarget)}
        title="Choose which persona to message"
      >
        <span className={`${SELECTOR_CLASS}-persona`}>{personaLabel}</span>
      </Button>
      <Menu
        anchorEl={personaAnchor}
        open={!!personaAnchor}
        onClose={() => setPersonaAnchor(null)}
        {...menuAnchorProps}
      >
        {personas.map(p => (
          <MenuItem
            key={p.id}
            selected={p.id === selectedId}
            onClick={() => handlePersona(p.id)}
          >
            <Avatar url={p.avatar_url} />
            <ListItemText
              primary={p.name}
              classes={{ primary: `${MENU_CLASS}-name` }}
            />
            {p.id === selectedId ? (
              <CheckIcon className={`${MENU_CLASS}-check`} fontSize="small" />
            ) : null}
          </MenuItem>
        ))}
        <MenuItem
          selected={selectedId === null}
          onClick={() => handlePersona(null)}
        >
          <Avatar url={null} />
          <ListItemText
            primary={NO_ONE_LABEL}
            classes={{ primary: `${MENU_CLASS}-name` }}
          />
          {selectedId === null ? (
            <CheckIcon className={`${MENU_CLASS}-check`} fontSize="small" />
          ) : null}
        </MenuItem>
      </Menu>

      <UsageChip usage={usage} />

      {controls.length ? (
        <>
          <span className={`${SELECTOR_CLASS}-divider`} />
          <ControlsRow controls={controls} onChange={handleControl} />
        </>
      ) : null}

      {/* Controls contributed by other extensions for the selected persona
          (e.g. a persona's settings button), rendered after the model selector
          and its settings so they sit to the right of them. */}
      {selectedId &&
        controlRegistry?.getControls(selectedId).map(control => {
          const Control = control.component;
          return (
            <Control
              key={control.id}
              personaId={selectedId}
              chatModel={chatModel}
              model={model}
            />
          );
        })}
    </div>
  );
}
