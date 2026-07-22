/**
 * Component tests for the searchable menu's open/close/commit and focus-ring
 * behavior in a real DOM. These guard the regressions that shipped in the first
 * cut of the searchable menu:
 *
 *  - clicking the trigger must OPEN and keep the menu open (a ClickAwayListener
 *    race closed it on the same click), and clicking an option must commit it;
 *  - Tab confirms and advances between persona controls, returning to the chat
 *    input past the last one — and never dead-ends (the "Tab again is a no-op"
 *    bug when a value was already selected).
 */
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
  SearchableMenu,
  SearchableOption,
  useSearchableTrigger
} from '../searchable-menu';

const OPTIONS: SearchableOption[] = [
  { value: 'claude', label: 'Claude' },
  { value: 'gpt', label: 'Gippity' },
  { value: null, label: 'No one' }
];

// A single control that opens a searchable menu, in the DOM shape the focus
// ring keys off: a `.jp-jai-personaControls-group` holding trigger buttons.
function OneControl(props: {
  selected?: string | null;
  onChoose?: (v: string | null) => void;
  focusInput?: () => void;
}): JSX.Element {
  const menu = useSearchableTrigger({ focusInput: props.focusInput });
  return (
    <div className="jp-jai-personaControls-group">
      <button
        {...menu.triggerProps}
        className="jp-jai-personaControls-persona-btn"
      >
        trigger
      </button>
      <SearchableMenu
        anchorEl={menu.triggerRef.current}
        open={menu.open}
        options={OPTIONS}
        selectedValue={props.selected ?? null}
        searchLabel="Search personas"
        onChoose={(v, mode) => {
          props.onChoose?.(v);
          menu.choose(mode);
        }}
        onCancel={menu.cancel}
        onClickAway={menu.clickAway}
      />
    </div>
  );
}

async function openMenu(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByText('trigger'));
  await waitFor(() => expect(screen.queryByRole('listbox')).not.toBeNull());
}

describe('SearchableMenu open/close/commit', () => {
  it('stays open after clicking the trigger', async () => {
    const user = userEvent.setup();
    render(<OneControl />);
    await user.click(screen.getByText('trigger'));
    // Let effects (ClickAwayListener mount, rAF focus) settle, then confirm the
    // menu is still open — the opening click must not close it.
    await new Promise(r => setTimeout(r, 50));
    expect(screen.queryByRole('listbox')).not.toBeNull();
    expect(screen.getAllByRole('option')).toHaveLength(3);
  });

  it('commits the option you click', async () => {
    const user = userEvent.setup();
    const chosen: (string | null)[] = [];
    render(<OneControl onChoose={v => chosen.push(v)} />);
    await openMenu(user);
    await user.click(screen.getByText('Claude'));
    expect(chosen).toEqual(['claude']);
    // The menu closes after committing.
    await waitFor(() => expect(screen.queryByRole('listbox')).toBeNull());
  });

  it('opens on focus (Tab into the control) and closes on Escape', async () => {
    const user = userEvent.setup();
    render(<OneControl />);
    await user.tab();
    await waitFor(() => expect(screen.queryByRole('listbox')).not.toBeNull());
    await user.keyboard('{Escape}');
    await waitFor(() => expect(screen.queryByRole('listbox')).toBeNull());
  });

  it('filters options as you type (fuzzy search)', async () => {
    const user = userEvent.setup();
    render(<OneControl />);
    await openMenu(user);
    await user.keyboard('gip');
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(1));
    expect(screen.getByRole('option').textContent).toContain('Gippity');
  });
});

describe('focus ring', () => {
  it('returns focus to the input when Tab confirms from the only control', async () => {
    const user = userEvent.setup();
    let inputFocused = 0;
    render(<OneControl focusInput={() => inputFocused++} />);

    // Tab in (opens), then Tab again to confirm. With no next control, focus
    // returns to the input rather than dead-ending or walking to other buttons.
    await user.tab();
    await waitFor(() => expect(screen.queryByRole('listbox')).not.toBeNull());
    await user.keyboard('{Tab}');
    await waitFor(() => expect(inputFocused).toBeGreaterThan(0));
  });

  it('confirming a control with a value already selected still advances', async () => {
    // Regression: with a value already selected, Tab-to-confirm was a no-op.
    const user = userEvent.setup();
    let inputFocused = 0;
    render(<OneControl selected="claude" focusInput={() => inputFocused++} />);
    await user.tab();
    await waitFor(() => expect(screen.queryByRole('listbox')).not.toBeNull());
    await user.keyboard('{Tab}');
    await waitFor(() => expect(inputFocused).toBeGreaterThan(0));
  });
});

describe('focus ring across multiple controls', () => {
  function TwoControls(props: { focusInput: () => void }): JSX.Element {
    const a = useSearchableTrigger({ focusInput: props.focusInput });
    const b = useSearchableTrigger({ focusInput: props.focusInput });
    return (
      <div className="jp-jai-personaControls-group">
        <button
          {...a.triggerProps}
          className="jp-jai-personaControls-persona-btn"
        >
          persona
        </button>
        <SearchableMenu
          anchorEl={a.triggerRef.current}
          open={a.open}
          options={OPTIONS}
          selectedValue={null}
          searchLabel="Search personas"
          onChoose={(_v, mode) => a.choose(mode)}
          onCancel={a.cancel}
          onClickAway={a.clickAway}
        />
        <button
          {...b.triggerProps}
          className="jp-jai-personaControls-control-btn"
        >
          model
        </button>
        <SearchableMenu
          anchorEl={b.triggerRef.current}
          open={b.open}
          options={OPTIONS}
          selectedValue={null}
          searchLabel="Search Model options"
          onChoose={(_v, mode) => b.choose(mode)}
          onCancel={b.cancel}
          onClickAway={b.clickAway}
        />
      </div>
    );
  }

  it('Tab moves from one control to the next, then out to the input', async () => {
    const user = userEvent.setup();
    let inputFocused = 0;
    render(<TwoControls focusInput={() => inputFocused++} />);

    // Tab into the first control (persona) — its menu opens.
    await user.tab();
    await waitFor(() =>
      expect(document.activeElement?.getAttribute('aria-label')).toBe(
        'Search personas'
      )
    );

    // Tab confirms and advances to the model control, opening its menu.
    await user.keyboard('{Tab}');
    await waitFor(() =>
      expect(document.activeElement?.getAttribute('aria-label')).toBe(
        'Search Model options'
      )
    );
    // No premature exit to the input yet.
    expect(inputFocused).toBe(0);

    // Tab from the last control returns focus to the input.
    await user.keyboard('{Tab}');
    await waitFor(() => expect(inputFocused).toBeGreaterThan(0));
  });
});
