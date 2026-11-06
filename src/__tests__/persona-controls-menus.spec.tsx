/**
 * Renders the control menus and pins their user-observable contract: each
 * dropdown opens titled by its control's heading, the heading names the menu
 * for assistive tech without joining keyboard traversal, and arrow keys,
 * type-ahead, and Enter drive the choice rows.
 */
import React from 'react';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Control, ControlMenu } from '../persona-controls';

const SUBHEADER_SELECTOR = 'li.jp-jai-controlMenu-subheader';
const KBD_FOCUSED_SELECTOR = 'li.jp-jai-controlMenu-kbd-focused';
const RESULTS_SELECTOR = 'div.jp-jai-controlMenu-search-list';

// The trigger button shows the current effective value as its own label
// (e.g. "Beta" when that's selected), which can collide with a row of the
// same name in the open dropdown - scope result-row queries to the results
// list, not the whole document, to avoid matching the button too.
function results(): HTMLElement {
  return document.querySelector(RESULTS_SELECTOR) as HTMLElement;
}

function modelControl(selection: string | null): Control {
  return {
    id: '__model__',
    kind: 'model',
    label: 'Model',
    current: 'alpha',
    selection,
    options: [
      { id: 'alpha', name: 'Alpha', description: null },
      { id: 'beta', name: 'Beta', description: null }
    ]
  };
}

function heading(): HTMLElement {
  return document.querySelector(SUBHEADER_SELECTOR) as HTMLElement;
}

function focused(): HTMLElement {
  return document.activeElement as HTMLElement;
}

// ControlMenu shows keyboard navigation as a visual highlight on the
// currently-considered row rather than moving real DOM focus off the search
// input (see the comment on ControlMenu in persona-controls.tsx) - so
// keyboard-driven tests read this instead of `focused()`.
function highlighted(): HTMLElement | null {
  return document.querySelector(KBD_FOCUSED_SELECTOR);
}

describe('control menus', () => {
  describe('control dropdown', () => {
    async function openMenu(
      control: Control,
      onSelect: (value: string | null) => void = () => undefined
    ): Promise<ReturnType<typeof userEvent.setup>> {
      const user = userEvent.setup();
      render(<ControlMenu control={control} onSelect={onSelect} />);
      await user.click(screen.getByRole('button'));
      return user;
    }

    it('opens titled by a heading that names the menu', async () => {
      await openMenu(modelControl('beta'));
      expect(heading().textContent).toBe('Model');
      expect(heading().hasAttribute('tabindex')).toBe(false);
      expect(heading().id).toBeTruthy();
      // Sticky, so the title stays visible while a long option list scrolls.
      expect(heading().classList.contains('MuiListSubheader-sticky')).toBe(
        true
      );
    });

    it('opens with the search box focused, so typing filters immediately', async () => {
      await openMenu(modelControl('beta'));
      expect(focused()).toBe(screen.getByPlaceholderText('Search model'));
    });

    it('highlights the selected row on open, without moving real focus off the search box', async () => {
      const user = await openMenu(modelControl('beta'));
      expect(highlighted()?.textContent).toContain('Beta');
      // Real DOM focus never leaves the search box - arrow keys move the
      // highlight, not focus, so the user can keep typing at any point.
      await user.keyboard('{ArrowDown}');
      expect(focused()).toBe(screen.getByPlaceholderText('Search model'));
    });

    it('falls back to the Default row when the selection is stale', async () => {
      // A selection id the persona no longer advertises matches no row;
      // initial highlight then falls back to the Default row.
      await openMenu(modelControl('stale'));
      expect(highlighted()?.textContent).toContain('Default (Alpha)');
    });

    it('moves the highlight with the arrow keys, wrapping at both ends', async () => {
      // Rows: "Default (Alpha)", "Alpha", "Beta"; starts on "Beta" (selected).
      const user = await openMenu(modelControl('beta'));
      expect(highlighted()?.textContent).toContain('Beta');
      await user.keyboard('{ArrowDown}');
      expect(highlighted()?.textContent).toContain('Default (Alpha)');
      await user.keyboard('{ArrowUp}');
      expect(highlighted()?.textContent).toContain('Beta');
    });

    it('filters rows by the search text, always keeping the Default row', async () => {
      const user = await openMenu(modelControl('beta'));
      await user.keyboard('bet');
      // getByText throws if not found, so a successful call is itself the
      // presence assertion.
      within(results()).getByText('Beta');
      within(results()).getByText('Default (Alpha)');
      expect(within(results()).queryByText('Alpha')).toBeNull();
    });

    it('resets the highlight to the first (filtered) row as the search text changes', async () => {
      const user = await openMenu(modelControl('beta'));
      await user.keyboard('bet');
      // Only "Default (Alpha)" and "Beta" match; highlight resets to the
      // first row rather than staying on an index that may no longer exist.
      expect(highlighted()?.textContent).toContain('Default (Alpha)');
    });

    it('activates the highlighted row with Enter', async () => {
      const onSelect = jest.fn();
      const user = await openMenu(modelControl(null), onSelect);
      await user.keyboard('{ArrowDown}{ArrowDown}{Enter}');
      expect(onSelect).toHaveBeenCalledWith('beta');
    });

    it('closes on Escape without selecting anything', async () => {
      const onSelect = jest.fn();
      const user = await openMenu(modelControl('beta'), onSelect);
      await user.keyboard('{Escape}');
      expect(screen.queryByPlaceholderText('Search model')).toBeNull();
      expect(onSelect).not.toHaveBeenCalled();
    });
  });
});
