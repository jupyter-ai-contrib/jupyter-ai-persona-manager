/**
 * Renders the control menus and pins their user-observable contract: each
 * dropdown opens titled by its control's heading, the heading names the menu
 * for assistive tech without joining keyboard traversal, and arrow keys,
 * type-ahead, and Enter drive the choice rows.
 */
import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {
  Control,
  ControlMenu,
  OverflowControlsMenu
} from '../persona-controls';

const SUBHEADER_SELECTOR = 'li.jp-jai-controlMenu-subheader';

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
      expect(screen.getByRole('menu').getAttribute('aria-labelledby')).toBe(
        heading().id
      );
      // Sticky, so the title stays visible while a long option list scrolls.
      expect(heading().classList.contains('MuiListSubheader-sticky')).toBe(
        true
      );
    });

    it('focuses the selected row and moves focus with the keyboard', async () => {
      const user = await openMenu(modelControl('beta'));
      // Rows: heading, "Default (Alpha)", "Alpha", "Beta"; "Beta" is selected.
      expect(focused()).toBe(screen.getByRole('menuitem', { name: 'Beta' }));
      // Wrapping from the last row skips the heading to the first choice row.
      await user.keyboard('{ArrowDown}');
      expect(focused()).toBe(
        screen.getByRole('menuitem', { name: 'Default (Alpha)' })
      );
      await user.keyboard('{ArrowDown}');
      expect(focused()).toBe(screen.getByRole('menuitem', { name: 'Alpha' }));
      // Type-ahead: "b" jumps to Beta.
      await user.keyboard('b');
      expect(focused()).toBe(screen.getByRole('menuitem', { name: 'Beta' }));
    });

    it('leaves focus in place when type-ahead matches only the heading', async () => {
      const user = await openMenu(modelControl('beta'));
      // First key after open, so it cannot buffer onto a previous press: "m"
      // matches the heading ("Model") and no choice row, and the heading
      // never takes focus.
      await user.keyboard('m');
      expect(focused()).toBe(screen.getByRole('menuitem', { name: 'Beta' }));
    });

    it('falls back to the first choice row when the selection is stale', async () => {
      // A selection id the persona no longer advertises leaves no row
      // selected; initial focus then skips the heading to the first row.
      await openMenu(modelControl('stale'));
      expect(focused()).toBe(
        screen.getByRole('menuitem', { name: 'Default (Alpha)' })
      );
    });

    it('activates the focused row with Enter', async () => {
      const onSelect = jest.fn();
      const user = await openMenu(modelControl(null), onSelect);
      await user.keyboard('{ArrowDown}{ArrowDown}{Enter}');
      expect(onSelect).toHaveBeenCalledWith('beta');
    });
  });

  describe('overflow menu', () => {
    it('is labeled and arrows skip the section headings', async () => {
      const first: Control = {
        id: 'a',
        kind: 'setting',
        label: 'Aaa',
        current: null,
        selection: 'a1',
        options: [{ id: 'a1', name: 'A one', description: null }]
      };
      const second: Control = {
        id: 'b',
        kind: 'setting',
        label: 'Bbb',
        current: 'b1',
        selection: 'stale',
        options: [{ id: 'b1', name: 'B one', description: null }]
      };
      const user = userEvent.setup();
      const anchor = document.createElement('button');
      document.body.appendChild(anchor);
      render(
        <OverflowControlsMenu
          controls={[first, second]}
          anchor={anchor}
          onClose={() => undefined}
          onChange={() => undefined}
        />
      );
      expect(screen.getByRole('menu').getAttribute('aria-label')).toBe(
        'More controls'
      );
      const headings = Array.from(
        document.querySelectorAll(SUBHEADER_SELECTOR)
      );
      expect(headings.map(h => h.textContent)).toEqual(['Aaa', 'Bbb']);
      // Sticky, so the section a scrolled row belongs to stays visible.
      for (const h of headings) {
        expect(h.classList.contains('MuiListSubheader-sticky')).toBe(true);
      }
      // "A one" is the only selected row; ArrowDown crosses the "Bbb" heading
      // to the next section's first choice row.
      expect(focused()).toBe(screen.getByRole('menuitem', { name: 'A one' }));
      await user.keyboard('{ArrowDown}');
      expect(focused()).toBe(
        screen.getByRole('menuitem', { name: 'Default (B one)' })
      );
      anchor.remove();
    });
  });
});
