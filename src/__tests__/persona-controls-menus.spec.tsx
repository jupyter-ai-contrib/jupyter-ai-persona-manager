/**
 * Renders the control menus and pins their user-observable contract: each
 * dropdown opens titled by its control's heading, the heading names the menu
 * for assistive tech without joining keyboard traversal, and arrow keys,
 * type-ahead, and Enter drive the choice rows.
 */
import React, { act } from 'react';
import { createRoot, Root } from 'react-dom/client';
import {
  Control,
  ControlMenu,
  OverflowControlsMenu
} from '../persona-controls';

(
  globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;

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

describe('control menus', () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
  });

  function menu(): HTMLElement {
    return document.querySelector('ul[role="menu"]') as HTMLElement;
  }

  function focused(): HTMLElement {
    return document.activeElement as HTMLElement;
  }

  async function press(key: string): Promise<void> {
    await act(async () => {
      focused().dispatchEvent(
        new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true })
      );
    });
  }

  describe('control dropdown', () => {
    async function openMenu(
      control: Control,
      onSelect: (value: string | null) => void = () => undefined
    ): Promise<void> {
      await act(async () => {
        root.render(<ControlMenu control={control} onSelect={onSelect} />);
      });
      await act(async () => {
        (container.querySelector('button') as HTMLElement).dispatchEvent(
          new MouseEvent('click', { bubbles: true })
        );
      });
    }

    it('opens titled by a heading that names the menu', async () => {
      await openMenu(modelControl('beta'));
      const heading = document.querySelector(SUBHEADER_SELECTOR) as HTMLElement;
      expect(heading.textContent).toBe('Model');
      expect(heading.hasAttribute('tabindex')).toBe(false);
      expect(heading.id).toBeTruthy();
      expect(menu().getAttribute('aria-labelledby')).toBe(heading.id);
    });

    it('focuses the selected row and moves focus with the keyboard', async () => {
      await openMenu(modelControl('beta'));
      // Rows: heading, "Default (Alpha)", "Alpha", "Beta"; "Beta" is selected.
      expect(focused().textContent).toBe('Beta');
      expect(focused().getAttribute('role')).toBe('menuitem');
      // Wrapping from the last row skips the heading to the first choice row.
      await press('ArrowDown');
      expect(focused().textContent).toBe('Default (Alpha)');
      await press('ArrowDown');
      expect(focused().textContent).toBe('Alpha');
      // Type-ahead: "b" jumps to Beta.
      await press('b');
      expect(focused().textContent).toBe('Beta');
    });

    it('leaves focus in place when type-ahead matches only the heading', async () => {
      await openMenu(modelControl('beta'));
      // First key after open, so it cannot buffer onto a previous press: "m"
      // matches the heading ("Model") and no choice row, and the heading
      // never takes focus.
      await press('m');
      expect(focused().textContent).toBe('Beta');
    });

    it('falls back to the first choice row when the selection is stale', async () => {
      // A selection id the persona no longer advertises leaves no row
      // selected; initial focus then skips the heading to the first row.
      await openMenu(modelControl('stale'));
      expect(focused().textContent).toBe('Default (Alpha)');
      expect(focused().getAttribute('role')).toBe('menuitem');
    });

    it('activates the focused row with Enter', async () => {
      const onSelect = jest.fn();
      await openMenu(modelControl(null), onSelect);
      await press('ArrowDown');
      await press('ArrowDown');
      await press('Enter');
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
        current: null,
        selection: 'stale',
        options: [{ id: 'b1', name: 'B one', description: null }]
      };
      const anchor = document.createElement('button');
      document.body.appendChild(anchor);
      await act(async () => {
        root.render(
          <OverflowControlsMenu
            controls={[first, second]}
            anchor={anchor}
            onClose={() => undefined}
            onChange={() => undefined}
          />
        );
      });
      expect(menu().getAttribute('aria-label')).toBe('More controls');
      const headings = Array.from(
        document.querySelectorAll(SUBHEADER_SELECTOR)
      ).map(h => h.textContent);
      expect(headings).toEqual(['Aaa', 'Bbb']);
      // "A one" is the only selected row; ArrowDown crosses the "Bbb" heading
      // to the next section's first choice row.
      expect(focused().textContent).toBe('A one');
      await press('ArrowDown');
      expect(focused().textContent).toBe('Default');
      anchor.remove();
    });
  });
});
