/**
 * Tests for the pure highlight-seeding rule of the searchable menu: on open (or
 * after a filter), the highlight lands on the committed value if it's still in
 * the list, otherwise on the first (best-ranked) option.
 */

import { seedHighlight, SearchableOption } from '../searchable-menu';

const options: SearchableOption[] = [
  { value: null, label: 'Default' },
  { value: 'alpha', label: 'Model Alpha' },
  { value: 'beta', label: 'Model Beta' }
];

describe('seedHighlight', () => {
  it('highlights the committed value when present', () => {
    expect(seedHighlight(options, 'beta')).toBe(2);
    expect(seedHighlight(options, null)).toBe(0);
  });

  it('falls back to the first option when the committed value is absent', () => {
    // e.g. the current selection was filtered out of the list.
    expect(seedHighlight(options, 'gamma')).toBe(0);
  });

  it('returns -1 for an empty list', () => {
    expect(seedHighlight([], 'beta')).toBe(-1);
  });
});
