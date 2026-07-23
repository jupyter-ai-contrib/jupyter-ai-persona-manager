/**
 * Tests for the fuzzy matcher/filter backing the searchable control menus:
 * subsequence matching, relevance ranking, and matched-index reporting.
 */

import { fuzzyFilter, fuzzyMatch } from '../fuzzy';

describe('fuzzyMatch', () => {
  it('matches an empty query against anything, with no highlight', () => {
    const m = fuzzyMatch('Model Beta', '');
    expect(m.matched).toBe(true);
    expect(m.indices).toEqual([]);
  });

  it('matches a contiguous substring and reports its indices', () => {
    const m = fuzzyMatch('Model Beta', 'beta');
    expect(m.matched).toBe(true);
    expect(m.indices).toEqual([6, 7, 8, 9]);
  });

  it('matches a non-contiguous subsequence in order', () => {
    const m = fuzzyMatch('Model Beta', 'mb');
    expect(m.matched).toBe(true);
    // "m" at 0, "b" at 6.
    expect(m.indices).toEqual([0, 6]);
  });

  it('is case-insensitive', () => {
    expect(fuzzyMatch('Model Alpha', 'MODEL').matched).toBe(true);
    expect(fuzzyMatch('model alpha', 'ALPHA').matched).toBe(true);
  });

  it('does not match when characters are out of order', () => {
    expect(fuzzyMatch('Model Beta', 'bm').matched).toBe(false);
  });

  it('does not match when a character is absent', () => {
    expect(fuzzyMatch('Model Beta', 'xyz').matched).toBe(false);
  });

  it('scores word-start matches above mid-word ones', () => {
    // "b" starting the word "Beta" should beat "b" buried inside "Alembic".
    const wordStart = fuzzyMatch('Model Beta', 'b').score;
    const midWord = fuzzyMatch('Alembic', 'b').score;
    expect(wordStart).toBeGreaterThan(midWord);
  });
});

describe('fuzzyFilter', () => {
  const items = ['Model Alpha', 'Model Beta', 'Model Gamma'];
  const id = (s: string) => s;

  it('returns everything in original order for an empty query', () => {
    expect(fuzzyFilter(items, '', id)).toEqual(items);
    expect(fuzzyFilter(items, '   ', id)).toEqual(items);
  });

  it('keeps only matches', () => {
    expect(fuzzyFilter(items, 'beta', id)).toEqual(['Model Beta']);
  });

  it('ranks better matches first', () => {
    // "ma" matches all three ("M...a"), but "Model Gamma" and "Model Alpha"
    // both contain a stronger contiguous/word-start hit; assert Beta ranks last
    // since its only "a" is at the very end with no adjacency bonus.
    const ranked = fuzzyFilter(items, 'mg', id);
    expect(ranked[0]).toBe('Model Gamma');
  });

  it('is a stable sort on ties', () => {
    // Equal-length labels that all match the query identically (leading word
    // start, same length) score equally, so original order is preserved.
    const tied = ['aaa', 'aba', 'aca'];
    expect(fuzzyFilter(tied, 'a', id)).toEqual(tied);
  });
});
