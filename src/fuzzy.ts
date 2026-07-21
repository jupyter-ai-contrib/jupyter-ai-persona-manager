/**
 * Lightweight fuzzy matching for the searchable control menus. A query matches a
 * candidate when its characters appear in the candidate in order (a subsequence,
 * not just a contiguous substring), so "mb" matches "Model Beta". Matches are
 * scored so the most relevant land first, and the matched character positions
 * are returned so the UI can embolden them.
 *
 * This is deliberately small and dependency-free; it isn't a general fuzzy
 * library, just enough for short option lists (personas, models, settings).
 */

export type FuzzyMatch = {
  /** Whether the query matched at all. */
  matched: boolean;
  /** Higher is a better match. Meaningless when `matched` is false. */
  score: number;
  /** Indices in the original text that the query matched, ascending. */
  indices: number[];
};

const NON_MATCH: FuzzyMatch = { matched: false, score: 0, indices: [] };

/**
 * Whether the character at `text[i]` starts a "word" — the first character, or
 * one that follows a separator or a lower→upper camelCase boundary. Matches at
 * word starts score higher, so "mb" ranks "Model Beta" above an incidental hit.
 */
function isWordStart(text: string, i: number): boolean {
  if (i === 0) {
    return true;
  }
  const prev = text[i - 1];
  const cur = text[i];
  if (/[\s\-_/.:]/.test(prev)) {
    return true;
  }
  return (
    prev === prev.toLowerCase() &&
    cur === cur.toUpperCase() &&
    /[A-Z]/.test(cur)
  );
}

/**
 * Match `query` against `text` as a case-insensitive subsequence. Empty (or
 * whitespace-only) queries match everything with a neutral score and no
 * highlight. Scoring rewards consecutive matches and word-start matches, so the
 * ordering reads as "most relevant first" rather than merely "in list order".
 */
export function fuzzyMatch(text: string, query: string): FuzzyMatch {
  const q = query.trim().toLowerCase();
  if (q.length === 0) {
    return { matched: true, score: 0, indices: [] };
  }
  const lower = text.toLowerCase();
  const indices: number[] = [];
  let score = 0;
  let ti = 0;
  let prevMatch = -2;
  for (let qi = 0; qi < q.length; qi++) {
    const ch = q[qi];
    let found = -1;
    for (let i = ti; i < lower.length; i++) {
      if (lower[i] === ch) {
        found = i;
        break;
      }
    }
    if (found === -1) {
      return NON_MATCH;
    }
    indices.push(found);
    // Base point per matched character, plus bonuses for adjacency (a
    // contiguous run) and for landing on a word start.
    score += 1;
    if (found === prevMatch + 1) {
      score += 2;
    }
    if (isWordStart(text, found)) {
      score += 3;
    }
    prevMatch = found;
    ti = found + 1;
  }
  // Prefer shorter candidates when scores otherwise tie, so an exact-ish short
  // label beats a long one that merely contains the subsequence.
  score -= text.length * 0.01;
  return { matched: true, score, indices };
}

/**
 * Filter and rank `items` by how well `text(item)` fuzzy-matches `query`. An
 * empty/whitespace query returns every item in its original order (no ranking),
 * so an unfiltered menu reads in its natural, meaningful order. Otherwise only
 * matches are kept, best score first; ties keep their original relative order
 * (a stable sort), so the list doesn't jitter.
 */
export function fuzzyFilter<T>(
  items: T[],
  query: string,
  text: (item: T) => string
): T[] {
  if (query.trim().length === 0) {
    return items;
  }
  const scored: { item: T; score: number; order: number }[] = [];
  items.forEach((item, order) => {
    const match = fuzzyMatch(text(item), query);
    if (match.matched) {
      scored.push({ item, score: match.score, order });
    }
  });
  scored.sort((a, b) => b.score - a.score || a.order - b.order);
  return scored.map(s => s.item);
}
