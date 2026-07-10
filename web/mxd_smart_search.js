// Token-based fuzzy matcher for model filename search (lora/checkpoint/etc pickers).
// Lets queries like "wan 2.2 i2v lightning low" match
// "wan2.2_i2v_14b_lightning_v1.1_low.safetensors" -- order-independent, separator/case
// insensitive, with light typo tolerance. Pure function, no DOM dependency.

const RANK_EXACT = 0;
const RANK_PREFIX = 1;
const RANK_TOKEN_SUBSTRING = 2;
const RANK_SQUASHED = 3;
const RANK_FUZZY = 4;

function levenshtein(a, b) {
  if (a === b) return 0;
  const al = a.length;
  const bl = b.length;
  if (al === 0) return bl;
  if (bl === 0) return al;

  let prev = new Array(bl + 1);
  let curr = new Array(bl + 1);
  for (let j = 0; j <= bl; j++) prev[j] = j;

  for (let i = 1; i <= al; i++) {
    curr[0] = i;
    const ca = a.charCodeAt(i - 1);
    for (let j = 1; j <= bl; j++) {
      const cost = ca === b.charCodeAt(j - 1) ? 0 : 1;
      curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost);
    }
    [prev, curr] = [curr, prev];
  }
  return prev[bl];
}

function squashAlnum(str) {
  return String(str)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "");
}

function tokenize(str) {
  return String(str)
    .toLowerCase()
    .replace(/\.(?!\d)/g, " ") // dots not followed by a digit are separators (also strips extensions)
    .replace(/[_\-/\\,()[\]]+/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

function scoreWord(word, tokens, squashedCandidate) {
  let best = null;
  for (const tok of tokens) {
    if (tok === word) return RANK_EXACT;
    if (tok.startsWith(word)) {
      if (best === null || best > RANK_PREFIX) best = RANK_PREFIX;
    } else if (tok.includes(word)) {
      if (best === null || best > RANK_TOKEN_SUBSTRING) best = RANK_TOKEN_SUBSTRING;
    }
  }
  if (best !== null) return best;

  const squashedWord = squashAlnum(word);
  if (squashedWord.length >= 2 && squashedCandidate.includes(squashedWord)) {
    return RANK_SQUASHED;
  }

  if (word.length >= 4) {
    let bestDist = Infinity;
    for (const tok of tokens) {
      if (Math.abs(tok.length - word.length) > 2) continue;
      const d = levenshtein(word, tok);
      if (d < bestDist) bestDist = d;
      if (bestDist === 0) break;
    }
    const threshold = word.length >= 8 ? 2 : 1;
    if (bestDist <= threshold) return RANK_FUZZY;
  }

  return null;
}

// Returns a numeric score (lower = better match) or null if the query doesn't match.
export function matchScore(query, candidateText) {
  const q = String(query || "").trim().toLowerCase();
  if (!q) return 0;

  const words = q.split(/\s+/).filter(Boolean);
  const tokens = tokenize(candidateText);
  const squashedCandidate = squashAlnum(candidateText);

  let total = 0;
  for (const word of words) {
    const wordScore = scoreWord(word, tokens, squashedCandidate);
    if (wordScore === null) return null;
    total += wordScore;
  }
  return total;
}

export function isSmartMatch(query, candidateText) {
  return matchScore(query, candidateText) !== null;
}
