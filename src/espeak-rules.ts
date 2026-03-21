/**
 * Simplified port of espeak-ng's English letter-to-phoneme rules.
 *
 * Parses the en_rules file format and applies rules to convert
 * unknown English words to IPA phonemes.
 */

// ── Espeak internal phoneme → IPA mapping ──

const PHONEME_TO_IPA: Record<string, string> = {
  // Vowels
  'a': 'æ',
  'a#': 'ɐ',
  'A:': 'ɑː',
  'A@': 'ɑːɹ',  // espeak A@ = /ɑːɹ/
  'aa': 'ɑː',
  'a:': 'ɑː',
  'ai': 'aɪ',
  'aI': 'aɪ',
  'aU': 'aʊ',
  'au': 'aʊ',
  'e': 'ɛ',
  'E': 'ɛ',
  'E2': 'ɛ',
  'e#': 'ɛ',
  'eI': 'eɪ',
  'e@': 'ɛɹ',
  'i': 'ɪ',
  'i:': 'iː',
  'I': 'ɪ',
  'I2': 'ɪ',
  'I#': 'ɪ',
  '0': 'ɒ',
  '0#': 'ɒ',
  'oU': 'oʊ',
  'O:': 'ɔː',
  'O@': 'ɔːɹ',
  'O2': 'ɒ',
  'OI': 'ɔɪ',
  'u:': 'uː',
  'U': 'ʊ',
  'V': 'ʌ',
  'VR': 'ɜːɹ',
  '3:': 'ɜː',
  '3': 'ɜ',
  '@': 'ə',
  '@2': 'ə',
  '@5': 'ə',
  '@L': 'əl',
  '@-': 'ə',

  // Diphthongs (already covered above, but be explicit)

  // Consonants
  'b': 'b',
  'd': 'd',
  'f': 'f',
  'g': 'ɡ',
  'h': 'h',
  'j': 'j',
  'k': 'k',
  'l': 'l',
  'L': 'l',
  'm': 'm',
  'n': 'n',
  'N': 'ŋ',
  'p': 'p',
  'r': 'ɹ',
  'R': 'ɹ',
  's': 's',
  'S': 'ʃ',
  't': 't',
  't2': 't',
  'T': 'θ',
  'D': 'ð',
  'v': 'v',
  'w': 'w',
  'x': 'x',
  'z': 'z',
  'Z': 'ʒ',
  'dZ': 'dʒ',
  'tS': 'tʃ',
  '?': 'ʔ',

  // Stress / prosody
  "'": 'ˈ',
  ',': 'ˌ',
  '%': '',     // unstressed marker — omit
  ':': 'ː',
  '=': '',     // syllable boundary marker — omit
  '#': '',     // unused/variant marker — omit
  '-': '',     // morpheme boundary — omit
  '|': '',     // tie marker — omit
  '_': ' ',    // word boundary in phonemes

  // Clusters
  'IR': 'ɪɹ',
  'th': 'tθ',  // rarely used
  'n-': 'n',
  'z#': 'z',
  'z/2': 'z',
};

// Order matters: try longer keys first
const PHONEME_KEYS_SORTED = Object.keys(PHONEME_TO_IPA).sort((a, b) => b.length - a.length);

/**
 * Convert espeak internal phoneme string to IPA.
 */
export function espeakToIPA(espeak: string): string {
  let result = '';
  let i = 0;
  while (i < espeak.length) {
    let matched = false;
    // Try longest match first
    for (const key of PHONEME_KEYS_SORTED) {
      if (espeak.startsWith(key, i)) {
        result += PHONEME_TO_IPA[key];
        i += key.length;
        matched = true;
        break;
      }
    }
    if (!matched) {
      // Skip unknown characters
      i++;
    }
  }
  return result;
}

// ── Rule parsing ──

/** Letter groups defined in en_rules */
const LETTER_GROUPS: Record<string, Set<string>> = {};

/** A single pronunciation rule */
interface Rule {
  /** The letter pattern being matched (the group letters) */
  pattern: string;
  /** Pre-condition (letters before the pattern) — reversed for matching */
  pre: string;
  /** Post-condition (letters after the pattern) */
  post: string;
  /** Espeak phoneme output */
  phonemes: string;
  /** Conditional flag number (0 = unconditional) */
  conditionNum: number;
  /** Whether the condition is negated */
  conditionNeg: boolean;
}

/** Map from group key to list of rules */
const RULE_GROUPS: Map<string, Rule[]> = new Map();

const VOWELS = new Set('aeiouyàáâãäåæèéêëìíîïòóôõöùúûüý'.split(''));
const CONSONANTS = new Set('bcdfghjklmnpqrstvwxz'.split(''));

function isVowel(ch: string): boolean {
  return VOWELS.has(ch.toLowerCase());
}

function isConsonant(ch: string): boolean {
  return CONSONANTS.has(ch.toLowerCase());
}

function isLetter(ch: string): boolean {
  return /[a-zA-Zàáâãäåæèéêëìíîïòóôõöùúûüýÿ]/u.test(ch);
}

/**
 * Check if a condition character matches at a position in a word.
 *
 * espeak-ng condition characters in pre/post contexts:
 * - Uppercase letter: literal match (case-insensitive)
 * - Lowercase letter: literal match
 * - A: any vowel
 * - B: voiced consonant (b,d,g,j,l,m,n,r,v,w,z)
 * - C: any consonant
 * - D: any digit
 * - H: some specific set (we use: ch, sh, th digraph)
 * - K: any non-vowel (consonant or boundary)
 * - N: nasal (m, n, ng)
 * - P: a pair of digits (used for $w_alt)
 * - S: suffix marker (number of chars)
 * - X: any letter
 * - Y: consonant cluster boundary
 * - _: word boundary
 * - &: any letter (vowel or consonant) that has been matched
 * - +: one or more additional letters following
 * - #: word boundary or vowel
 */

const VOICED_CONSONANTS = new Set('bdgjlmnrvwz'.split(''));

/**
 * Match a post-condition string against text starting at position.
 * Returns number of characters consumed from text, or -1 if no match.
 */
function matchPost(cond: string, text: string, pos: number): number {
  let ci = 0;
  let ti = pos;

  while (ci < cond.length) {
    const cc = cond[ci];

    // Skip conditional markers ?N, !?N, @PN, $w_alt etc.
    if (cc === '?' || cc === '@' || cc === '$') {
      // Skip the rest — these are advanced conditions we don't handle
      return ti - pos; // Accept what we've matched so far
    }

    if (cc === '+') {
      // One or more letters following
      if (ti < text.length && isLetter(text[ti])) {
        ci++;
        // Match one, then continue
        ti++;
        // Plus means "one or more" - we already consumed one, that's enough
        continue;
      }
      return -1;
    }

    if (cc === '_') {
      // Word boundary
      if (ti >= text.length || !isLetter(text[ti])) {
        ci++;
        continue;
      }
      return -1;
    }

    if (cc === '#') {
      // Word boundary or vowel
      if (ti >= text.length || !isLetter(text[ti]) || isVowel(text[ti])) {
        ci++;
        continue;
      }
      return -1;
    }

    if (cc === 'A') {
      // Any vowel
      if (ti < text.length && isVowel(text[ti])) {
        ci++;
        ti++;
        continue;
      }
      return -1;
    }

    if (cc === 'B') {
      // Voiced consonant
      if (ti < text.length && VOICED_CONSONANTS.has(text[ti].toLowerCase())) {
        ci++;
        ti++;
        continue;
      }
      return -1;
    }

    if (cc === 'C') {
      // Any consonant
      if (ti < text.length && isConsonant(text[ti])) {
        ci++;
        ti++;
        continue;
      }
      return -1;
    }

    if (cc === 'D') {
      // Digit
      if (ti < text.length && /\d/.test(text[ti])) {
        ci++;
        ti++;
        continue;
      }
      return -1;
    }

    if (cc === 'K') {
      // Not a vowel (consonant or boundary)
      if (ti >= text.length || !isVowel(text[ti])) {
        ci++;
        if (ti < text.length && isLetter(text[ti])) ti++;
        continue;
      }
      return -1;
    }

    if (cc === 'N') {
      // Nasal
      if (ti < text.length && 'mn'.includes(text[ti].toLowerCase())) {
        ci++;
        ti++;
        continue;
      }
      // Check for 'ng'
      if (ti + 1 < text.length && text[ti].toLowerCase() === 'n' && text[ti + 1].toLowerCase() === 'g') {
        ci++;
        ti += 2;
        continue;
      }
      return -1;
    }

    if (cc === 'X') {
      // Any letter
      if (ti < text.length && isLetter(text[ti])) {
        ci++;
        ti++;
        continue;
      }
      return -1;
    }

    if (cc === 'Y') {
      // Consonant cluster: matches if current is consonant (or boundary for start)
      if (ti < text.length && isConsonant(text[ti])) {
        ci++;
        ti++;
        continue;
      }
      if (ti >= text.length) {
        ci++;
        continue;
      }
      return -1;
    }

    // Check for letter group reference: L followed by two digits
    if (cc === 'L' && ci + 2 < cond.length && /\d/.test(cond[ci + 1]) && /\d/.test(cond[ci + 2])) {
      const groupId = cond[ci + 1] + cond[ci + 2];
      const group = LETTER_GROUPS[groupId];
      ci += 3;
      if (group) {
        // Try to match any entry in the group (entries can be multi-char)
        let found = false;
        for (const entry of group) {
          if (text.substring(ti, ti + entry.length).toLowerCase() === entry) {
            ti += entry.length;
            found = true;
            break;
          }
        }
        if (!found) return -1;
      }
      continue;
    }

    // Literal match (case-insensitive)
    if (ti < text.length && text[ti].toLowerCase() === cc.toLowerCase()) {
      ci++;
      ti++;
      continue;
    }

    return -1;
  }

  return ti - pos;
}

/**
 * Match a pre-condition string against text ending at position (exclusive).
 * The pre-condition in the rules is written left-to-right as it appears before the pattern,
 * so we match it right-to-left against the text.
 *
 * Returns the number of characters consumed (going backwards), or -1 if no match.
 */
function matchPre(cond: string, text: string, pos: number): number {
  // Walk cond from right to left, text from pos-1 backwards
  let ci = cond.length - 1;
  let ti = pos - 1;

  while (ci >= 0) {
    const cc = cond[ci];

    // Skip conditional markers
    if (cc === '?' || cc === '@' || cc === '$') {
      return pos - ti - 1; // Accept
    }

    if (cc === '_') {
      // Word boundary
      if (ti < 0 || !isLetter(text[ti])) {
        ci--;
        continue;
      }
      return -1;
    }

    if (cc === '#') {
      if (ti < 0 || !isLetter(text[ti]) || isVowel(text[ti])) {
        ci--;
        continue;
      }
      return -1;
    }

    if (cc === '&' || cc === '@') {
      // & matches any letter that has been part of previous matches
      // @ matches a vowel for pre-context
      if (ti >= 0 && isLetter(text[ti])) {
        ci--;
        ti--;
        continue;
      }
      return -1;
    }

    if (cc === 'A') {
      if (ti >= 0 && isVowel(text[ti])) {
        ci--;
        ti--;
        continue;
      }
      return -1;
    }

    if (cc === 'B') {
      if (ti >= 0 && VOICED_CONSONANTS.has(text[ti].toLowerCase())) {
        ci--;
        ti--;
        continue;
      }
      return -1;
    }

    if (cc === 'C') {
      if (ti >= 0 && isConsonant(text[ti])) {
        ci--;
        ti--;
        continue;
      }
      return -1;
    }

    if (cc === 'D') {
      if (ti >= 0 && /\d/.test(text[ti])) {
        ci--;
        ti--;
        continue;
      }
      return -1;
    }

    if (cc === 'K') {
      if (ti < 0 || !isVowel(text[ti])) {
        ci--;
        if (ti >= 0 && isLetter(text[ti])) ti--;
        continue;
      }
      return -1;
    }

    if (cc === 'X') {
      if (ti >= 0 && isLetter(text[ti])) {
        ci--;
        ti--;
        continue;
      }
      return -1;
    }

    // Letter group: look for Lnn pattern — but since we scan right-to-left,
    // check if cc is a digit and preceded by another digit and 'L'
    if (/\d/.test(cc) && ci >= 2 && /\d/.test(cond[ci - 1]) && cond[ci - 2] === 'L') {
      const groupId = cond[ci - 1] + cc;
      const group = LETTER_GROUPS[groupId];
      ci -= 3;
      if (group) {
        let found = false;
        for (const entry of group) {
          const start = ti - entry.length + 1;
          if (start >= 0 && text.substring(start, ti + 1).toLowerCase() === entry) {
            ti = start - 1;
            found = true;
            break;
          }
        }
        if (!found) return -1;
      }
      continue;
    }

    // Literal
    if (ti >= 0 && text[ti].toLowerCase() === cc.toLowerCase()) {
      ci--;
      ti--;
      continue;
    }

    return -1;
  }

  return pos - ti - 1;
}

/**
 * Parse the en_rules file and populate RULE_GROUPS and LETTER_GROUPS.
 */
export function parseRules(rulesText: string): void {
  RULE_GROUPS.clear();
  // Clear letter groups
  for (const k of Object.keys(LETTER_GROUPS)) delete LETTER_GROUPS[k];

  const lines = rulesText.split('\n');
  let currentGroup = '';
  let inReplace = false;

  for (let lineNum = 0; lineNum < lines.length; lineNum++) {
    let line = lines[lineNum];

    // Remove comments (but be careful — // can appear inside the rule)
    const commentIdx = line.indexOf('//');
    if (commentIdx >= 0) {
      line = line.substring(0, commentIdx);
    }

    line = line.trimEnd();
    if (!line.trim()) continue;

    // Letter group definitions: .L01  a b c ...
    const groupMatch = line.match(/^\.L(\d+)\s+(.*)/);
    if (groupMatch) {
      const groupId = groupMatch[1];
      const entries = groupMatch[2].trim().split(/\s+/);
      LETTER_GROUPS[groupId] = new Set(entries.map(e => e.toLowerCase()));
      continue;
    }

    // Section markers
    if (line.trim() === '.replace') {
      inReplace = true;
      continue;
    }

    if (line.trim().startsWith('.group')) {
      inReplace = false;
      currentGroup = line.trim().substring(7).trim();
      if (!RULE_GROUPS.has(currentGroup)) {
        RULE_GROUPS.set(currentGroup, []);
      }
      continue;
    }

    if (inReplace) continue;

    // Parse a rule line. Format:
    //   [?N] [pre) ] pattern [(post] phonemes
    // where pattern is the letter(s) being matched

    // Skip lines that don't look like rules
    if (!currentGroup && !RULE_GROUPS.has('')) continue;

    const rule = parseRuleLine(line.trim());
    if (rule) {
      const group = currentGroup;
      if (!RULE_GROUPS.has(group)) {
        RULE_GROUPS.set(group, []);
      }
      RULE_GROUPS.get(group)!.push(rule);
    }
  }
}

function parseRuleLine(line: string): Rule | null {
  if (!line) return null;

  let conditionNum = 0;
  let conditionNeg = false;

  // Check for conditional: ?N or ?!N or !?N at start
  let idx = 0;
  while (idx < line.length && (line[idx] === ' ' || line[idx] === '\t')) idx++;

  if (line[idx] === '?' || line[idx] === '!') {
    if (line[idx] === '!' && line[idx + 1] === '?') {
      conditionNeg = true;
      idx += 2;
    } else if (line[idx] === '?' && line[idx + 1] === '!') {
      conditionNeg = true;
      idx += 2;
    } else if (line[idx] === '?') {
      idx++;
    }
    // Read the condition number
    let numStr = '';
    while (idx < line.length && /\d/.test(line[idx])) {
      numStr += line[idx];
      idx++;
    }
    conditionNum = parseInt(numStr, 10) || 0;
  }

  // Skip whitespace
  while (idx < line.length && (line[idx] === ' ' || line[idx] === '\t')) idx++;

  const rest = line.substring(idx);
  if (!rest) return null;

  // Find the pattern and phonemes sections
  // Format: [pre)] pattern [(post] phonemes
  // The pattern is the group's letter sequence
  // pre ends with )
  // post starts with (

  let pre = '';
  let pattern = '';
  let post = '';
  let phonemes = '';

  // Find the ) for pre-condition
  const parenClose = rest.indexOf(')');
  const parenOpen = rest.indexOf('(');

  let patternStart = 0;

  if (parenClose >= 0) {
    pre = rest.substring(0, parenClose).trim();
    patternStart = parenClose + 1;
  }

  // Skip whitespace after pre
  while (patternStart < rest.length && rest[patternStart] === ' ') patternStart++;

  if (parenOpen >= 0 && parenOpen > parenClose) {
    // There's a post-condition
    pattern = rest.substring(patternStart, parenOpen).trim();
    // Find the end of the post condition — it goes until whitespace
    let postEnd = parenOpen + 1;
    while (postEnd < rest.length && rest[postEnd] !== ' ' && rest[postEnd] !== '\t') {
      postEnd++;
    }
    post = rest.substring(parenOpen + 1, postEnd);
    phonemes = rest.substring(postEnd).trim();
  } else {
    // No post-condition — pattern then phonemes separated by whitespace
    // Pattern is the contiguous non-space chars, then phonemes is the rest
    let patternEnd = patternStart;
    while (patternEnd < rest.length && rest[patternEnd] !== ' ' && rest[patternEnd] !== '\t') {
      patternEnd++;
    }
    pattern = rest.substring(patternStart, patternEnd).trim();
    phonemes = rest.substring(patternEnd).trim();
  }

  if (!pattern) return null;

  return { pattern, pre, post, phonemes, conditionNum, conditionNeg };
}

/**
 * Apply rules to a single word, producing espeak internal phoneme string.
 */
function applyRulesToWord(word: string): string {
  const lower = word.toLowerCase();
  let result = '';
  let i = 0;

  while (i < lower.length) {
    let bestRule: Rule | null = null;
    let bestPatternLen = 0;
    let bestScore = -1;

    // Try multi-char groups first (longer patterns = better match)
    // espeak tries groups in order of the letters at the current position
    // Try progressively shorter substrings from current position as group keys
    const groupKeys: string[] = [];

    // Try longest group keys first (e.g., "th", "ch", "sh", "ou", etc.)
    for (let len = Math.min(4, lower.length - i); len >= 1; len--) {
      const key = lower.substring(i, i + len);
      if (RULE_GROUPS.has(key)) {
        groupKeys.push(key);
      }
    }

    // Also try the single-letter group and the default group
    const singleKey = lower[i];
    if (!groupKeys.includes(singleKey) && RULE_GROUPS.has(singleKey)) {
      groupKeys.push(singleKey);
    }

    for (const groupKey of groupKeys) {
      const rules = RULE_GROUPS.get(groupKey);
      if (!rules) continue;

      for (const rule of rules) {
        // Skip conditional rules for now (we use default dialect)
        if (rule.conditionNum > 0) continue;

        // Check if the full pattern matches at position i
        const patternLower = rule.pattern.toLowerCase();

        // The group key is already the start of the pattern for multi-char groups
        // But the rule.pattern might extend beyond the group key
        // For group "th", a rule pattern might be "the" or "th" or just have pre/post conditions
        if (!lower.startsWith(patternLower, i)) continue;

        const patternEnd = i + patternLower.length;

        // Check pre-condition
        if (rule.pre) {
          const preMatch = matchPre(rule.pre, lower, i);
          if (preMatch < 0) continue;
        }

        // Check post-condition
        if (rule.post) {
          const postMatch = matchPost(rule.post, lower, patternEnd);
          if (postMatch < 0) continue;
        }

        // Score: longer pattern = better, pre+post conditions = better
        const score = patternLower.length * 100 + rule.pre.length * 10 + rule.post.length;
        if (score > bestScore) {
          bestScore = score;
          bestRule = rule;
          bestPatternLen = patternLower.length;
        }
      }
    }

    // Also try the default group (empty key "")
    const defaultRules = RULE_GROUPS.get('');
    if (defaultRules) {
      for (const rule of defaultRules) {
        if (rule.conditionNum > 0) continue;
        const patternLower = rule.pattern.toLowerCase();
        if (!lower.startsWith(patternLower, i)) continue;

        const patternEnd = i + patternLower.length;

        if (rule.pre) {
          const preMatch = matchPre(rule.pre, lower, i);
          if (preMatch < 0) continue;
        }
        if (rule.post) {
          const postMatch = matchPost(rule.post, lower, patternEnd);
          if (postMatch < 0) continue;
        }

        const score = patternLower.length * 100 + rule.pre.length * 10 + rule.post.length;
        if (score > bestScore) {
          bestScore = score;
          bestRule = rule;
          bestPatternLen = patternLower.length;
        }
      }
    }

    if (bestRule) {
      result += bestRule.phonemes;
      i += bestPatternLen;
    } else {
      // No rule matched — skip character
      i++;
    }
  }

  return result;
}

/**
 * Convert a word to IPA using the rule engine.
 */
export function wordToIPA(word: string): string {
  const espeak = applyRulesToWord(word);
  return espeakToIPA(espeak);
}

// ── Initialization ──

let _rulesLoaded = false;
let _rulesText: string | null = null;

/**
 * Initialize the rule engine with the rules text.
 * Call this once with the contents of en_rules.
 */
export function initRules(rulesText: string): void {
  _rulesText = rulesText;
  parseRules(rulesText);
  _rulesLoaded = true;
}

export function isRulesLoaded(): boolean {
  return _rulesLoaded;
}
