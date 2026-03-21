/**
 * Browser-side phonemization for Kitten TTS.
 *
 * Primary: espeak-ng via WASM (phonemizer.js by xenova) — matches official kittentts package.
 * Fallback: 234K-word espeak-ng dictionary + rule engine for unknown words.
 */

import { phonemize } from 'phonemizer';
import { DEFAULT_CONFIG } from './types.js';
import { wordToIPA, initRules, isRulesLoaded } from './espeak-rules.js';

/** Symbol to index mapping for the 178-symbol phoneme table. */
const symbolToIndex: Map<string, number> = new Map();
DEFAULT_CONFIG.symbols.forEach((s, i) => symbolToIndex.set(s, i));

/**
 * Convert pre-phonemized text (IPA string) to input_ids.
 *
 * The phonemizer splits tokens with regex: /\w+|[^\w\s]/g
 * then joins with spaces, looks up each char in symbol table,
 * and wraps with start/end tokens.
 */
export function phonemesToInputIds(phonemes: string): number[] {
  // Split into "words" (including Unicode letters like IPA ə, ɪ, ʊ) and punctuation.
  // CRITICAL: Must use \p{L} (Unicode letter class) instead of \w, because
  // JS \w only matches [a-zA-Z0-9_] and misses IPA characters like ə, ɪ, ʊ, ɛ, etc.
  const tokens = phonemes.match(/[\p{L}\p{N}_]+|[^\p{L}\p{N}_\s]/gu) || [];
  const joined = tokens.join(' ');

  // Map characters to indices, skip unknown
  const ids: number[] = [0]; // Start token ($)
  for (const char of joined) {
    const idx = symbolToIndex.get(char);
    if (idx !== undefined) {
      ids.push(idx);
    }
  }
  // End token — matches official kittentts package: [0, ...ids..., 0]
  ids.push(0);  // $ (stop)

  return ids;
}

/** Punctuation chars recognized by the kittentts symbol table */
const PUNCT_RE = /[;:,.!?¡¿—…""«»""]/g;

/**
 * Convert English text to IPA phonemes using espeak-ng WASM.
 * Preserves punctuation to match official kittentts (which uses
 * phonemizer.backend.EspeakBackend with preserve_punctuation=True).
 *
 * espeak-ng strips punctuation, so we:
 * 1. Extract punctuation + positions from the original text
 * 2. Phonemize the stripped text
 * 3. Re-insert punctuation at their original word boundaries
 */
export async function textToPhonemesEspeak(text: string): Promise<string> {
  const normalized = text.trim();
  if (!normalized) return '';

  // Split text into alternating (words, punct) segments
  // e.g. "Hello, world!" → ["Hello", ",", " world", "!"]
  const segments: { text: string; isPunct: boolean }[] = [];
  let last = 0;
  const punctPositions: { char: string; afterWordIdx: number }[] = [];

  // Simple approach: strip punctuation, phonemize, then append punctuation
  // that appeared at sentence-final position
  // This matches the Python phonemizer's preserve_punctuation behavior
  // for the simple case (punctuation at word boundaries)

  // Extract all punctuation with their positions relative to words
  const words: string[] = [];
  const puncts: string[] = [];
  // Split by whitespace, tracking punctuation attached to words
  const rawTokens = normalized.match(/\S+/g) || [];

  const cleanWords: string[] = [];
  const trailingPuncts: string[] = [];

  for (const token of rawTokens) {
    // Strip trailing punctuation from each word
    let word = token;
    let trailing = '';
    while (word.length > 0 && /[;:,.!?¡¿—…""«»""]/.test(word[word.length - 1])) {
      trailing = word[word.length - 1] + trailing;
      word = word.slice(0, -1);
    }
    // Also strip leading punctuation
    let leading = '';
    while (word.length > 0 && /[;:,.!?¡¿—…""«»""]/.test(word[0])) {
      leading += word[0];
      word = word.slice(1);
    }
    cleanWords.push(word);
    trailingPuncts.push(leading + trailing);
  }

  // Phonemize the cleaned text (without punctuation)
  const cleanText = cleanWords.filter(w => w.length > 0).join(' ');
  const result = await phonemize(cleanText, 'en-us');
  const phonemized = result[0] || '';

  if (!phonemized) return '';

  // Split phonemized output into words to re-insert punctuation
  const phonemeWords = phonemized.split(/\s+/);

  // Re-insert punctuation after corresponding phoneme words
  // The phonemizer may merge/split words differently, so we append
  // all trailing punctuation at the end if word counts don't match
  const outputParts: string[] = [];
  let pIdx = 0;
  for (let i = 0; i < cleanWords.length; i++) {
    if (cleanWords[i].length > 0 && pIdx < phonemeWords.length) {
      outputParts.push(phonemeWords[pIdx]);
      pIdx++;
    }
    if (trailingPuncts[i]) {
      outputParts.push(trailingPuncts[i]);
    }
  }
  // Append any remaining phoneme words
  while (pIdx < phonemeWords.length) {
    outputParts.push(phonemeWords[pIdx]);
    pIdx++;
  }

  return outputParts.join(' ');
}

// ── Large dictionary + rule engine fallback ──

/**
 * Common function words in their unstressed/connected-speech forms.
 * espeak-ng reduces these when they appear in connected speech (not standalone).
 * Without this, the dictionary returns citation forms with full stress.
 */
const FUNCTION_WORD_CONNECTED: Record<string, string> = {
  'a': 'ɐ',
  'an': 'ɐn',
  'the': 'ðə',
  'to': 'tə',
  'of': 'ʌv',
  'in': 'ɪn',
  'on': 'ɔn',
  'at': 'æt',
  'by': 'baɪ',
  'for': 'fɔːɹ',
  'or': 'ɔːɹ',
  'and': 'ænd',
  'but': 'bʌt',
  'that': 'ðæt',
  'this': 'ðɪs',
  'these': 'ðiːz',
  'those': 'ðoʊz',
  'i': 'aɪ',
  'you': 'juː',
  'he': 'hiː',
  'she': 'ʃiː',
  'we': 'wiː',
  'it': 'ɪt',
  'is': 'ɪz',
  'was': 'wʌz',
  'are': 'ɑːɹ',
  'were': 'wɜː',
  'be': 'biː',
  'been': 'bɪn',
  'have': 'hæv',
  'has': 'hæz',
  'had': 'hæd',
  'do': 'duː',
  'does': 'dʌz',
  'did': 'dɪd',
  'will': 'wɪl',
  'would': 'wʊd',
  'could': 'kʊd',
  'should': 'ʃʊd',
  'can': 'kæn',
  'may': 'meɪ',
  'might': 'maɪt',
  'must': 'mʌst',
  'shall': 'ʃæl',
  'not': 'nˌɑːt',
  'no': 'nˈoʊ',
  'if': 'ɪf',
  'how': 'hˌaʊ',
  'with': 'wɪð',
  'from': 'fɹʌm',
  'your': 'jʊɹ',
  'my': 'maɪ',
  'his': 'hɪz',
  'her': 'hɜː',
  'its': 'ɪts',
  'our': 'aʊɚ',
  'their': 'ðɛɹ',
  'some': 'sˌʌm',
  'new': 'nˈuː',
  'all': 'ˈɔːl',
};

/** Lazily-loaded dictionary: word → IPA phonemes */
let dictMap: Map<string, string> | null = null;
let dictLoadPromise: Promise<Map<string, string>> | null = null;

/**
 * Load the large espeak-ng dictionary from the TSV file.
 * Returns a Map of word → IPA phonemes.
 */
async function loadDictionary(): Promise<Map<string, string>> {
  if (dictMap) return dictMap;
  if (dictLoadPromise) return dictLoadPromise;

  dictLoadPromise = (async () => {
    const map = new Map<string, string>();
    try {
      const resp = await fetch('./espeak-en-dict.tsv');
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const text = await resp.text();
      const lines = text.split('\n');
      for (const line of lines) {
        if (!line) continue;
        const tab = line.indexOf('\t');
        if (tab < 0) continue;
        const word = line.substring(0, tab);
        const phonemes = line.substring(tab + 1);
        map.set(word, phonemes);
      }
      console.log(`Loaded espeak dictionary: ${map.size} entries`);
    } catch (e) {
      console.warn('Failed to load espeak-en-dict.tsv:', e);
    }
    dictMap = map;
    return map;
  })();

  return dictLoadPromise;
}

/** Lazily-loaded rules */
let rulesLoadPromise: Promise<void> | null = null;

async function ensureRulesLoaded(): Promise<void> {
  if (isRulesLoaded()) return;
  if (rulesLoadPromise) return rulesLoadPromise;

  rulesLoadPromise = (async () => {
    try {
      const resp = await fetch('./en_rules');
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const text = await resp.text();
      initRules(text);
      console.log('Loaded espeak en_rules');
    } catch (e) {
      console.warn('Failed to load en_rules:', e);
    }
  })();

  return rulesLoadPromise;
}

/**
 * Convert English text to IPA phonemes using the large dictionary + rule engine.
 * Preserves punctuation (matching espeak preserve_punctuation=True).
 *
 * espeak-ng with preserve_punctuation=True attaches punctuation directly to
 * the adjacent phoneme word (no space before sentence-final punctuation).
 */
export async function textToPhonemesDictRules(text: string): Promise<string> {
  const [dict] = await Promise.all([loadDictionary(), ensureRulesLoaded()]);

  const normalized = text.trim();
  if (!normalized) return '';

  // Split into whitespace-separated tokens, then strip punctuation per-token
  // (same approach as textToPhonemesEspeak for consistency)
  const rawTokens = normalized.match(/\S+/g) || [];

  const cleanWords: string[] = [];
  const trailingPuncts: string[] = [];
  const leadingPuncts: string[] = [];

  for (const token of rawTokens) {
    let word = token;
    let trailing = '';
    while (word.length > 0 && /[;:,.!?¡¿—…""«»""]/.test(word[word.length - 1])) {
      trailing = word[word.length - 1] + trailing;
      word = word.slice(0, -1);
    }
    let leading = '';
    while (word.length > 0 && /[;:,.!?¡¿—…""«»""]/.test(word[0])) {
      leading += word[0];
      word = word.slice(1);
    }
    cleanWords.push(word);
    trailingPuncts.push(trailing);
    leadingPuncts.push(leading);
  }

  // Phonemize each word, then re-insert punctuation
  const outputParts: string[] = [];
  const wordCount = cleanWords.filter(w => w.length > 0).length;
  for (let i = 0; i < cleanWords.length; i++) {
    let part = '';
    if (leadingPuncts[i]) part += leadingPuncts[i];
    if (cleanWords[i].length > 0) {
      const lower = cleanWords[i].toLowerCase();
      // Use connected-speech forms for function words in multi-word context
      if (wordCount > 1 && lower in FUNCTION_WORD_CONNECTED) {
        part += FUNCTION_WORD_CONNECTED[lower];
      } else {
        part += lookupWord(lower, dict, cleanWords[i]);
      }
    }
    if (trailingPuncts[i]) part += trailingPuncts[i];
    if (part) outputParts.push(part);
  }

  return outputParts.join(' ');
}

/**
 * Look up a word in the dictionary, with morphological fallbacks,
 * then use the rule engine for completely unknown words.
 */
function lookupWord(word: string, dict: Map<string, string>, originalWord?: string): string {
  // Direct lookup
  if (dict.has(word)) return dict.get(word)!;

  // Try without trailing 's (possessive)
  if (word.endsWith("'s")) {
    const base = word.slice(0, -2);
    if (dict.has(base)) return dict.get(base)! + 'z';
  }

  // Try common suffixes
  if (word.endsWith('s') && word.length > 2) {
    const base = word.slice(0, -1);
    if (dict.has(base)) {
      // Determine s/z voicing
      const lastChar = base[base.length - 1];
      const voiced = 'bdgjlmnrvwz'.includes(lastChar) || 'aeiou'.includes(lastChar);
      return dict.get(base)! + (voiced ? 'z' : 's');
    }
    // Try -es → -e
    if (word.endsWith('es')) {
      const base2 = word.slice(0, -1); // remove just 's', keep 'e'
      if (dict.has(base2)) return dict.get(base2)! + 'z';
    }
  }

  if (word.endsWith('ed') && word.length > 3) {
    const base = word.slice(0, -2);
    if (dict.has(base)) return dict.get(base)! + 'd';
    // Try -ed where base ends in e (e.g., "captured" → "capture")
    const baseE = word.slice(0, -1); // "captured" → "capture" doesn't work, try base + 'e'
    if (dict.has(base + 'e')) return dict.get(base + 'e')!.replace(/[ə]$/, '') + 'd';
  }

  if (word.endsWith('ing') && word.length > 4) {
    const base = word.slice(0, -3);
    if (dict.has(base)) return dict.get(base)! + 'ɪŋ';
    if (dict.has(base + 'e')) return dict.get(base + 'e')!.replace(/[ə]$/, '') + 'ɪŋ';
    // Doubled consonant: "running" → "run"
    if (base.length > 1 && base[base.length - 1] === base[base.length - 2]) {
      const dedup = base.slice(0, -1);
      if (dict.has(dedup)) return dict.get(dedup)! + 'ɪŋ';
    }
  }

  if (word.endsWith('ly') && word.length > 3) {
    const base = word.slice(0, -2);
    if (dict.has(base)) return dict.get(base)! + 'li';
  }

  if (word.endsWith('er') && word.length > 3) {
    const base = word.slice(0, -2);
    if (dict.has(base)) return dict.get(base)! + 'ɚ';
    if (dict.has(base + 'e')) return dict.get(base + 'e')!.replace(/[ə]$/, '') + 'ɚ';
  }

  if (word.endsWith('est') && word.length > 4) {
    const base = word.slice(0, -3);
    if (dict.has(base)) return dict.get(base)! + 'ɪst';
  }

  if (word.endsWith('ness') && word.length > 5) {
    const base = word.slice(0, -4);
    if (dict.has(base)) return dict.get(base)! + 'nəs';
  }

  // Hyphenated compound words
  if (word.includes('-')) {
    const parts = word.split('-');
    const phonemeParts = parts.map(p => {
      // Use connected forms for function words inside compounds
      if (p in FUNCTION_WORD_CONNECTED) return FUNCTION_WORD_CONNECTED[p];
      return lookupWord(p, dict);
    });
    return phonemeParts.join('');
  }

  // All-uppercase abbreviations: spell them out (e.g., "TTS" → "tˌiːtˌiːˈɛs")
  const orig = originalWord || word;
  if (orig.length >= 2 && orig === orig.toUpperCase() && /^[A-Z]+$/.test(orig)) {
    const spelled = word.split('').map(c => LETTER_NAMES[c] || c).join('');
    if (spelled !== word) return spelled;
  }

  // Try compound word splitting (e.g., "seashells" → "sea" + "shells")
  const compound = trySplitCompound(word, dict);
  if (compound) return compound;

  // Fall back to rule engine
  if (isRulesLoaded()) {
    return wordToIPA(word);
  }

  // Last resort: letter-by-letter
  return letterFallback(word);
}

/** Letter names as spoken by espeak-ng (for abbreviation spelling) */
const LETTER_NAMES: Record<string, string> = {
  'a': 'ˈeɪ', 'b': 'bˈiː', 'c': 'sˈiː', 'd': 'dˈiː', 'e': 'ˈiː',
  'f': 'ˈɛf', 'g': 'dʒˈiː', 'h': 'ˈeɪtʃ', 'i': 'ˈaɪ', 'j': 'dʒˈeɪ',
  'k': 'kˈeɪ', 'l': 'ˈɛl', 'm': 'ˈɛm', 'n': 'ˈɛn', 'o': 'ˈoʊ',
  'p': 'pˈiː', 'q': 'kjˈuː', 'r': 'ˈɑːɹ', 's': 'ˈɛs', 't': 'tˈiː',
  'u': 'jˈuː', 'v': 'vˈiː', 'w': 'dˈʌbəljˌuː', 'x': 'ˈɛks', 'y': 'wˈaɪ',
  'z': 'zˈiː',
};

/**
 * Try splitting a word into two known dictionary parts (compound word).
 * Returns the combined phonemes if a valid split is found, null otherwise.
 */
function trySplitCompound(word: string, dict: Map<string, string>): string | null {
  // Try splits from longest prefix to shortest (minimum 2 chars each part)
  for (let i = word.length - 2; i >= 2; i--) {
    const prefix = word.slice(0, i);
    const suffix = word.slice(i);
    if (dict.has(prefix) && dict.has(suffix)) {
      return dict.get(prefix)! + dict.get(suffix)!;
    }
    // Try suffix with common inflections
    if (suffix.endsWith('s') && dict.has(suffix.slice(0, -1))) {
      if (dict.has(prefix)) {
        const base = dict.get(suffix.slice(0, -1))!;
        return dict.get(prefix)! + base + 'z';
      }
    }
  }
  return null;
}

const LETTER_PHONEMES: Record<string, string> = {
  'a': 'æ', 'b': 'b', 'c': 'k', 'd': 'd', 'e': 'ɛ', 'f': 'f',
  'g': 'ɡ', 'h': 'h', 'i': 'ɪ', 'j': 'dʒ', 'k': 'k', 'l': 'l',
  'm': 'm', 'n': 'n', 'o': 'ɑː', 'p': 'p', 'q': 'k', 'r': 'ɹ',
  's': 's', 't': 't', 'u': 'ʌ', 'v': 'v', 'w': 'w', 'x': 'ks',
  'y': 'j', 'z': 'z',
};

function letterFallback(word: string): string {
  return word.split('').map(c => LETTER_PHONEMES[c] || c).join('');
}

/**
 * Convert English text to input_ids.
 * Primary: espeak-ng WASM (exact match with official kittentts).
 * Fallback: large dictionary + rule engine (for Safari where WASM fails).
 */
export async function textToInputIds(text: string): Promise<{ ids: number[]; method: 'wasm' | 'dictionary' }> {
  try {
    // Race WASM phonemizer against a timeout — on Safari/iOS the WASM worker
    // can hang indefinitely without throwing, so we need a hard cutoff.
    const phonemes = await Promise.race([
      textToPhonemesEspeak(text),
      new Promise<null>((_, reject) =>
        setTimeout(() => reject(new Error('WASM phonemizer timeout (5s)')), 5000)
      ),
    ]);
    if (phonemes) {
      return { ids: phonemesToInputIds(phonemes), method: 'wasm' };
    }
  } catch (e) {
    console.warn('espeak-ng WASM failed, falling back to dictionary phonemizer:', e);
  }
  // Fallback for Safari / environments where WASM doesn't load
  const phonemes = await textToPhonemesDictRules(text);
  return { ids: phonemesToInputIds(phonemes), method: 'dictionary' };
}
