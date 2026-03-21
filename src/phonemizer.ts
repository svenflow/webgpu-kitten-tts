/**
 * Browser-side phonemization for Kitten TTS.
 *
 * Primary: espeak-ng via WASM (phonemizer.js by xenova) — matches official kittentts package.
 * Fallback: pre-computed dictionary of 6000+ words (for offline/no-WASM environments).
 */

import { phonemize } from 'phonemizer';
import { DEFAULT_CONFIG } from './types.js';
import { PHONEME_DICT } from './phoneme-dict.js';

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

/**
 * Convert English text to input_ids.
 * Primary: espeak-ng WASM (exact match with official kittentts).
 * Fallback: dictionary phonemizer (for Safari where WASM fails).
 */
export async function textToInputIds(text: string): Promise<{ ids: number[]; method: 'wasm' | 'dictionary' }> {
  try {
    const phonemes = await textToPhonemesEspeak(text);
    if (phonemes) {
      return { ids: phonemesToInputIds(phonemes), method: 'wasm' };
    }
  } catch (e) {
    console.warn('espeak-ng WASM failed, falling back to dictionary phonemizer:', e);
  }
  // Fallback for Safari / environments where WASM doesn't load
  const phonemes = textToPhonemesDictionary(text);
  return { ids: phonemesToInputIds(phonemes), method: 'dictionary' };
}

// ── Dictionary fallback (kept for reference/offline use) ──

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
 * Convert English text to IPA phonemes using the pre-computed dictionary.
 * Falls back to letter-by-letter phonemization for unknown words.
 * @deprecated Use textToPhonemesEspeak() instead for accurate phonemization.
 */
export function textToPhonemesDictionary(text: string): string {
  let normalized = text.trim();
  if (normalized && !/[.!?,;:]$/.test(normalized)) {
    normalized += ',';
  }
  const tokens = normalized.toLowerCase().match(/[\w']+|[^\w\s]/g) || [];
  return tokens.map(w => {
    if (PHONEME_DICT[w]) return PHONEME_DICT[w];
    if (w.endsWith('s') && PHONEME_DICT[w.slice(0, -1)]) {
      return PHONEME_DICT[w.slice(0, -1)] + 'z';
    }
    if (w.endsWith('ed') && PHONEME_DICT[w.slice(0, -2)]) {
      return PHONEME_DICT[w.slice(0, -2)] + 'd';
    }
    if (w.endsWith('ing') && PHONEME_DICT[w.slice(0, -3)]) {
      return PHONEME_DICT[w.slice(0, -3)] + 'ɪŋ';
    }
    if (w.endsWith('ing') && PHONEME_DICT[w.slice(0, -3) + 'e']) {
      return PHONEME_DICT[w.slice(0, -3) + 'e'].replace(/[ə]$/, '') + 'ɪŋ';
    }
    if (w.endsWith('ly') && PHONEME_DICT[w.slice(0, -2)]) {
      return PHONEME_DICT[w.slice(0, -2)] + 'li';
    }
    if (/^[^\w]+$/.test(w)) return w;
    return letterFallback(w);
  }).join(' ');
}
