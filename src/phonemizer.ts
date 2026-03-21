/**
 * Browser-side phonemization for Kitten TTS.
 *
 * Uses a pre-computed dictionary of 6000+ words generated from espeak-ng (en-us).
 * Unknown words are phonemized letter-by-letter using a basic English letter-to-phoneme map.
 */

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
 * and wraps with [0] start/end tokens.
 */
export function phonemesToInputIds(phonemes: string): number[] {
  // Split into "words" (including Unicode letters like IPA ə, ɪ, ʊ) and punctuation.
  // CRITICAL: Must use \p{L} (Unicode letter class) instead of \w, because
  // JS \w only matches [a-zA-Z0-9_] and misses IPA characters like ə, ɪ, ʊ, ɛ, etc.
  // Python's \w matches Unicode letters by default, so the reference pipeline
  // groups "həl" as one token, but JS \w would split it as "h", "ə", "l".
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
  // End sequence: … ($) — matches official kittentts package
  // The … (index 10) signals end-of-utterance to the model
  ids.push(10); // … (ellipsis)
  ids.push(0);  // $ (stop)

  return ids;
}

/**
 * Basic letter-to-phoneme fallback for unknown words.
 * This is a rough approximation — not accurate, but better than passing raw ASCII.
 */
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
 */
export function textToPhonemes(text: string): string {
  // Ensure text ends with punctuation (matches official kittentts package behavior)
  let normalized = text.trim();
  if (normalized && !/[.!?,;:]$/.test(normalized)) {
    normalized += ',';
  }
  const tokens = normalized.toLowerCase().match(/[\w']+|[^\w\s]/g) || [];
  return tokens.map(w => {
    // Try exact match first
    if (PHONEME_DICT[w]) return PHONEME_DICT[w];
    // Try without trailing 's' (basic plural handling)
    if (w.endsWith('s') && PHONEME_DICT[w.slice(0, -1)]) {
      return PHONEME_DICT[w.slice(0, -1)] + 'z';
    }
    // Try without 'ed' suffix
    if (w.endsWith('ed') && PHONEME_DICT[w.slice(0, -2)]) {
      return PHONEME_DICT[w.slice(0, -2)] + 'd';
    }
    // Try without 'ing' suffix
    if (w.endsWith('ing') && PHONEME_DICT[w.slice(0, -3)]) {
      return PHONEME_DICT[w.slice(0, -3)] + 'ɪŋ';
    }
    if (w.endsWith('ing') && PHONEME_DICT[w.slice(0, -3) + 'e']) {
      return PHONEME_DICT[w.slice(0, -3) + 'e'].replace(/[ə]$/, '') + 'ɪŋ';
    }
    // Try without 'ly' suffix
    if (w.endsWith('ly') && PHONEME_DICT[w.slice(0, -2)]) {
      return PHONEME_DICT[w.slice(0, -2)] + 'li';
    }
    // Punctuation passes through
    if (/^[^\w]+$/.test(w)) return w;
    // Letter-by-letter fallback
    return letterFallback(w);
  }).join(' ');
}

export function textToInputIds(text: string): number[] {
  return phonemesToInputIds(textToPhonemes(text));
}
