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
  // End sequence: … ($) — matches official kittentts package
  ids.push(10); // … (ellipsis)
  ids.push(0);  // $ (stop)

  return ids;
}

/**
 * Convert English text to IPA phonemes using espeak-ng WASM.
 * This matches the official kittentts package phonemization exactly.
 */
export async function textToPhonemesEspeak(text: string): Promise<string> {
  // Ensure text ends with punctuation (matches official kittentts package)
  let normalized = text.trim();
  if (normalized && !/[.!?,;:]$/.test(normalized)) {
    normalized += ',';
  }

  // phonemize() returns an array (one entry per input sentence)
  const result = await phonemize(normalized, 'en-us');
  return result[0] || '';
}

/**
 * Convert English text to input_ids using espeak-ng WASM (async).
 * This is the primary phonemization path — matches official kittentts exactly.
 */
export async function textToInputIds(text: string): Promise<number[]> {
  const phonemes = await textToPhonemesEspeak(text);
  return phonemesToInputIds(phonemes);
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
