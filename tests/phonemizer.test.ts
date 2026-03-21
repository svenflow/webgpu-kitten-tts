/**
 * Phonemizer correctness and performance tests.
 *
 * Tests the dictionary + rule engine fallback phonemizer against
 * espeak-ng reference output (en-us, preserve_punctuation=True, with_stress=True).
 *
 * Reference phonemes were generated with:
 *   phonemizer v3.3.0, espeak-ng v1.51 (en-us)
 *
 * Usage:
 *   npx tsx tests/phonemizer.test.ts
 */

import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { initRules, wordToIPA, isRulesLoaded } from '../src/espeak-rules.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, '..');

// ── Reference data ──
// Each entry: { text, phonemes } where phonemes is the espeak-ng output.
const REFERENCE: { text: string; phonemes: string }[] = [
  { text: "The quick brown fox jumps over the lazy dog.", phonemes: "ðə kwˈɪk bɹˈaʊn fˈɑːks dʒˈʌmps ˌoʊvɚ ðə lˈeɪzi dˈɑːɡ." },
  { text: "Hello world, how are you today?", phonemes: "həlˈoʊ wˈɜːld, hˌaʊ ɑːɹ juː tədˈeɪ?" },
  { text: "She sells seashells by the seashore.", phonemes: "ʃiː sˈɛlz sˈiːʃɛlz baɪ ðə sˈiːʃɔːɹ." },
  { text: "Peter Piper picked a peck of pickled peppers.", phonemes: "pˈiːɾɚ pˈaɪpɚ pˈɪkt ɐ pˈɛk ʌv pˈɪkəld pˈɛpɚz." },
  { text: "I scream, you scream, we all scream for ice cream.", phonemes: "aɪ skɹˈiːm, juː skɹˈiːm, wiː ˈɔːl skɹˈiːm fɔːɹ ˈaɪs kɹˈiːm." },
  { text: "The rain in Spain stays mainly in the plain.", phonemes: "ðə ɹˈeɪn ɪn spˈeɪn stˈeɪz mˈeɪnli ɪnðə plˈeɪn." },
  { text: "To be or not to be, that is the question.", phonemes: "təbi ɔːɹ nˌɑːt tə bˈiː, ðæt ɪz ðə kwˈɛstʃən." },
  { text: "A stitch in time saves nine.", phonemes: "ɐ stˈɪtʃ ɪn tˈaɪm sˈeɪvz nˈaɪn." },
  { text: "Knowledge is power.", phonemes: "nˈɑːlɪdʒ ɪz pˈaʊɚ." },
  { text: "The cat sat on the mat.", phonemes: "ðə kˈæt sˈæt ɔnðə mˈæt." },
  { text: "Beautiful flowers bloom in spring.", phonemes: "bjˈuːɾifəl flˈaʊɚz blˈuːm ɪn spɹˈɪŋ." },
  { text: "The children played happily in the garden.", phonemes: "ðə tʃˈɪldɹən plˈeɪd hˈæpɪli ɪnðə ɡˈɑːɹdən." },
  { text: "Technology changes rapidly these days.", phonemes: "tɛknˈɑːlədʒi tʃˈeɪndʒᵻz ɹˈæpɪdli ðiːz dˈeɪz." },
  { text: "Would you like some coffee or tea?", phonemes: "wʊd juː lˈaɪk sˌʌm kˈɔfi ɔːɹ tˈiː?" },
  { text: "The mountain was covered in snow.", phonemes: "ðə mˈaʊntɪn wʌz kˈʌvɚd ɪn snˈoʊ." },
  { text: "Calm waters run deep.", phonemes: "kˈɑːm wˈɔːɾɚz ɹˈʌn dˈiːp." },
  { text: "The photographer captured a breathtaking sunset.", phonemes: "ðə fətˈɑːɡɹəfɚ kˈæptʃɚd ɐ bɹˈɛθteɪkɪŋ sˈʌnsɛt." },
  { text: "Music brings people together.", phonemes: "mjˈuːzɪk bɹˈɪŋz pˈiːpəl təɡˈɛðɚ." },
  { text: "The university announced new scholarships.", phonemes: "ðə jˌuːnɪvˈɜːsᵻɾi ɐnˈaʊnst nˈuː skˈɑːlɚʃˌɪps." },
  { text: "Extraordinary circumstances require extraordinary measures.", phonemes: "ɛkstɹˈɔːɹdɪnˌɛɹi sˈɜːkəmstˌænsᵻz ɹᵻkwˈaɪɚɹ ɛkstɹˈɔːɹdɪnˌɛɹi mˈɛʒɚz." },
  { text: "Kitten TTS is an open-source text-to-speech model that runs entirely in your browser using WebGPU.", phonemes: "kˈɪʔn̩ tˌiːtˌiːˈɛs ɪz ɐn ˈoʊpənsˈɔːɹs tˈɛksttəspˈiːtʃ mˈɑːdəl ðæt ɹˈʌnz ɛntˈaɪɚli ɪn jʊɹ bɹˈaʊzɚ jˈuːzɪŋ wˈɛb dʒˌiːpˌiːjˈuː." },
  { text: "It generates natural-sounding speech in real time, with multiple voice options and adjustable speed.", phonemes: "ɪt dʒˈɛnɚɹˌeɪts nˈætʃɚɹəlsˈaʊndɪŋ spˈiːtʃ ɪn ɹˈiːəl tˈaɪm, wɪð mˌʌltɪpəl vˈɔɪs ˈɑːpʃənz ænd ɐdʒˈʌstəbəl spˈiːd." },
  { text: "No server required — your audio never leaves your device.", phonemes: "nˈoʊ sˈɜːvɚ ɹᵻkwˈaɪɚd — jʊɹ ˈɔːdɪˌoʊ nˈɛvɚ lˈiːvz jʊɹ dɪvˈaɪs." },
];

// ── Load dictionary ──
const dictMap = new Map<string, string>();
const dictTsv = readFileSync(resolve(ROOT, 'public/espeak-en-dict.tsv'), 'utf-8');
for (const line of dictTsv.split('\n')) {
  if (!line) continue;
  const tab = line.indexOf('\t');
  if (tab < 0) continue;
  dictMap.set(line.substring(0, tab), line.substring(tab + 1));
}

// ── Load rules ──
const rulesText = readFileSync(resolve(ROOT, 'public/en_rules'), 'utf-8');
initRules(rulesText);

// ── Connected-speech function word forms ──
// espeak-ng reduces common function words in multi-word context.
const FUNCTION_WORD_CONNECTED: Record<string, string> = {
  'a': 'ɐ', 'an': 'ɐn', 'the': 'ðə', 'to': 'tə', 'of': 'ʌv',
  'in': 'ɪn', 'on': 'ɔn', 'at': 'æt', 'by': 'baɪ', 'for': 'fɔːɹ',
  'or': 'ɔːɹ', 'and': 'ænd', 'but': 'bʌt', 'that': 'ðæt',
  'this': 'ðɪs', 'these': 'ðiːz', 'those': 'ðoʊz',
  'i': 'aɪ', 'you': 'juː', 'he': 'hiː', 'she': 'ʃiː', 'we': 'wiː',
  'it': 'ɪt', 'is': 'ɪz', 'was': 'wʌz', 'are': 'ɑːɹ', 'were': 'wɜː',
  'be': 'biː', 'been': 'bɪn', 'have': 'hæv', 'has': 'hæz', 'had': 'hæd',
  'do': 'duː', 'does': 'dʌz', 'did': 'dɪd', 'will': 'wɪl',
  'would': 'wʊd', 'could': 'kʊd', 'should': 'ʃʊd', 'can': 'kæn',
  'may': 'meɪ', 'might': 'maɪt', 'must': 'mʌst', 'shall': 'ʃæl',
  'not': 'nˌɑːt', 'no': 'nˈoʊ', 'if': 'ɪf', 'how': 'hˌaʊ',
  'with': 'wɪð', 'from': 'fɹʌm', 'your': 'jʊɹ', 'my': 'maɪ',
  'his': 'hɪz', 'her': 'hɜː', 'its': 'ɪts', 'our': 'aʊɚ',
  'their': 'ðɛɹ', 'some': 'sˌʌm', 'new': 'nˈuː', 'all': 'ˈɔːl',
};

// ── Letter names for abbreviation spelling ──
const LETTER_NAMES: Record<string, string> = {
  'a': 'ˈeɪ', 'b': 'bˈiː', 'c': 'sˈiː', 'd': 'dˈiː', 'e': 'ˈiː',
  'f': 'ˈɛf', 'g': 'dʒˈiː', 'h': 'ˈeɪtʃ', 'i': 'ˈaɪ', 'j': 'dʒˈeɪ',
  'k': 'kˈeɪ', 'l': 'ˈɛl', 'm': 'ˈɛm', 'n': 'ˈɛn', 'o': 'ˈoʊ',
  'p': 'pˈiː', 'q': 'kjˈuː', 'r': 'ˈɑːɹ', 's': 'ˈɛs', 't': 'tˈiː',
  'u': 'jˈuː', 'v': 'vˈiː', 'w': 'dˈʌbəljˌuː', 'x': 'ˈɛks', 'y': 'wˈaɪ',
  'z': 'zˈiː',
};

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

function trySplitCompound(word: string): string | null {
  for (let i = word.length - 2; i >= 2; i--) {
    const prefix = word.slice(0, i);
    const suffix = word.slice(i);
    if (dictMap.has(prefix) && dictMap.has(suffix)) {
      return dictMap.get(prefix)! + dictMap.get(suffix)!;
    }
    if (suffix.endsWith('s') && dictMap.has(suffix.slice(0, -1))) {
      if (dictMap.has(prefix)) {
        return dictMap.get(prefix)! + dictMap.get(suffix.slice(0, -1))! + 'z';
      }
    }
  }
  return null;
}

function lookupWord(word: string, originalWord?: string): string {
  if (dictMap.has(word)) return dictMap.get(word)!;

  if (word.endsWith("'s")) {
    const base = word.slice(0, -2);
    if (dictMap.has(base)) return dictMap.get(base)! + 'z';
  }

  if (word.endsWith('s') && word.length > 2) {
    const base = word.slice(0, -1);
    if (dictMap.has(base)) {
      const lastChar = base[base.length - 1];
      const voiced = 'bdgjlmnrvwz'.includes(lastChar) || 'aeiou'.includes(lastChar);
      return dictMap.get(base)! + (voiced ? 'z' : 's');
    }
    if (word.endsWith('es')) {
      const base2 = word.slice(0, -1);
      if (dictMap.has(base2)) return dictMap.get(base2)! + 'z';
    }
  }

  if (word.endsWith('ed') && word.length > 3) {
    const base = word.slice(0, -2);
    if (dictMap.has(base)) return dictMap.get(base)! + 'd';
    if (dictMap.has(base + 'e')) return dictMap.get(base + 'e')!.replace(/[ə]$/, '') + 'd';
  }

  if (word.endsWith('ing') && word.length > 4) {
    const base = word.slice(0, -3);
    if (dictMap.has(base)) return dictMap.get(base)! + 'ɪŋ';
    if (dictMap.has(base + 'e')) return dictMap.get(base + 'e')!.replace(/[ə]$/, '') + 'ɪŋ';
    if (base.length > 1 && base[base.length - 1] === base[base.length - 2]) {
      const dedup = base.slice(0, -1);
      if (dictMap.has(dedup)) return dictMap.get(dedup)! + 'ɪŋ';
    }
  }

  if (word.endsWith('ly') && word.length > 3) {
    const base = word.slice(0, -2);
    if (dictMap.has(base)) return dictMap.get(base)! + 'li';
  }

  if (word.endsWith('er') && word.length > 3) {
    const base = word.slice(0, -2);
    if (dictMap.has(base)) return dictMap.get(base)! + 'ɚ';
    if (dictMap.has(base + 'e')) return dictMap.get(base + 'e')!.replace(/[ə]$/, '') + 'ɚ';
  }

  if (word.endsWith('est') && word.length > 4) {
    const base = word.slice(0, -3);
    if (dictMap.has(base)) return dictMap.get(base)! + 'ɪst';
  }

  if (word.endsWith('ness') && word.length > 5) {
    const base = word.slice(0, -4);
    if (dictMap.has(base)) return dictMap.get(base)! + 'nəs';
  }

  // Hyphenated compound words
  if (word.includes('-')) {
    const parts = word.split('-');
    return parts.map(p => {
      if (p in FUNCTION_WORD_CONNECTED) return FUNCTION_WORD_CONNECTED[p];
      return lookupWord(p, p);
    }).join('');
  }

  // Abbreviations (all uppercase in original)
  const orig = originalWord || word;
  if (orig.length >= 2 && orig === orig.toUpperCase() && /^[A-Z]+$/.test(orig)) {
    const spelled = word.split('').map(c => LETTER_NAMES[c] || c).join('');
    if (spelled !== word) return spelled;
  }

  // Compound word splitting
  const compound = trySplitCompound(word);
  if (compound) return compound;

  // Rule engine
  if (isRulesLoaded()) return wordToIPA(word);

  return letterFallback(word);
}

/**
 * Phonemize a full sentence using the dictionary + rule engine.
 * This mirrors the browser-side textToPhonemesDictRules() function.
 */
function phonemize(text: string): string {
  const normalized = text.trim();
  if (!normalized) return '';

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

  const wordCount = cleanWords.filter(w => w.length > 0).length;
  const outputParts: string[] = [];
  for (let i = 0; i < cleanWords.length; i++) {
    let part = '';
    if (leadingPuncts[i]) part += leadingPuncts[i];
    if (cleanWords[i].length > 0) {
      const lower = cleanWords[i].toLowerCase();
      if (wordCount > 1 && lower in FUNCTION_WORD_CONNECTED) {
        part += FUNCTION_WORD_CONNECTED[lower];
      } else {
        part += lookupWord(lower, cleanWords[i]);
      }
    }
    if (trailingPuncts[i]) part += trailingPuncts[i];
    if (part) outputParts.push(part);
  }

  return outputParts.join(' ');
}

// ── Levenshtein distance for character-level similarity ──

function levenshtein(a: string, b: string): number {
  const m = a.length, n = b.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = Math.min(
        dp[i - 1][j] + 1, dp[i][j - 1] + 1,
        dp[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)
      );
    }
  }
  return dp[m][n];
}

function charSimilarity(a: string, b: string): number {
  const maxLen = Math.max(a.length, b.length);
  if (maxLen === 0) return 1;
  return 1 - levenshtein(a, b) / maxLen;
}

// ── Main test runner ──

console.log(`Dictionary: ${dictMap.size} entries`);
console.log(`Rules: ${isRulesLoaded() ? 'loaded' : 'NOT loaded'}`);
console.log();
console.log('='.repeat(80));
console.log('Phonemizer Correctness & Performance Test');
console.log('Reference: espeak-ng (en-us, preserve_punctuation=True, with_stress=True)');
console.log('='.repeat(80));
console.log();

let exactMatches = 0;
let totalSimilarity = 0;
let totalTime = 0;
const failures: { text: string; expected: string; got: string; similarity: number }[] = [];

for (const ref of REFERENCE) {
  const start = performance.now();
  const result = phonemize(ref.text);
  const elapsed = performance.now() - start;
  totalTime += elapsed;

  const exact = result === ref.phonemes;
  if (exact) exactMatches++;
  const similarity = charSimilarity(result, ref.phonemes);
  totalSimilarity += similarity;

  const status = exact ? '✓ EXACT' : `✗ ${(similarity * 100).toFixed(1)}%`;
  console.log(`${status}  "${ref.text}"`);
  if (!exact) {
    console.log(`  expected: ${ref.phonemes}`);
    console.log(`  got:      ${result}`);
    failures.push({ text: ref.text, expected: ref.phonemes, got: result, similarity });
  }
  console.log(`  time: ${elapsed.toFixed(2)}ms`);
  console.log();
}

// ── Summary ──
console.log('='.repeat(80));
console.log('SUMMARY');
console.log('='.repeat(80));
console.log();
console.log(`Exact matches:     ${exactMatches}/${REFERENCE.length} (${(exactMatches / REFERENCE.length * 100).toFixed(1)}%)`);
console.log(`Char similarity:   ${(totalSimilarity / REFERENCE.length * 100).toFixed(1)}% average`);
console.log(`Total time:        ${totalTime.toFixed(1)}ms`);
console.log(`Avg per sentence:  ${(totalTime / REFERENCE.length).toFixed(2)}ms`);
console.log(`Dictionary size:   ${dictMap.size} entries`);
console.log();

if (failures.length > 0) {
  console.log(`${failures.length} non-exact results (sorted by worst similarity):`);
  failures.sort((a, b) => a.similarity - b.similarity);
  for (const f of failures) {
    console.log(`  ${(f.similarity * 100).toFixed(1)}% — "${f.text}"`);
  }
}

// ── Assertions ──
// These thresholds should be maintained or improved over time.
const MIN_EXACT_RATE = 0.40;    // At least 40% exact matches
const MIN_SIMILARITY = 0.95;    // At least 95% average char similarity
const MAX_MS_PER_SENTENCE = 5;  // Under 5ms per sentence

const avgSimilarity = totalSimilarity / REFERENCE.length;
const exactRate = exactMatches / REFERENCE.length;
const avgMs = totalTime / REFERENCE.length;

console.log();
console.log('Assertions:');

let passed = true;

if (exactRate >= MIN_EXACT_RATE) {
  console.log(`  ✓ Exact match rate ${(exactRate * 100).toFixed(1)}% >= ${(MIN_EXACT_RATE * 100).toFixed(0)}%`);
} else {
  console.log(`  ✗ FAIL: Exact match rate ${(exactRate * 100).toFixed(1)}% < ${(MIN_EXACT_RATE * 100).toFixed(0)}%`);
  passed = false;
}

if (avgSimilarity >= MIN_SIMILARITY) {
  console.log(`  ✓ Avg similarity ${(avgSimilarity * 100).toFixed(1)}% >= ${(MIN_SIMILARITY * 100).toFixed(0)}%`);
} else {
  console.log(`  ✗ FAIL: Avg similarity ${(avgSimilarity * 100).toFixed(1)}% < ${(MIN_SIMILARITY * 100).toFixed(0)}%`);
  passed = false;
}

if (avgMs <= MAX_MS_PER_SENTENCE) {
  console.log(`  ✓ Avg time ${avgMs.toFixed(2)}ms <= ${MAX_MS_PER_SENTENCE}ms`);
} else {
  console.log(`  ✗ FAIL: Avg time ${avgMs.toFixed(2)}ms > ${MAX_MS_PER_SENTENCE}ms`);
  passed = false;
}

console.log();
if (passed) {
  console.log('All assertions passed ✓');
} else {
  console.log('Some assertions FAILED ✗');
  process.exit(1);
}
