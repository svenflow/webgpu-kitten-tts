/**
 * webgpu-kitten-tts — Run Kitten TTS (80M) in the browser via WebGPU.
 *
 * One function is all you need:
 *
 * ```typescript
 * import { textToSpeech } from 'webgpu-kitten-tts';
 *
 * const wav = await textToSpeech('Hello, world!');
 * const audio = new Audio(URL.createObjectURL(wav));
 * audio.play();
 * ```
 */

// Polyfill: Safari lacks ReadableStream async iterator support,
// which phonemizer.js needs for WASM decompression
if (
  typeof ReadableStream !== 'undefined' &&
  !(Symbol.asyncIterator in ReadableStream.prototype)
) {
  (ReadableStream.prototype as any)[Symbol.asyncIterator] = async function* (this: ReadableStream) {
    const reader = this.getReader();
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        yield value;
      }
    } finally {
      reader.releaseLock();
    }
  };
}

import { KittenTTSEngine } from './engine.js';
import { textToInputIds } from './phonemizer.js';
import { float32ToWav } from './wav.js';

/** Options for the `textToSpeech` function. */
export interface TextToSpeechOptions {
  /** Voice name. Default: `'Bella'`. */
  voice?: string;
  /** Speaking speed multiplier. Default: `1.0`. Range: 0.5 – 2.0. */
  speed?: number;
  /** Progress callback, called with stage descriptions like `'Loading model…'`. */
  onProgress?: (stage: string) => void;
}

const HF_BASE = 'https://huggingface.co/KittenML/kitten-tts-mini-0.8/resolve/main';
const MODEL_URL = `${HF_BASE}/kitten_tts_mini_v0_8.onnx`;
const VOICES_URL = `${HF_BASE}/voices.npz`;

/** Singleton engine, lazily initialized on first call. */
let engine: KittenTTSEngine | null = null;
let initPromise: Promise<KittenTTSEngine> | null = null;

/**
 * Get or initialize the singleton engine.
 * Safe to call concurrently — only one init runs.
 */
async function getEngine(onProgress?: (stage: string) => void): Promise<KittenTTSEngine> {
  if (engine) return engine;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    if (!navigator.gpu) {
      throw new Error(
        'WebGPU is not available. Use a browser that supports WebGPU (Chrome 113+, Edge 113+, or Safari 18+).'
      );
    }

    onProgress?.('Initializing WebGPU…');
    const e = new KittenTTSEngine();
    await e.init();

    onProgress?.('Downloading model (74.6 MB)…');
    await e.loadModel(MODEL_URL, VOICES_URL);

    engine = e;
    return e;
  })();

  try {
    return await initPromise;
  } catch (err) {
    // Reset so a retry can succeed
    initPromise = null;
    throw err;
  }
}

/**
 * Convert text to a WAV audio blob using Kitten TTS.
 *
 * On the first call, this downloads the model (~75 MB) and initializes WebGPU.
 * Subsequent calls reuse the loaded model and are fast (~1 second).
 *
 * @param text - English text to synthesize (up to ~500 characters recommended)
 * @param options - Optional voice, speed, and progress callback
 * @returns A WAV audio Blob (16-bit PCM, 24 kHz, mono)
 *
 * @example
 * ```typescript
 * import { textToSpeech } from 'webgpu-kitten-tts';
 *
 * // Simple usage
 * const wav = await textToSpeech('Hello, world!');
 *
 * // Play it
 * const audio = new Audio(URL.createObjectURL(wav));
 * audio.play();
 *
 * // With options
 * const wav2 = await textToSpeech('Slow and steady wins the race.', {
 *   voice: 'Bella',
 *   speed: 0.8,
 *   onProgress: (stage) => console.log(stage),
 * });
 * ```
 */
export async function textToSpeech(
  text: string,
  options?: TextToSpeechOptions,
): Promise<Blob> {
  if (!text?.trim()) {
    throw new Error('Text must be a non-empty string.');
  }

  const { voice = 'Bella', speed = 1.0, onProgress } = options ?? {};

  const e = await getEngine(onProgress);

  onProgress?.('Phonemizing…');
  const { ids: inputIds } = await textToInputIds(text);

  onProgress?.('Generating speech…');
  const { waveform } = await e.generate(inputIds, voice, speed, text.length);

  onProgress?.('Encoding WAV…');
  return float32ToWav(waveform, 24000);
}

// Re-export building blocks for advanced users
export { KittenTTSEngine } from './engine.js';
export { textToInputIds } from './phonemizer.js';
export { float32ToWav } from './wav.js';
