/**
 * kitten-tts-webgpu — Run Kitten TTS (80M) in the browser via WebGPU.
 *
 * One function is all you need:
 *
 * ```typescript
 * import { textToSpeech } from 'kitten-tts-webgpu';
 *
 * const wav = await textToSpeech('Hello, world!');
 * const audio = new Audio(URL.createObjectURL(wav));
 * audio.play();
 * ```
 */

import { installStreamPolyfill } from './polyfills.js';
import { KittenTTSEngine } from './engine.js';
import { textToInputIds } from './phonemizer.js';
import { float32ToWav } from './wav.js';

/** Model size variant. Default: `'nano'`. */
export type ModelSize = 'mini' | 'micro' | 'nano';

/** Options for the `textToSpeech` function. */
export interface TextToSpeechOptions {
  /** Voice name. Default: `'Bella'`. */
  voice?: string;
  /** Speaking speed multiplier. Default: `1.0`. Range: 0.5 – 2.0. */
  speed?: number;
  /** Model size variant. Default: `'nano'` (15M, fastest, 24 MB). `'micro'` (40M) and `'mini'` (80M, best quality) are larger. */
  model?: ModelSize;
  /** Progress callback, called with stage descriptions like `'Loading model…'`. */
  onProgress?: (stage: string) => void;
}

const MODEL_URLS: Record<ModelSize, { url: string; voicesUrl: string; size: string; params: string }> = {
  mini:  { url: 'https://huggingface.co/KittenML/kitten-tts-mini-0.8/resolve/main/kitten_tts_mini_v0_8.onnx',      voicesUrl: 'https://huggingface.co/KittenML/kitten-tts-mini-0.8/resolve/main/voices.npz',      size: '78 MB',  params: '80M' },
  micro: { url: 'https://huggingface.co/KittenML/kitten-tts-micro-0.8/resolve/main/kitten_tts_micro_v0_8.onnx',    voicesUrl: 'https://huggingface.co/KittenML/kitten-tts-micro-0.8/resolve/main/voices.npz',    size: '41 MB',  params: '40M' },
  nano:  { url: 'https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8/resolve/main/kitten_tts_nano_v0_8.onnx', voicesUrl: 'https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8/resolve/main/voices.npz', size: '24 MB',  params: '15M' },
};

/** Per-model singleton engines, lazily initialized. */
const engines = new Map<ModelSize, KittenTTSEngine>();
const initPromises = new Map<ModelSize, Promise<KittenTTSEngine>>();

/**
 * Get or initialize an engine for the given model size.
 * Safe to call concurrently — only one init runs per model.
 */
async function getEngine(model: ModelSize = 'mini', onProgress?: (stage: string) => void): Promise<KittenTTSEngine> {
  const existing = engines.get(model);
  if (existing) return existing;

  const pending = initPromises.get(model);
  if (pending) return pending;

  const promise = (async () => {
    // Lazy polyfill — only runs once, avoids top-level side effects for tree-shaking
    installStreamPolyfill();

    if (!navigator.gpu) {
      throw new Error(
        'WebGPU is not available. Use a browser that supports WebGPU (Chrome 113+, Edge 113+, or Safari 26+).'
      );
    }

    const cfg = MODEL_URLS[model];
    onProgress?.('Initializing WebGPU…');
    const e = new KittenTTSEngine();
    await e.init();

    onProgress?.(`Downloading ${model} model (${cfg.size})…`);
    await e.loadModel(cfg.url, cfg.voicesUrl);

    engines.set(model, e);
    return e;
  })();

  initPromises.set(model, promise);

  try {
    return await promise;
  } catch (err) {
    // Reset so a retry can succeed
    initPromises.delete(model);
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
 * import { textToSpeech } from 'kitten-tts-webgpu';
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

  const { voice = 'Bella', speed = 1.0, model = 'nano', onProgress } = options ?? {};

  const e = await getEngine(model, onProgress);

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
