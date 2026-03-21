/**
 * WebGPU TTS Pipeline Diagnostic — runs the real engine stage by stage
 * with explicit GPU syncs and timing to identify where iOS Safari hangs.
 */

// Safari ReadableStream polyfill (needed for WASM phonemizer)
if (
  typeof ReadableStream !== 'undefined' &&
  !ReadableStream.prototype[Symbol.asyncIterator]
) {
  (ReadableStream.prototype as any)[Symbol.asyncIterator] = async function* () {
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

const log = document.getElementById('log')!;
const status = document.getElementById('status')!;

function emit(msg: string, cls: string = 'info') {
  const line = document.createElement('div');
  line.className = cls;
  line.textContent = `[${new Date().toISOString().slice(11, 23)}] ${msg}`;
  log.appendChild(line);
  // Auto-scroll
  window.scrollTo(0, document.body.scrollHeight);
}

function setStatus(msg: string, cls: string) {
  status.textContent = msg;
  status.className = cls;
}

async function run() {
  setStatus('Starting diagnostics...', 'running');

  // ── 1. WebGPU check ──
  emit('Checking WebGPU support...');
  if (!navigator.gpu) {
    emit('FAIL: navigator.gpu not available', 'err');
    setStatus('FAILED — no WebGPU', 'failed');
    return;
  }
  emit('navigator.gpu available', 'ok');

  // ── 2. Engine init + model load ──
  emit('');
  emit('=== Loading TTS engine ===');
  const engine = new KittenTTSEngine();

  const t0 = performance.now();
  try {
    emit('Calling engine.init() — creating WebGPU device...');
    await engine.init();
    emit(`WebGPU device ready in ${((performance.now() - t0) / 1000).toFixed(1)}s`, 'ok');
  } catch (e: any) {
    emit(`FAIL engine.init(): ${e.message}`, 'err');
    setStatus('FAILED — engine init', 'failed');
    return;
  }

  // ── 2b. Load model weights + voices ──
  emit('');
  emit('=== Loading model weights + voices ===');
  const HF_BASE = 'https://huggingface.co/KittenML/kitten-tts-mini-0.8/resolve/main';
  const modelUrl = `${HF_BASE}/kitten_tts_mini_v0_8.onnx`;
  const voicesUrl = `${HF_BASE}/voices.npz`;

  const t1model = performance.now();
  try {
    emit('Downloading model (~75MB) + voices...');
    await engine.loadModel(modelUrl, voicesUrl);
    emit(`Model loaded in ${((performance.now() - t1model) / 1000).toFixed(1)}s`, 'ok');
  } catch (e: any) {
    emit(`FAIL loadModel(): ${e.message}`, 'err');
    setStatus('FAILED — model load', 'failed');
    return;
  }

  // ── 3. Phonemizer ──
  emit('');
  emit('=== Phonemizer ===');
  const testText = 'Hello world.';
  try {
    const t1 = performance.now();
    const { ids, method } = await textToInputIds(testText);
    emit(`Phonemized "${testText}" → ${ids.length} tokens (${method}) in ${(performance.now() - t1).toFixed(0)}ms`, 'ok');
  } catch (e: any) {
    emit(`FAIL phonemizer: ${e.message}`, 'err');
    setStatus('FAILED — phonemizer', 'failed');
    return;
  }

  // ── 4. Full generate with stage callbacks ──
  emit('');
  emit('=== Running TTS generate (stage by stage) ===');
  emit('Each stage will flush GPU and report timing.');
  emit(`Text: "${testText}"`);

  // Enable profiling so we get per-stage GPU syncs
  engine.profile = true;

  const tGen = performance.now();
  let lastStage = '';
  let stageStart = performance.now();

  try {
    const { ids: inputIds } = await textToInputIds(testText);

    const { waveform } = await engine.generate(inputIds, 'Bella', 1.0, testText.length, (stage) => {
      const now = performance.now();
      if (lastStage) {
        emit(`  ✓ ${lastStage} — ${(now - stageStart).toFixed(0)}ms`, 'ok');
      }
      emit(`  → ${stage}...`);
      lastStage = stage;
      stageStart = now;
    });

    // Final stage timing
    const now = performance.now();
    if (lastStage) {
      emit(`  ✓ ${lastStage} — ${(now - stageStart).toFixed(0)}ms`, 'ok');
    }

    emit('');
    emit(`=== GENERATE COMPLETE ===`, 'ok');
    emit(`Total: ${((now - tGen) / 1000).toFixed(2)}s`, 'ok');
    emit(`Waveform: ${waveform.length} samples (${(waveform.length / 24000).toFixed(2)}s audio)`, 'ok');
    setStatus('ALL PASSED — generation complete!', 'done');

    // Play audio
    const wav = encodeWav(waveform, 24000);
    const blob = new Blob([wav], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    const audio = document.getElementById('audio') as HTMLAudioElement;
    audio.src = url;
    audio.style.display = 'block';
    emit('Audio ready — tap play above ↑', 'ok');

  } catch (e: any) {
    const now = performance.now();
    if (lastStage) {
      emit(`  ✗ ${lastStage} — FAILED after ${(now - stageStart).toFixed(0)}ms`, 'err');
    }
    emit(`FAIL generate: ${e.message}`, 'err');
    emit(`Stack: ${e.stack?.split('\n').slice(0, 3).join(' | ')}`, 'err');
    setStatus(`FAILED at ${lastStage || 'unknown stage'}`, 'failed');
  }
}

/** Encode Float32Array as 16-bit WAV */
function encodeWav(samples: Float32Array, sampleRate: number): ArrayBuffer {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  return buffer;
}

// Auto-run
run().catch((e) => {
  emit(`UNCAUGHT: ${e.message}`, 'err');
  setStatus('CRASHED', 'failed');
});
