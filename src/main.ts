/**
 * Main entry point for Kitten TTS WebGPU demo.
 */

// Polyfill: Safari lacks ReadableStream async iterator support,
// which phonemizer.js needs for WASM decompression (see phonemizer.js#2)
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

// ── DOM Elements ──
const loadingScreen = document.getElementById('loading-screen')!;
const startSection = document.getElementById('start-section')!;
const progressSection = document.getElementById('progress-section')!;
const startBtn = document.getElementById('start-btn')!;
const progressFill = document.getElementById('progress-fill')!;
const progressText = document.getElementById('progress-text')!;
const appEl = document.getElementById('app')!;
const textInput = document.getElementById('text-input') as HTMLTextAreaElement;
const charCount = document.getElementById('char-count')!;
const voiceSelect = document.getElementById('voice-select') as HTMLSelectElement;
const speedSlider = document.getElementById('speed-slider') as HTMLInputElement;
const speedVal = document.getElementById('speed-val')!;
const generateBtn = document.getElementById('generate-btn') as HTMLButtonElement;
const outputSection = document.getElementById('output-section')!;
const waveformCanvas = document.getElementById('waveform-canvas') as HTMLCanvasElement;
const waveformWrap = document.getElementById('waveform-wrap')!;
const playhead = document.getElementById('playhead')!;
const playBtn = document.getElementById('play-btn')!;
const downloadBtn = document.getElementById('download-btn')!;
const playIcon = document.getElementById('play-icon')!;
const pauseIcon = document.getElementById('pause-icon')!;
const metaDuration = document.getElementById('meta-duration')!;
const metaSpeed = document.getElementById('meta-speed')!;
const metaSamples = document.getElementById('meta-samples')!;
const timingIndicator = document.getElementById('timing-indicator')!;
const logToggle = document.getElementById('log-toggle')!;
const logContent = document.getElementById('log-content')!;
const audioEl = document.getElementById('audio-el') as HTMLAudioElement;

let engine: KittenTTSEngine | null = null;
let lastSamples: Float32Array | null = null;
let lastBlobUrl: string | null = null;

// Handle GPU device loss
window.addEventListener('webgpu-device-lost', ((e: CustomEvent) => {
  const info = e.detail;
  log(`GPU device lost: ${info.reason} — ${info.message}`, 'error');
  generateBtn.disabled = true;
  generateBtn.textContent = 'GPU Lost — Reload Page';
  // Auto-open log so user can see the error
  if (!logContent.classList.contains('open')) logToggle.click();
}) as EventListener);

// Handle uncaptured GPU errors
window.addEventListener('webgpu-error', ((e: CustomEvent) => {
  log(`GPU error: ${e.detail}`, 'error');
  if (!logContent.classList.contains('open')) logToggle.click();
}) as EventListener);

// ── Logging ──
// Show crash log from previous session (survives page refresh)
const prevCrashLog = localStorage.getItem('kitten-tts-crash-log');
if (prevCrashLog) {
  localStorage.removeItem('kitten-tts-crash-log');
  const banner = document.createElement('div');
  banner.style.cssText = 'background:#ff000033;color:#ff6b6b;padding:12px;margin:8px 0;border-radius:8px;font-family:monospace;font-size:11px;white-space:pre-wrap;';
  banner.textContent = '⚠️ Previous session crashed. Last log:\n' + prevCrashLog;
  document.querySelector('.container')?.prepend(banner);
}

function log(msg: string, type: 'info' | 'error' | 'success' = 'info') {
  const entry = document.createElement('div');
  entry.className = `log-entry log-${type}`;
  const time = new Date().toLocaleTimeString();
  entry.textContent = `[${time}] ${msg}`;
  logContent.appendChild(entry);
  logContent.scrollTop = logContent.scrollHeight;
  console.log(`[${type}] ${msg}`);
  // Persist to localStorage so we survive crash-refreshes
  const logs = localStorage.getItem('kitten-tts-crash-log') || '';
  localStorage.setItem('kitten-tts-crash-log', logs + `[${time}] ${msg}\n`);
}

// Clear crash log when generation completes successfully
function clearCrashLog() {
  localStorage.removeItem('kitten-tts-crash-log');
}

// ── Character count ──
function updateCharCount() {
  charCount.textContent = `${textInput.value.length} chars`;
}
textInput.addEventListener('input', updateCharCount);
updateCharCount();

// ── Speed slider with filled track ──
function updateSpeedSlider() {
  const val = parseFloat(speedSlider.value);
  speedVal.textContent = `${val.toFixed(1)}×`;
  speedSlider.setAttribute('aria-valuetext', `${val.toFixed(1)}x`);
  const pct = ((val - 0.5) / 1.5) * 100;
  speedSlider.style.setProperty('--fill-pct', `${pct}%`);
}
speedSlider.addEventListener('input', updateSpeedSlider);
updateSpeedSlider();

// ── Log toggle ──
logToggle.addEventListener('click', () => {
  const isOpen = logContent.classList.toggle('open');
  logToggle.textContent = isOpen ? '▾ Activity Log' : '▸ Activity Log';
  logToggle.setAttribute('aria-expanded', String(isOpen));
});

// ── Platform-aware shortcut hint ──
const isMac = navigator.platform?.includes('Mac') || navigator.userAgent.includes('Mac');
const shortcutHint = document.getElementById('shortcut-hint')!;
shortcutHint.textContent = `${isMac ? '⌘' : 'Ctrl'} Enter to generate`;

// ── Preset chips ──
const presetChips = document.querySelectorAll('.preset-chip');
presetChips.forEach(chip => {
  chip.addEventListener('click', () => {
    // Clear active state from all chips
    presetChips.forEach(c => c.classList.remove('active'));
    chip.classList.add('active');
    textInput.value = (chip as HTMLElement).dataset.text || '';
    updateCharCount();
    textInput.focus();
  });
});

// Clear active chip when user types
textInput.addEventListener('input', () => {
  presetChips.forEach(c => c.classList.remove('active'));
});

// ── Waveform drawing ──
function drawWaveform(samples: Float32Array) {
  const canvas = waveformCanvas;
  // Cap DPR at 2 to avoid massive canvas allocations on 3x retina iPhones
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return; // Not visible yet
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;

  const ctx = canvas.getContext('2d')!;
  ctx.scale(dpr, dpr);

  const w = rect.width;
  const h = rect.height;
  const mid = h / 2;

  ctx.clearRect(0, 0, w, h);

  // Center line
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(229, 165, 75, 0.12)';
  ctx.lineWidth = 1;
  ctx.moveTo(0, mid);
  ctx.lineTo(w, mid);
  ctx.stroke();

  // Draw waveform
  const step = Math.max(1, Math.floor(samples.length / w));
  ctx.beginPath();
  ctx.strokeStyle = '#e5a54b';
  ctx.lineWidth = 1.2;

  for (let x = 0; x < w; x++) {
    const idx = Math.floor((x / w) * samples.length);
    let min = Infinity, max = -Infinity;
    for (let j = 0; j < step && idx + j < samples.length; j++) {
      const s = samples[idx + j];
      if (s < min) min = s;
      if (s > max) max = s;
    }
    const yMin = mid - min * mid * 0.85;
    const yMax = mid - max * mid * 0.85;
    ctx.moveTo(x, yMin);
    ctx.lineTo(x, yMax);
  }
  ctx.stroke();
}

// ── Playback + playhead ──
let isPlaying = false;

audioEl.addEventListener('play', () => {
  isPlaying = true;
  playIcon.style.display = 'none';
  pauseIcon.style.display = 'block';
  playBtn.setAttribute('aria-label', 'Pause audio');
  playhead.classList.add('active');
});
audioEl.addEventListener('pause', () => {
  isPlaying = false;
  playIcon.style.display = 'block';
  pauseIcon.style.display = 'none';
  playBtn.setAttribute('aria-label', 'Play audio');
});
audioEl.addEventListener('ended', () => {
  isPlaying = false;
  playIcon.style.display = 'block';
  pauseIcon.style.display = 'none';
  playBtn.setAttribute('aria-label', 'Play audio');
  playhead.classList.remove('active');
  playhead.style.left = '0';
});

// Update playhead position during playback
audioEl.addEventListener('timeupdate', () => {
  if (audioEl.duration && isFinite(audioEl.duration)) {
    const pct = audioEl.currentTime / audioEl.duration;
    const rect = waveformWrap.getBoundingClientRect();
    playhead.style.left = `${pct * rect.width}px`;
  }
});

// Click on waveform to seek
waveformWrap.addEventListener('click', (e) => {
  if (!audioEl.src || !audioEl.duration) return;
  const rect = waveformWrap.getBoundingClientRect();
  const pct = (e.clientX - rect.left) / rect.width;
  audioEl.currentTime = pct * audioEl.duration;
  if (!isPlaying) audioEl.play().catch(() => {});
});

playBtn.addEventListener('click', () => {
  if (isPlaying) {
    audioEl.pause();
  } else {
    audioEl.play().catch(() => {});
  }
});

// ── Download ──
downloadBtn.addEventListener('click', () => {
  if (!lastBlobUrl) return;
  const a = document.createElement('a');
  a.href = lastBlobUrl;
  a.download = 'kitten-tts-output.wav';
  a.click();
});

// ── Model loading ──
startBtn.addEventListener('click', () => {
  startSection.style.display = 'none';
  progressSection.style.display = 'block';
  loadModel();
});

// Auto-start model loading if ?autostart=1
if (new URLSearchParams(location.search).get('autostart') === '1') {
  startBtn.click();
}

async function loadModel() {
  log('Initializing WebGPU...');
  progressText.textContent = 'Initializing WebGPU…';

  if (!navigator.gpu) {
    log('WebGPU not available in this browser!', 'error');
    progressText.textContent = 'WebGPU not available — try Chrome or Edge.';
    return;
  }

  engine = new KittenTTSEngine();
  engine.profile = true;

  try {
    await engine.init();
    log('WebGPU device ready', 'success');
  } catch (e) {
    log(`WebGPU init failed: ${e}`, 'error');
    progressText.textContent = `WebGPU init failed: ${e}`;
    return;
  }

  // Model URLs
  const HF_BASE = 'https://huggingface.co/KittenML/kitten-tts-mini-0.8/resolve/main';
  const isLocal = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
  const modelUrl = isLocal
    ? '/models/kitten-tts-mini-0.8/kitten_tts_mini_v0_8.onnx'
    : `${HF_BASE}/kitten_tts_mini_v0_8.onnx`;
  const voicesUrl = isLocal
    ? '/models/kitten-tts-mini-0.8/voices.npz'
    : `${HF_BASE}/voices.npz`;

  log('Loading model weights (74.6 MB)...');
  progressText.textContent = 'Downloading model (74.6 MB)…';

  try {
    const start = performance.now();
    await engine.loadModel(modelUrl, voicesUrl);

    progressFill.style.width = '100%';
    const elapsed = ((performance.now() - start) / 1000).toFixed(1);
    log(`Model loaded in ${elapsed}s`, 'success');
    progressText.textContent = `Ready — loaded in ${elapsed}s`;

    // Transition to app
    setTimeout(() => {
      loadingScreen.classList.add('hidden');
      appEl.classList.add('visible');
      generateBtn.disabled = false;
      generateBtn.textContent = 'Generate Speech';
      textInput.focus();

      // Auto-generate if ?autostart=1 is in URL
      if (new URLSearchParams(location.search).get('autostart') === '1') {
        log('Auto-start: triggering generation...');
        setTimeout(() => generateBtn.click(), 500);
      }
    }, 400);

  } catch (e) {
    log(`Model load failed: ${e}`, 'error');
    progressText.textContent = `Load failed: ${e}`;
    return;
  }
}

// ── Generation ──
generateBtn.addEventListener('click', async () => {
  if (!engine) return;

  const text = textInput.value.trim();
  if (!text) {
    log('Please enter some text!', 'error');
    return;
  }

  generateBtn.disabled = true;
  generateBtn.textContent = 'Generating…';
  outputSection.classList.remove('visible');

  const voice = voiceSelect.value;
  const speed = parseFloat(speedSlider.value);

  log(`Generating: "${text.slice(0, 80)}${text.length > 80 ? '…' : ''}" (voice=${voice}, speed=${speed}×)`);

  try {
    const { ids: inputIds, method } = await textToInputIds(text);
    log(`Phonemized: ${inputIds.length} tokens (${method})`);

    const start = performance.now();
    const { waveform } = await engine.generate(inputIds, voice, speed, text.length, (stage) => {
      generateBtn.textContent = stage;
      log(`Stage: ${stage}`);
    });
    const elapsed = ((performance.now() - start) / 1000).toFixed(2);
    const duration = (waveform.length / 24000).toFixed(2);

    log(`Generated ${waveform.length} samples (${duration}s) in ${elapsed}s`, 'success');

    // Log per-stage timing breakdown
    if (engine.lastTimings.length > 0) {
      log('── Stage Timings ──');
      const maxMs = Math.max(...engine.lastTimings.map(t => t.ms));
      for (const { name, ms } of engine.lastTimings) {
        const barLen = Math.max(1, Math.round((ms / Math.max(maxMs, 1)) * 20));
        const bar = '█'.repeat(barLen);
        log(`  ${name}: ${ms.toFixed(0)}ms ${bar}`);
      }
      // Auto-open log on desktop only (the CSS transition from closed→open
      // triggers compositor layer allocation that crashes iOS Safari WebContent)
      const isMobileForLog = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
      if (!isMobileForLog && !logContent.classList.contains('open')) logToggle.click();
    }

    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);

    // Store for resize redraw (skip on mobile to save memory)
    if (!isMobile) {
      lastSamples = waveform;
    }

    // Update metadata
    metaDuration.textContent = `${duration}s`;
    metaSpeed.textContent = `${elapsed}s`;
    metaSamples.textContent = waveform.length.toLocaleString();

    // Update timing indicator in header
    timingIndicator.textContent = `${elapsed}s`;
    timingIndicator.classList.add('visible');

    // Skip waveform canvas on mobile to avoid OOM crashes
    if (!isMobile) {
      log('Drawing waveform...');
      drawWaveform(waveform);
      log('Waveform drawn');
    } else {
      log('Skipping waveform canvas (mobile)');
      // Hide the waveform container entirely on mobile
      waveformCanvas.style.display = 'none';
    }

    // Create audio blob
    log('Encoding WAV...');
    const wavBlob = float32ToWav(waveform, 24000);
    log(`WAV encoded: ${(wavBlob.size / 1024).toFixed(0)}KB`);
    if (lastBlobUrl) URL.revokeObjectURL(lastBlobUrl);
    lastBlobUrl = URL.createObjectURL(wavBlob);

    // Show output section BEFORE setting audio src (avoid simultaneous allocs)
    outputSection.classList.add('visible');

    // On mobile: don't auto-play (avoids audio buffer allocation on top of
    // GPU memory), and don't auto-open log (the CSS transition from closed→open
    // triggers compositor layer allocation that crashes WebContent)
    if (isMobile) {
      log('Done! Tap play to listen.');
      // Defer audio src assignment to let layout settle
      await new Promise(r => setTimeout(r, 100));
      audioEl.src = lastBlobUrl;
    } else {
      audioEl.src = lastBlobUrl;
      log('Playing audio...');
      audioEl.play().catch(() => {});
    }

    log('Done!');
    clearCrashLog();

  } catch (e) {
    log(`Generation failed: ${e}`, 'error');
  }

  generateBtn.disabled = false;
  generateBtn.textContent = 'Generate Speech';
});

// ── Keyboard shortcut ──
textInput.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    if (!generateBtn.disabled) generateBtn.click();
  }
});

// ── Resize handler ──
window.addEventListener('resize', () => {
  if (lastSamples && outputSection.classList.contains('visible')) {
    drawWaveform(lastSamples);
  }
});

// ── WAV encoding ──
function float32ToWav(samples: Float32Array, sampleRate: number): Blob {
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
  const dataSize = samples.length * (bitsPerSample / 8);
  const headerSize = 44;
  const buffer = new ArrayBuffer(headerSize + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, numChannels * (bitsPerSample / 8), true);
  view.setUint16(34, bitsPerSample, true);
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = headerSize;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}
