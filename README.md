# Kitten TTS WebGPU

[![npm](https://img.shields.io/npm/v/kitten-tts-webgpu)](https://www.npmjs.com/package/kitten-tts-webgpu)
[![license](https://img.shields.io/npm/l/kitten-tts-webgpu)](./LICENSE)

**Pure WebGPU text-to-speech for the browser. 80M params, sub-second on desktop, ~1.2s on iPhone. No ONNX Runtime, no WASM inference — just 29 compute shaders.**

[**Live Demo**](https://svenflow.github.io/kitten-tts-webgpu/) | [npm](https://www.npmjs.com/package/kitten-tts-webgpu) | [Model Card](https://huggingface.co/KittenML/kitten-tts-mini-0.8)

---

## Quick Start

```bash
npm install kitten-tts-webgpu
```

```typescript
import { textToSpeech } from 'kitten-tts-webgpu';

const blob = await textToSpeech("The quick brown fox jumps over the lazy dog.");
const audio = new Audio(URL.createObjectURL(blob));
audio.play();
```

One function. Text in, WAV blob out (16-bit PCM, 24 kHz mono). The model downloads on first call and is cached for subsequent calls. Full TypeScript types included.

> **Note:** This library requires WebGPU. For server-side rendering frameworks (Next.js, Nuxt), dynamically import on the client side only.

## Models

Three [Kitten TTS v0.8](https://huggingface.co/KittenML) sizes, same API:

| Model | Params | Download | M4 Pro (Chrome) | iPhone 17 Pro Max |
|-------|--------|----------|------------------|-------------------|
| **Mini** | 80M | 78 MB | 1.80s (3.3x RT) | ~1.2s |
| **Micro** | 40M | 41 MB | 1.05s (6.2x RT) | untested |
| **Nano** | 15M | 24 MB | 0.93s (7.3x RT) | untested |

*RT = real-time multiplier (audio duration / generation time). Higher is better. Times are warm (model cached in GPU). First call includes ~2-4s model download.*

```typescript
await textToSpeech("Hello world");                        // Default: mini
await textToSpeech("Hello world", { model: 'micro' });    // Balanced
await textToSpeech("Hello world", { model: 'nano' });     // Fastest
```

## Options

```typescript
const blob = await textToSpeech("Welcome to the future.", {
  voice: "Leo",        // 8 voices: Bella, Luna, Rosie, Kiki, Jasper, Bruno, Hugo, Leo
  speed: 1.2,          // 0.5x – 2.0x
  model: "micro",      // mini | micro | nano
  onProgress: (stage) => console.log(stage), // string: "Initializing WebGPU…", "Downloading…", "Generating speech…", etc.
});
```

### Voices

| Female | Male |
|--------|------|
| Bella  | Jasper |
| Luna   | Bruno |
| Rosie  | Hugo |
| Kiki   | Leo |

## Error Handling

```typescript
// Check for WebGPU support
if (!navigator.gpu) {
  console.log("WebGPU not available — use Chrome 113+, Edge 113+, or Safari 26+");
}

// textToSpeech throws on:
// - No WebGPU support
// - Network error (model download fails)
// - Empty text input
try {
  const blob = await textToSpeech("Hello");
} catch (err) {
  console.error("TTS failed:", err.message);
}
```

## Advanced: Direct Engine Access

For repeated generations or fine-grained control:

```typescript
import { KittenTTSEngine, textToInputIds, float32ToWav } from 'kitten-tts-webgpu';

const engine = new KittenTTSEngine();
await engine.init();
await engine.loadModel(onnxUrl, voicesUrl);

const { ids } = await textToInputIds("Hello world");
const { waveform } = await engine.generate(ids, "Bella", 1.0);
// waveform: Float32Array of 24kHz PCM samples

const wavBlob = float32ToWav(waveform, 24000);
```

## How It Works

29 hand-written [WGSL compute shaders](./src/shaders.ts) execute the full TTS pipeline on GPU:

```
Text → Phonemes (234K-word dictionary + espeak rules in pure JS)
  → ALBERT encoder (embedding, multi-head attention, FFN)
  → Duration predictor (LSTM + CNN)
  → Acoustic decoder (LSTM + AdaIN + CNN, style-conditioned)
  → HiFi-GAN vocoder (ConvTranspose1d, Snake activations, iSTFT)
  → 24kHz WAV
```

**Why not ONNX Runtime Web?**

Most browser TTS uses ONNX Runtime Web (~2MB WASM binary + C++ runtime). This project takes a different approach:

- **Custom ONNX parser** — dequantizes int8/uint8/float16 weights in pure TypeScript, no C++ runtime
- **234K-word phonemizer** — espeak-ng rules ported to pure JS (WASM espeak hangs on iOS Safari)
- **GPU buffer pooling** — reuses buffers across HiFi-GAN iterations, ~130MB peak on mobile
- **Dynamic architecture** — detects model dimensions from weight shapes, one engine for all 3 sizes

## Browser Support

| Browser | Status |
|---------|--------|
| Chrome 113+ | ✅ |
| Edge 113+ | ✅ |
| Safari 26+ (macOS/iOS) | ✅ |
| Firefox Nightly | Experimental |

## FAQ

**Max input length?** Recommended under ~500 characters per call. For longer text, split into sentences.

**Languages?** English only (matches the upstream Kitten TTS model).

**Offline?** Yes, after the model is cached in the browser. No server needed for inference.

**Self-hosting models?** Pass custom URLs to `KittenTTSEngine.loadModel(onnxUrl, voicesUrl)`.

**Bundle size?** ~750KB gzipped (includes engine, shaders, and 234K-word phonemizer dictionary). Model weights (24-78MB) download separately at runtime.

**Model license?** Kitten TTS models are released under [Apache 2.0](https://huggingface.co/KittenML/kitten-tts-mini-0.8). Code in this repo is MIT.

## Development

```bash
git clone https://github.com/svenflow/kitten-tts-webgpu.git
cd kitten-tts-webgpu
npm install
npm run dev       # Dev server
npm run build     # Production build
npm test          # Phonemizer tests
```

## Credits

- [Kitten TTS](https://huggingface.co/KittenML/kitten-tts-mini-0.8) models by KittenML (Apache 2.0)
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) pronunciation dictionary and letter-to-sound rules (GPL-3.0, bundled as data files)
- [phonemizer](https://www.npmjs.com/package/phonemizer) by Xenova (espeak-ng WASM, used as primary backend on Chrome/Firefox; pure JS fallback on Safari)

## License

MIT
