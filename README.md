# WebGPU Kitten TTS

**80M-parameter text-to-speech running entirely in the browser via WebGPU compute shaders. No ONNX Runtime. No C++ inference engines. Just TypeScript and WGSL.**

[**Live Demo**](https://svenflow.github.io/kitten-tts-webgpu/) | [Model Card](https://huggingface.co/KittenML/kitten-tts-mini-0.8)

---

## What is this?

A from-scratch neural TTS inference engine where 100% of the model execution happens on the GPU via [WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API) compute shaders. The entire 80M-parameter [Kitten TTS](https://huggingface.co/KittenML/kitten-tts-mini-0.8) model -- BERT encoder, duration predictor, acoustic decoder, and HiFi-GAN vocoder -- runs as GPU dispatches through 20+ hand-written WGSL shaders. No server, no ONNX Runtime, no heavyweight tensor libraries.

- **~1.2s** generation on iPhone 17 Pro Max (Safari, iOS 26)
- **Sub-second** on desktop GPUs
- **74.6 MB** model, loaded once from HuggingFace CDN
- **8 voices** (4 female, 4 male) with adjustable speed
- **No ML framework overhead** -- only `phonemizer` (espeak-ng WASM, for text-to-phoneme conversion)

## Quick Start

```bash
npm install kitten-tts-webgpu
```

```typescript
import { textToSpeech } from 'kitten-tts-webgpu';

const blob = await textToSpeech("The quick brown fox jumps over the lazy dog.");

// Play it
const audio = new Audio(URL.createObjectURL(blob));
audio.play();
```

One function, one line, WAV audio out. The model is initialized on the first call and cached for subsequent calls.

## Usage

### Basic

```typescript
const blob = await textToSpeech("Hello world");
```

### Custom voice and speed

```typescript
const blob = await textToSpeech("Welcome to the future of browser TTS.", {
  voice: "Leo",
  speed: 1.2,
});
```

### Progress callback

```typescript
const blob = await textToSpeech("Generating speech with progress updates.", {
  onProgress: (stage) => console.log(stage),
  // Logs: "Initializing WebGPU…", "Downloading model…", "Phonemizing…",
  //       "Generating speech…", "Encoding WAV…"
});
```

### Direct engine access

For repeated generations or fine-grained control, use the engine directly:

```typescript
import { KittenTTSEngine, textToInputIds } from 'kitten-tts-webgpu';

const engine = new KittenTTSEngine();
await engine.init();
await engine.loadModel(onnxUrl, voicesUrl);

const { ids } = await textToInputIds("Hello world");
const { waveform } = await engine.generate(ids, "Bella", 1.0);
// waveform is a Float32Array of 24kHz PCM samples
```

## Voices

| Name   | Gender | ID               |
|--------|--------|------------------|
| Bella  | F      | expr-voice-2-f   |
| Luna   | F      | expr-voice-3-f   |
| Rosie  | F      | expr-voice-4-f   |
| Kiki   | F      | expr-voice-5-f   |
| Jasper | M      | expr-voice-2-m   |
| Bruno  | M      | expr-voice-3-m   |
| Hugo   | M      | expr-voice-4-m   |
| Leo    | M      | expr-voice-5-m   |

Pass the name (e.g. `"Bella"`) or the raw ID (e.g. `"expr-voice-2-f"`) to the `voice` option.

## Architecture

The inference pipeline has four stages. All neural network computation runs as WebGPU compute shader dispatches -- no CPU-side tensor math, no ONNX Runtime, no WASM inference.

```
Text
 |  Phonemizer (espeak-ng WASM + 234K-word dictionary fallback)
 v
Phoneme IDs
 |  ALBERT-based BERT encoder
 |  (embedding, multi-head attention, feed-forward)
 v
Hidden States
 |  Duration predictor (LSTM + CNN) --> per-phoneme durations
 |  Acoustic decoder (LSTM + AdaIN + CNN, style-conditioned)
 v
Mel Spectrogram
 |  HiFi-GAN vocoder
 |  (transposed convolutions, Snake activations, residual blocks, iSTFT)
 v
24kHz Waveform (WAV)
```

### What makes this different

Most browser-based ML projects use ONNX Runtime Web (a large C++/WASM runtime) or similar frameworks to execute models. This project takes a fundamentally different approach:

- **Custom ONNX parser** written in TypeScript extracts and dequantizes weights (int8/uint8/float16) directly -- no C++ runtime needed.
- **20+ hand-written WGSL compute shaders** implement every operation from scratch: embedding lookup, layer normalization, matrix multiplication, multi-head attention, Conv1d, ConvTranspose1d, LSTM, instance normalization, adaptive instance normalization (AdaIN), GELU, LeakyReLU, Snake activation, softmax, iSTFT, and more.
- **Buffer pooling and memory management** tuned for iOS Safari's Metal backend, keeping peak GPU memory under control to prevent jetsam kills on mobile devices.

The result is a lean, zero-bloat inference engine with no framework overhead.

## Browser Support

| Browser                | Status              |
|------------------------|---------------------|
| Chrome 113+            | Supported           |
| Edge 113+              | Supported           |
| Safari 26+ (macOS)     | Supported (WebGPU)  |
| Safari 26+ (iOS)       | Supported (WebGPU)  |
| Firefox Nightly        | Experimental        |

WebGPU is required. Safari gained WebGPU support in version 26 (iOS 26 / macOS 26), making this the first generation of iPhones and iPads that can run full neural network inference natively in the browser.

## Performance

Benchmarked on real hardware, measuring time from `generate()` call to waveform output (excludes one-time model loading):

| Device                  | Time     |
|-------------------------|----------|
| iPhone 17 Pro Max       | ~1.24s   |
| Desktop GPU (M-series)  | < 1s     |

Model loading (first run only): ~2-4s depending on network speed. The 74.6 MB ONNX model and 3.1 MB voice embeddings are fetched from HuggingFace CDN.

## Development

```bash
git clone https://github.com/svenflow/kitten-tts-webgpu.git
cd kitten-tts-webgpu
npm install
npm run dev
```

For local development with model files, place them in `models/kitten-tts-mini-0.8/`:
- `kitten_tts_mini_v0_8.onnx` (74.6 MB)
- `voices.npz` (3.1 MB)

### Build

```bash
npm run build
npm run preview
```

### Tests

```bash
npm test
```

## Credits

- [Kitten TTS](https://huggingface.co/KittenML/kitten-tts-mini-0.8) model by KittenML
- [phonemizer](https://www.npmjs.com/package/phonemizer) (espeak-ng WASM) by Xenova
- WebGPU inference engine by [svenflow](https://github.com/svenflow)

## License

MIT
