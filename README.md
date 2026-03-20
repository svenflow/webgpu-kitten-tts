# WebGPU Kitten TTS

Run [Kitten TTS](https://huggingface.co/KittenML/kitten-tts-mini-0.8) (80M parameters) locally in the browser via WebGPU. Zero runtime dependencies.

**[Live Demo →](https://svenflow.github.io/webgpu-kitten-tts/)**

## Features

- Full TTS inference pipeline running entirely on the GPU via WebGPU compute shaders
- 74.6 MB model loaded from HuggingFace CDN
- 8 voices (4F/4M) with adjustable speed
- Zero runtime dependencies — only Vite for bundling
- No WASM, no ONNX Runtime — pure WebGPU compute

## Requirements

- A browser with WebGPU support (Chrome 113+, Edge 113+, or Firefox Nightly)
- A GPU with WebGPU driver support

## Development

```bash
npm install
npm run dev
```

Place model files in `models/kitten-tts-mini-0.8/` for local development:
- `kitten_tts_mini_v0_8.onnx` (74.6 MB)
- `voices.npz` (3.1 MB)

## Build

```bash
npm run build
npm run preview
```

## License

MIT
