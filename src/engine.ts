/**
 * WebGPU inference engine for Kitten TTS V0.8 (80M).
 *
 * Runs the full TTS pipeline on the GPU:
 * phoneme_ids + style + speed → waveform (24kHz audio)
 */

import { DEFAULT_CONFIG, type GpuTensor, type KittenConfig } from './types.js';
import { OnnxParser, parseNpz, float16ToFloat32, dequantizeInt8, dequantizeUint8 } from './onnx.js';
import {
  embeddingShader, layerNormShader, matmulShader, conv1dShader,
  instanceNormShader, adainShader, adainRowMajorShader, convTranspose1dShader,
  depthwiseConvTranspose1dShader, resize1dShader,
  leakyReluShader, geluShader, softmaxShader, addShader, scaleShader,
  concatChannelsShader, concatBroadcastShader, reflectionPad1dShader, alphaResidualShader, snakeShader,
  tanhShader, sigmoidShader,
  mhaShader, matmulGeluShader, transposeShader, lstmShader, istftShader,
  expandRowMajorShader, expandChannelFirstShader,
} from './shaders.js';

interface CompiledPipeline {
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
}

export class KittenTTSEngine {
  private device!: GPUDevice;
  private weights: Map<string, GpuTensor> = new Map();
  private pipelines: Map<string, CompiledPipeline> = new Map();
  private voices: Map<string, Float32Array> = new Map(); // [400, 256] per voice
  private config: KittenConfig;
  /** Uniform buffers created during dispatch, cleaned up after submit. */
  private pendingUniformBuffers: GPUBuffer[] = [];

  /** Pending command buffers for batch submission. */
  private pendingCommandBuffers: GPUCommandBuffer[] = [];

  /** Cached CPU copies of sin generator weights (avoid readBuffer every inference). */
  private sinGenWeights: {
    linearWeight: Float32Array;   // [9] — harmonic collapse linear layer
    linearBias: Float32Array;     // [1]
    fwdReal: Float32Array;        // [11*20] — forward STFT real basis
    fwdImag: Float32Array;        // [11*20] — forward STFT imag basis
  } | null = null;

  /** Debug mode: when true, intermediate activations are captured for comparison. */
  public debugCapture = false;
  /** Captured activations (name → {data, shape}). Only populated when debugCapture=true. */
  public debugActivations: Map<string, { data: Float32Array; shape: number[] }> = new Map();
  private debugBertBuffers: {
    normedEmb: GPUBuffer; projected: GPUBuffer | null; hidden: GPUBuffer;
    seqLen: number; embedDim: number; hiddenSize: number;
    layerBuffers: Map<string, GPUBuffer>;
  } | null = null;

  /** Performance profiling: when true, logs timing per pipeline stage. */
  public profile = false;
  private timings: Map<string, number> = new Map();
  private _stageStart = 0;

  constructor(config: KittenConfig = DEFAULT_CONFIG) {
    this.config = config;
  }

  /** Start timing a pipeline stage. Call endStage() to record. */
  private startStage(): void {
    if (!this.profile) return;
    this._stageStart = performance.now();
  }

  /** End timing and record the stage duration (includes GPU sync). */
  private async endStage(name: string): Promise<void> {
    if (!this.profile) return;
    // Flush any batched dispatches and wait for completion
    this.flushBatchEncoder();
    await this.device.queue.onSubmittedWorkDone();
    const elapsed = performance.now() - this._stageStart;
    this.timings.set(name, elapsed);
  }

  /** Last timing report from generate(), available after each call. */
  public lastTimings: { name: string; ms: number }[] = [];

  /** Print timing summary to console and store for external access. */
  private printTimings(): void {
    if (!this.profile) return;
    this.lastTimings = [];
    let total = 0;
    const lines: string[] = [];
    for (const [name, ms] of this.timings) {
      total += ms;
      this.lastTimings.push({ name, ms });
      lines.push(`  ${name.padEnd(35)} ${ms.toFixed(1).padStart(8)} ms`);
    }
    console.log(`\n[KittenTTS] ── Timing Report ──`);
    for (const line of lines) console.log(line);
    console.log(`  ${'─'.repeat(45)}`);
    console.log(`  ${'TOTAL'.padEnd(35)} ${total.toFixed(1).padStart(8)} ms`);
    this.timings.clear();
  }

  /** Capture a GPU buffer's contents as a named debug activation. No-op when debugCapture is off. */
  private async captureDebug(name: string, buffer: GPUBuffer, shape: number[]): Promise<void> {
    if (!this.debugCapture) return; // Early return — skips all GPU reads
    // Must flush any batched work before reading back
    this.endBatch();
    const size = shape.reduce((a, b) => a * b, 1);
    const data = await this.readBuffer(buffer, size);
    this.debugActivations.set(name, { data, shape });
    let min = Infinity, max = -Infinity, nanCount = 0;
    for (let i = 0; i < data.length; i++) {
      if (isNaN(data[i])) { nanCount++; continue; }
      if (data[i] < min) min = data[i];
      if (data[i] > max) max = data[i];
    }
    console.log(`[DEBUG] Captured ${name}: shape=[${shape}], range=[${min}, ${max}], NaN=${nanCount}/${data.length}`);
  }

  /** Initialize WebGPU device and compile shaders. */
  async init(): Promise<void> {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU not available');

    // Query adapter limits and request what's available (iOS has lower limits)
    const adapterLimits = adapter.limits;
    const maxBuf = Math.min(256 * 1024 * 1024, adapterLimits.maxStorageBufferBindingSize);
    const maxSize = Math.min(256 * 1024 * 1024, adapterLimits.maxBufferSize);
    console.log(`[KittenTTS] Adapter limits: maxStorageBuffer=${maxBuf}, maxBuffer=${maxSize}`);

    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: maxBuf,
        maxBufferSize: maxSize,
      },
    });

    // Handle device loss gracefully
    this.device.lost.then((info) => {
      console.error(`[KittenTTS] Device lost: ${info.reason} — ${info.message}`);
      window.dispatchEvent(new CustomEvent('webgpu-device-lost', { detail: info }));
    });

    // Catch uncaptured GPU errors (shader compilation failures, OOM, etc.)
    this.device.addEventListener('uncapturederror', (event: Event) => {
      const e = event as GPUUncapturedErrorEvent;
      console.error(`[KittenTTS] GPU error: ${e.error.message}`);
      window.dispatchEvent(new CustomEvent('webgpu-error', { detail: e.error.message }));
    });

    this.compileShaders();
    console.log('[KittenTTS] WebGPU device initialized');
  }

  /** Load model weights from ONNX file and voices from NPZ. */
  async loadModel(onnxUrl: string, voicesUrl: string): Promise<void> {
    console.log('[KittenTTS] Loading model...');

    // Load ONNX weights
    const onnxBuffer = await fetch(onnxUrl).then(r => r.arrayBuffer());
    const parser = new OnnxParser(onnxBuffer);
    const tensors = parser.parseInitializers();

    console.log(`[KittenTTS] Parsed ${tensors.size} weight tensors`);

    // Helper to find scale/zero_point for a quantized weight tensor.
    // Supports both scalar and per-axis quantization (e.g. LSTM weights with [2] scales).
    //
    // Naming conventions:
    //   onnx::MatMul_5884_quantized → onnx::MatMul_5884_scale, onnx::MatMul_5884_zero_point
    //   kmodel.foo.weight_quantized → kmodel.foo.weight_scale, kmodel.foo.weight_zero_point
    const getScaleZp = (qName: string): { scales: Float32Array; zeroPoints: Int32Array } => {
      let baseName: string;
      if (qName.endsWith('_quantized')) {
        baseName = qName.slice(0, -'_quantized'.length);
      } else {
        baseName = qName;
      }
      const scaleTensor = tensors.get(`${baseName}_scale`);
      const zpTensor = tensors.get(`${baseName}_zero_point`);

      let scales = new Float32Array([1.0]);
      let zeroPoints = new Int32Array([0]);

      if (scaleTensor && scaleTensor.rawData.length >= 4) {
        const numScales = scaleTensor.rawData.length / 4;
        // Copy to aligned buffer (protobuf data may not be 4-byte aligned)
        const aligned = new Uint8Array(numScales * 4);
        aligned.set(scaleTensor.rawData.subarray(0, numScales * 4));
        scales = new Float32Array(aligned.buffer);
      }
      if (zpTensor && zpTensor.rawData.length >= 1) {
        // Zero-point data may be stored as:
        //   1. raw_data: 1 byte per value for INT8/UINT8 (rawLen == numElements)
        //   2. int32_data: 4 bytes per value (rawLen == numElements * 4)
        // Detect by comparing rawData length to expected element count.
        const numElements = zpTensor.dims.length === 0 ? 1 : zpTensor.dims.reduce((a: number, b: number) => a * b, 1);
        const rawLen = zpTensor.rawData.length;
        const isInt32Encoded = (rawLen === numElements * 4) && rawLen !== numElements;

        if (isInt32Encoded) {
          // Data from int32_data field: read as int32 values
          const aligned = new Uint8Array(rawLen);
          aligned.set(zpTensor.rawData.subarray(0, rawLen));
          const i32 = new Int32Array(aligned.buffer);
          zeroPoints = new Int32Array(numElements);
          for (let j = 0; j < numElements; j++) zeroPoints[j] = i32[j];
        } else {
          // Data from raw_data: 1 byte per INT8/UINT8 value
          const numZp = rawLen;
          zeroPoints = new Int32Array(numZp);
          if (zpTensor.dataType === 3) { // INT8
            const i8 = new Int8Array(zpTensor.rawData.buffer, zpTensor.rawData.byteOffset, numZp);
            for (let j = 0; j < numZp; j++) zeroPoints[j] = i8[j];
          } else { // UINT8
            for (let j = 0; j < numZp; j++) zeroPoints[j] = zpTensor.rawData[j];
          }
        }
      }

      return { scales, zeroPoints };
    };

    // Upload weights to GPU, dequantizing quantized weights at load time.
    // This avoids needing DynamicQuantizeLinear + MatMulInteger on GPU —
    // we just do float matmul with pre-dequantized weights.
    for (const [name, tensor] of tensors) {
      // Skip scale/zero_point metadata tensors (used during dequantization above)
      if (name.endsWith('_scale') || name.endsWith('_zero_point')) continue;

      let f32Data: Float32Array;
      const totalElements = tensor.dims.reduce((a, b) => a * b, 1);

      if (totalElements === 0) continue;

      // Skip tensors with no raw data (e.g. external data references not supported)
      if (tensor.rawData.length === 0) {
        console.warn(`[KittenTTS] Skipping ${name}: no raw data (dims=${tensor.dims}, dtype=${tensor.dataType})`);
        continue;
      }

      try {

      switch (tensor.dataType) {
        case 1: { // FLOAT32
          // Copy to aligned buffer (protobuf data may not be 4-byte aligned)
          const aligned = new Uint8Array(totalElements * 4);
          aligned.set(tensor.rawData.subarray(0, totalElements * 4));
          f32Data = new Float32Array(aligned.buffer);
          break;
        }
        case 10: { // FLOAT16
          // Copy to aligned buffer (protobuf data may not be 2-byte aligned)
          const aligned16 = new Uint8Array(totalElements * 2);
          aligned16.set(tensor.rawData.subarray(0, totalElements * 2));
          f32Data = float16ToFloat32(new Uint16Array(aligned16.buffer));
          break;
        }
        case 3: { // INT8 — dequantize using scale/zero_point (scalar or per-axis)
          const { scales, zeroPoints } = getScaleZp(name);
          f32Data = new Float32Array(totalElements);
          // Int8Array doesn't need alignment, but use subarray for consistency
          const i8 = new Int8Array(tensor.rawData.buffer, tensor.rawData.byteOffset, totalElements);
          if (scales.length === 1) {
            // Scalar quantization
            const s = scales[0], zp = zeroPoints[0];
            for (let j = 0; j < totalElements; j++) {
              f32Data[j] = (i8[j] - zp) * s;
            }
          } else {
            // Per-axis quantization (axis 0): e.g. LSTM [2, H, 4H] with scale [2]
            const axisSize = totalElements / scales.length;
            for (let a = 0; a < scales.length; a++) {
              const s = scales[a], zp = zeroPoints[a];
              const offset = a * axisSize;
              for (let j = 0; j < axisSize; j++) {
                f32Data[offset + j] = (i8[offset + j] - zp) * s;
              }
            }
          }
          if (scales[0] !== 1.0) {
            console.log(`[KittenTTS] Dequantized INT8 ${name}: scales=[${Array.from(scales).map(s => s.toFixed(6)).join(',')}]`);
          }
          break;
        }
        case 2: { // UINT8 — dequantize using scale/zero_point (scalar or per-axis)
          const { scales, zeroPoints } = getScaleZp(name);
          f32Data = new Float32Array(totalElements);
          if (scales.length === 1) {
            const s = scales[0], zp = zeroPoints[0];
            for (let j = 0; j < totalElements; j++) {
              f32Data[j] = (tensor.rawData[j] - zp) * s;
            }
          } else {
            const axisSize = totalElements / scales.length;
            for (let a = 0; a < scales.length; a++) {
              const s = scales[a], zp = zeroPoints[a];
              const offset = a * axisSize;
              for (let j = 0; j < axisSize; j++) {
                f32Data[offset + j] = (tensor.rawData[offset + j] - zp) * s;
              }
            }
          }
          console.log(`[KittenTTS] Dequantized UINT8 ${name}: scales=[${Array.from(scales).map(s => s.toFixed(6)).join(',')}], zp=[${Array.from(zeroPoints).join(',')}], f32[0..5]=[${Array.from(f32Data.subarray(0, 5)).map(v => v.toFixed(6)).join(',')}]`);
          break;
        }
        case 7: // INT64
          // Skip int64 tensors (shapes, indices)
          continue;
        default:
          console.warn(`[KittenTTS] Skipping ${name}: unsupported dtype ${tensor.dataType}`);
          continue;
      }

      const gpuBuffer = this.device.createBuffer({
        size: f32Data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        label: name,
      });
      this.device.queue.writeBuffer(gpuBuffer, 0, f32Data as unknown as ArrayBuffer);
      this.weights.set(name, { buffer: gpuBuffer, shape: tensor.dims, size: totalElements });
      } catch (e) {
        console.error(`[KittenTTS] Error processing tensor ${name} (dims=${tensor.dims}, dtype=${tensor.dataType}, rawLen=${tensor.rawData.length}, totalEl=${totalElements}):`, e);
        throw e;
      }
    }

    console.log(`[KittenTTS] Uploaded ${this.weights.size} weight buffers to GPU`);

    // Load voices
    const voicesBuffer = await fetch(voicesUrl).then(r => r.arrayBuffer());
    const voiceData = await parseNpz(voicesBuffer);
    for (const [name, { shape, data }] of voiceData) {
      this.voices.set(name, data);
      console.log(`[KittenTTS] Loaded voice: ${name} (${shape})`);
    }
  }

  /** Run TTS inference. */
  async generate(
    inputIds: number[],
    voice: string = 'Bella',
    speed: number = 1.0,
    textLength?: number, // Raw text character count for voice style selection
    onProgress?: (stage: string) => void,
  ): Promise<{ waveform: Float32Array; duration: Int32Array }> {
    // Resolve voice alias
    const voiceKey = this.config.voiceAliases[voice] || voice;
    const voiceEmbeddings = this.voices.get(voiceKey);
    if (!voiceEmbeddings) {
      throw new Error(`Voice not found: ${voice} (${voiceKey})`);
    }

    // Pick style vector based on raw text character length (matches official kittentts package)
    // ref_id = min(len(text), 399) — longer texts get different style conditioning
    const refId = Math.min(textLength ?? inputIds.length, 399);
    const styleVec = voiceEmbeddings.subarray(refId * 256, (refId + 1) * 256);

    // Upload inputs to GPU
    const inputIdsBuf = this.createBuffer(
      new Int32Array(inputIds),
      'input_ids',
      GPUBufferUsage.STORAGE
    );
    const styleBuf = this.createBuffer(
      styleVec,
      'style',
      GPUBufferUsage.STORAGE
    );
    const speedBuf = this.createBuffer(
      new Float32Array([speed]),
      'speed',
      GPUBufferUsage.STORAGE
    );

    const seqLen = inputIds.length;

    // ── 1. BERT/ALBERT Embedding ──────────────────────────────────────────
    this.startStage();
    onProgress?.('1/8 BERT embedding'); console.log('[KittenTTS] Running BERT embedding...');
    const wordEmbWeight = this.requireWeight('kmodel.bert.embeddings.word_embeddings.weight');
    const posEmbWeight = this.requireWeight('kmodel.bert.embeddings.position_embeddings.weight');
    const tokenTypeEmbWeight = this.requireWeight('kmodel.bert.embeddings.token_type_embeddings.weight');

    const embedDim = 128;
    const embeddingOut = this.createEmptyBuffer(seqLen * embedDim, 'bert_embedding');

    // Word embedding lookup
    this.dispatchEmbedding(wordEmbWeight.buffer, inputIdsBuf, embeddingOut, seqLen, embedDim, 178);

    // Debug: BERT word embedding (before position/token_type add) — matches /bert/embeddings/word_embeddings/Gather_output_0 [1, seqLen, 128]
    await this.captureDebug('/bert/embeddings/word_embeddings/Gather_output_0', embeddingOut, [1, seqLen, embedDim]);

    // Token type embedding (all zeros = segment 0 for TTS)
    const tokenTypeIds = this.createBuffer(
      new Int32Array(seqLen), // all zeros
      'token_type_ids',
      GPUBufferUsage.STORAGE
    );
    const tokenTypeEmbOut = this.createEmptyBuffer(seqLen * embedDim, 'token_type_embedding');
    this.dispatchEmbedding(tokenTypeEmbWeight.buffer, tokenTypeIds, tokenTypeEmbOut, seqLen, embedDim, 2);

    // Add word + token_type embeddings
    const wordPlusTokenType = this.createEmptyBuffer(seqLen * embedDim, 'bert_emb_wtt');
    this.dispatchAdd(embeddingOut, tokenTypeEmbOut, wordPlusTokenType, seqLen * embedDim);

    // Position embeddings
    const posIds = this.createBuffer(
      new Int32Array(Array.from({ length: seqLen }, (_, i) => i)),
      'pos_ids',
      GPUBufferUsage.STORAGE
    );
    const posEmbOut = this.createEmptyBuffer(seqLen * embedDim, 'pos_embedding');
    this.dispatchEmbedding(posEmbWeight.buffer, posIds, posEmbOut, seqLen, embedDim, 512);

    // Add (word + token_type) + position embeddings
    const bertEmbedding = this.createEmptyBuffer(seqLen * embedDim, 'bert_emb_sum');
    this.dispatchAdd(wordPlusTokenType, posEmbOut, bertEmbedding, seqLen * embedDim);

    // Debug: capture embedding sum (word + position, before LayerNorm)
    await this.captureDebug('bert/emb_sum', bertEmbedding, [1, seqLen, embedDim]);

    await this.endStage('BERT embedding');

    // ── 2. ALBERT Encoder (shared layers) ─────────────────────────────────
    this.startStage();
    onProgress?.('2/8 ALBERT encoder'); console.log('[KittenTTS] Running ALBERT encoder...');
    // Flush pending embedding dispatches before BERT creates its own encoder
    this.flushBatchEncoder();
    const bertOutput = this.runBertEncoder(inputIdsBuf, bertEmbedding, seqLen);
    console.log(`[KittenTTS] BERT encoder output: [1, ${seqLen}, 768]`);
    // Debug: capture BERT encoder intermediates with ONNX-matching names
    if (this.debugBertBuffers) {
      const db = this.debugBertBuffers;
      // Embedding LayerNorm → [1, seqLen, 128]
      await this.captureDebug('bert/emb_ln', db.normedEmb, [1, db.seqLen, db.embedDim]);
      // 128→768 projection → [1, seqLen, 768]
      if (db.projected) {
        await this.captureDebug('bert/proj_128_768', db.projected, [1, db.seqLen, db.hiddenSize]);
      }
      // Layer 0 intermediates
      for (const [label, buf] of db.layerBuffers) {
        if (label === 'attn_ln_0') {
          await this.captureDebug('bert/attn_ln_0', buf, [1, db.seqLen, db.hiddenSize]);
        } else if (label === 'hidden_1') {
          await this.captureDebug('bert/full_ln_0', buf, [1, db.seqLen, db.hiddenSize]);
        } else if (label === 'hidden_2') {
          await this.captureDebug('bert/full_ln_1', buf, [1, db.seqLen, db.hiddenSize]);
        }
      }
      // Final BERT output (iter 11 LN) → [1, seqLen, 768]
      await this.captureDebug('bert/final', db.hidden, [1, db.seqLen, db.hiddenSize]);
    }

    await this.endStage('ALBERT encoder (12 iters)');

    // ── 3. Text Encoder (CNN + LSTM) ──────────────────────────────────────
    this.startStage();
    onProgress?.('3/8 Text encoder'); console.log('[KittenTTS] Running text encoder...');
    const textEncoderOutput = await this.runTextEncoder(inputIdsBuf, seqLen);
    console.log(`[KittenTTS] Text encoder output: [${seqLen}, 2, 256]`);

    await this.endStage('Text encoder (CNN+LSTM)');

    // ── 4. Predictor Text Encoder (6 stacked bidir LSTMs + FC layers) ────
    this.startStage();
    onProgress?.('4/8 Predictor encoder'); console.log('[KittenTTS] Running predictor text encoder...');

    // BERT output → project 768 → 512 for predictor input
    // ONNX: /bert_encoder/MatMul (768→512) + /bert_encoder/Add (bias)
    const bertProjWeight = this.requireWeight('onnx::MatMul_6040_quantized');
    const bertProjBias = this.requireWeight('kmodel.bert_encoder.bias');
    const bertProjOut = this.createEmptyBuffer(seqLen * 512, 'bert_proj_512');
    this.dispatchMatmul(bertOutput, bertProjWeight.buffer, bertProjBias.buffer, bertProjOut, seqLen, 768, 512, true);
    await this.captureDebug('bert/encoder_proj', bertProjOut, [1, seqLen, 512]);

    // The predictor text encoder takes text encoder LSTM output [seqLen, 512]
    // and BERT projected features [seqLen, 512], concatenated → [seqLen, 640]
    // Actually the LSTM input is 640 = 512 (text) + 128 (style[:128] broadcast or FC output)
    // For LSTM.0, input = concat(text_encoder_output[512], bert_proj_slice[128])
    //
    // TODO: Implement the 6-layer predictor text encoder LSTM stack:
    //   LSTM.0: W=[2,640,1024] R=[2,256,1024] B=[2,2048] — onnx::LSTM_6094/6095/6093
    //   FC.1:   weight=[128,1024] bias=[1024] — kmodel.predictor.text_encoder.lstms.1.fc
    //   LSTM.2: W=[2,640,1024] R=[2,256,1024] B=[2,2048] — onnx::LSTM_6144/6145/6143
    //   FC.3:   weight=[128,1024] bias=[1024] — kmodel.predictor.text_encoder.lstms.3.fc
    //   LSTM.4: W=[2,640,1024] R=[2,256,1024] B=[2,2048] — onnx::LSTM_6194/6195/6193
    //   FC.5:   weight=[128,1024] bias=[1024] — kmodel.predictor.text_encoder.lstms.5.fc
    //
    // Each LSTM outputs [seqLen, 2, 256] = [seqLen, 512]
    // Each FC takes [seqLen, 512] → [seqLen, 1024] (projecting both directions)
    //   then the 1024 is reshaped/split to [seqLen, 128] (FC output) to concat with text features
    //
    // For now, run the predictor LSTM.0 + shared LSTM as a simplified path.
    // The predictor text encoder LSTMs refine the text features before duration prediction.

    // ── Predictor text encoder: 3 LSTM + FC pairs ──
    const predLstmConfigs = [
      { W: 'onnx::LSTM_6094_quantized', R: 'onnx::LSTM_6095_quantized', B: 'onnx::LSTM_6093' },
      { W: 'onnx::LSTM_6144_quantized', R: 'onnx::LSTM_6145_quantized', B: 'onnx::LSTM_6143' },
      { W: 'onnx::LSTM_6194_quantized', R: 'onnx::LSTM_6195_quantized', B: 'onnx::LSTM_6193' },
    ];
    const predFcConfigs = [
      { weight: 'kmodel.predictor.text_encoder.lstms.1.fc.weight_quantized', bias: 'kmodel.predictor.text_encoder.lstms.1.fc.bias' },
      { weight: 'kmodel.predictor.text_encoder.lstms.3.fc.weight_quantized', bias: 'kmodel.predictor.text_encoder.lstms.3.fc.bias' },
      { weight: 'kmodel.predictor.text_encoder.lstms.5.fc.weight_quantized', bias: 'kmodel.predictor.text_encoder.lstms.5.fc.bias' },
    ];

    // ── Style vector split (needed before predictor text encoder) ──
    // ONNX Slice_1 (node 72): style[:, 128:256] → predictors (N/F0 AdaIN + text encoder FC + LSTM concat)
    // ONNX Slice (node 73): style[:, 0:128] → decoder (AdaIN conditioning)
    const stylePredData = styleVec.subarray(128, 256); // [128] raw CPU data
    const stylePred = this.createBuffer(
      stylePredData,
      'style_pred',
      GPUBufferUsage.STORAGE
    );
    const styleDec = this.createBuffer(
      styleVec.subarray(0, 128),
      'style_dec',
      GPUBufferUsage.STORAGE
    );

    // Predictor text encoder: 3 LSTM + AdaIN pairs
    // Each LSTM takes concat(text_features[512], style_pred[128]) = [seqLen, 640]
    // Each FC layer is an AdaIN generator: style_pred[1,128] → [1,1024] → split to gamma[512]+beta[512]
    // AdaIN: gamma * LayerNorm(lstm_output) + beta → [seqLen, 512]
    const predIntermediates: GPUBuffer[] = [bertProjOut];

    // Initial text features = BERT encoder projected output [seqLen, 512]
    // Confirmed by ONNX graph: lstms.0 X input traces back to bert_encoder linear output,
    // NOT the text encoder LSTM output. The Concat_1 node concatenates bert_proj[512] + style[:128].
    let predTextFeatures = bertProjOut;

    for (let li = 0; li < 3; li++) {
      const cfg = predLstmConfigs[li];
      const fcCfg = predFcConfigs[li];

      const lstmInputSize = 640;
      const lstmHidden = 256;
      const lstmW = this.requireWeight(cfg.W);
      const lstmR = this.requireWeight(cfg.R);
      const lstmB = this.requireWeight(cfg.B);

      // LSTM input: concat(text_features[seqLen,512], style_pred[128]) on GPU → [seqLen, 640]
      const lstmInput = this.createEmptyBuffer(seqLen * lstmInputSize, `pred_lstm${li}_in`);
      this.dispatchConcatBroadcast(predTextFeatures, stylePred, lstmInput, seqLen, 512, 128);

      // GPU LSTM
      const lstmOut = this.createEmptyBuffer(seqLen * 2 * lstmHidden, `pred_lstm${li}_out`);
      this.dispatchLSTM(lstmInput, lstmW.buffer, lstmR.buffer, lstmB.buffer, lstmOut, seqLen, lstmInputSize, lstmHidden, 2);
      predIntermediates.push(lstmInput);

      // Debug: predictor LSTM output — matches /text_encoder/lstms.{2*li}/LSTM_output_0
      const predLstmIdx = 2 * li;
      await this.captureDebug(`/text_encoder/lstms.${predLstmIdx}/LSTM_output_0`, lstmOut, [seqLen, 2, 1, lstmHidden]);

      // FC layer: style_pred[1,128] → [1,1024] (AdaIN parameters: gamma[:512]+1, beta[512:])
      const fcWeight = this.requireWeight(fcCfg.weight);
      const fcBias = this.requireWeight(fcCfg.bias);
      const fcOut = this.createEmptyBuffer(1024, `pred_fc${li}_out`);
      this.dispatchMatmul(stylePred, fcWeight.buffer, fcBias.buffer, fcOut, 1, 128, 1024, true);

      // Debug: FC output
      await this.captureDebug(`/text_encoder/lstms.${2 * li + 1}/fc/Gemm_output_0`, fcOut, [1, 1024]);

      // LayerNorm weights
      const lnIdx = 2 * li + 1;
      let lnGammaName = `kmodel.predictor.text_encoder.lstms.${lnIdx}.norm.weight`;
      let lnBetaName = `kmodel.predictor.text_encoder.lstms.${lnIdx}.norm.bias`;
      if (!this.weights.has(lnGammaName)) {
        lnGammaName = `/text_encoder/lstms.${lnIdx}/Constant_7_output_0`;
        lnBetaName = `/text_encoder/lstms.${lnIdx}/Constant_8_output_0`;
      }
      const lnGamma = this.requireWeight(lnGammaName);
      const lnBeta = this.requireWeight(lnBetaName);

      // All-GPU: LayerNorm(lstm_output) → AdaIN(normed, fc_out) → predTextFeatures
      // LayerNorm: LSTM output [seqLen, 512] → normed [seqLen, 512]
      const normedLstm = this.createEmptyBuffer(seqLen * 512, `pred_ln${li}_out`);
      this.dispatchLayerNorm(lstmOut, lnGamma.buffer, lnBeta.buffer, normedLstm, seqLen, 512, 1e-5);
      predIntermediates.push(lstmOut);

      // AdaIN row-major: normed[seqLen,512] × fcOut[1024] → adainOut[seqLen,512]
      const adainOut = this.createEmptyBuffer(seqLen * 512, `pred_adain${li}_out`);
      this.dispatchAdaINRowMajor(normedLstm, fcOut, adainOut, 512, seqLen);
      predIntermediates.push(normedLstm, fcOut);

      // Debug: AdaIN output — matches /text_encoder/lstms.{2*li+1}/Add_2_output_0
      await this.captureDebug(`/text_encoder/lstms.${2 * li + 1}/Add_2_output_0`, adainOut, [1, seqLen, 512]);

      predIntermediates.push(predTextFeatures);
      predTextFeatures = adainOut;
      predIntermediates.push(lstmOut);
    }

    // After predictor text encoder, we have predTextFeatures [seqLen, 512]
    console.log(`[KittenTTS] Predictor text encoder output: [${seqLen}, 512]`);

    await this.endStage('Predictor text encoder');

    // ── 5. Duration Prediction + Length Expansion ──────────────────────────
    this.startStage();
    onProgress?.('5/8 Duration predictor'); console.log('[KittenTTS] Running duration predictor...');

    // ── Duration LSTM: predictor text encoder output → durations ──
    // /lstm LSTM takes concat(pred_text_features[512], style_pred[128]) = [seqLen, 640]
    // Weights: onnx::LSTM_6243/6244/6242 (separate from shared LSTM 6292/6293/6291)

    // Build concat input on GPU: predTextFeatures[seqLen,512] + stylePred[128] → [seqLen,640]
    const durationLstmInput = this.createEmptyBuffer(seqLen * 640, 'duration_lstm_in');
    this.dispatchConcatBroadcast(predTextFeatures, stylePred, durationLstmInput, seqLen, 512, 128);

    const durationLstmW = this.requireWeight('onnx::LSTM_6243_quantized');
    const durationLstmR = this.requireWeight('onnx::LSTM_6244_quantized');
    const durationLstmB = this.requireWeight('onnx::LSTM_6242');

    // GPU LSTM for duration predictor (was cpuLSTM — switching to GPU dispatchLSTM)
    const durationLstmOut = this.createEmptyBuffer(seqLen * 2 * 256, 'duration_lstm_out');
    this.dispatchLSTM(durationLstmInput, durationLstmW.buffer, durationLstmR.buffer, durationLstmB.buffer, durationLstmOut, seqLen, 640, 256, 2);
    predIntermediates.push(durationLstmInput);

    // Debug: duration LSTM output — matches /lstm/LSTM_output_0
    await this.captureDebug('/lstm/LSTM_output_0', durationLstmOut, [seqLen, 2, 1, 256]);

    // Debug: duration LSTM input
    await this.captureDebug('/lstm/Transpose_output_0', durationLstmInput, [seqLen, 1, 640]);

    // Duration projection: [seqLen, 512] × [512, 50] + bias[50] → sigmoid → ReduceSum(axis=-1)
    // LSTM output is [seqLen, 2, 256] = [seqLen, 512] in memory
    const durProjWeight = this.requireWeight('onnx::MatMul_6245');
    const durProjBias = this.requireWeight('kmodel.predictor.duration_proj.linear_layer.bias');
    const durProjOut = this.createEmptyBuffer(seqLen * 50, 'dur_proj');
    this.dispatchMatmul(durationLstmOut, durProjWeight.buffer, durProjBias.buffer, durProjOut, seqLen, 512, 50, true);
    predIntermediates.push(durationLstmOut);

    // Debug: duration projection output (before sigmoid)
    await this.captureDebug('/duration_proj/linear_layer/Add_output_0', durProjOut, [1, seqLen, 50]);

    // Sigmoid + ReduceSum(axis=-1) → [seqLen] fractional durations
    const durSigmoid = this.createEmptyBuffer(seqLen * 50, 'dur_sigmoid');
    this.dispatchSigmoid(durProjOut, durSigmoid, seqLen * 50);
    predIntermediates.push(durProjOut);

    // Debug: sigmoid output
    await this.captureDebug('/Sigmoid_output_0', durSigmoid, [1, seqLen, 50]);

    // Read back only sigmoid data (need durations to compute totalFrames)
    predIntermediates.push(durSigmoid);
    const durSigmoidData = await this.readBuffer(durSigmoid, seqLen * 50);

    const durations = new Int32Array(seqLen);
    const cumsum = new Uint32Array(seqLen);
    let totalFrames = 0;
    for (let i = 0; i < seqLen; i++) {
      let sum = 0;
      for (let j = 0; j < 50; j++) {
        sum += durSigmoidData[i * 50 + j];
      }
      // Divide by speed, round, clip(min=0)
      durations[i] = Math.max(0, Math.round(sum / speed));
      totalFrames += durations[i];
      cumsum[i] = totalFrames; // prefix sum for GPU expansion
    }
    console.log(`[KittenTTS] Duration prediction: durations=[${Array.from(durations).join(',')}] totalFrames=${totalFrames}`);

    // Debug: duration prediction
    if (this.debugCapture) {
      this.debugActivations.set('duration', {
        data: new Float32Array(durations),
        shape: [seqLen],
      });
    }

    // Upload cumsum to GPU for length expansion shaders
    const cumsumBuf = this.createBuffer(cumsum, 'duration_cumsum', GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);

    // GPU length expansion: durationLstmInput [seqLen, 640] → [totalFrames, 640] row-major
    const sharedLstmInput = this.createEmptyBuffer(totalFrames * 640, 'shared_lstm_in');
    this.dispatchExpandRowMajor(durationLstmInput, cumsumBuf, sharedLstmInput, seqLen, 640, totalFrames);

    // GPU length expansion with transpose: textEncoderOutput [seqLen, 512] → [512, totalFrames] channel-first
    // CRITICAL: Decoder input uses the BASE text encoder LSTM output (text_encoder/lstm),
    // NOT the predictor text encoder output. The ONNX model's MatMul_1 multiplies the
    // base text encoder LSTM output with the alignment matrix to produce expanded features.
    const expandedTextFeatures = this.createEmptyBuffer(512 * totalFrames, 'expanded_text_features');
    this.dispatchExpandChannelFirst(textEncoderOutput, cumsumBuf, expandedTextFeatures, seqLen, 512, totalFrames);
    textEncoderOutput.destroy(); // No longer needed after expansion
    predIntermediates.push(cumsumBuf);
    console.log(`[KittenTTS] Expanded features: [640, ${totalFrames}] for shared LSTM, [512, ${totalFrames}] for decoder`);

    // ── Shared LSTM: expanded features [totalFrames, 640] → [totalFrames, 512] ──
    const sharedLstmW = this.requireWeight('onnx::LSTM_6292_quantized');
    const sharedLstmR = this.requireWeight('onnx::LSTM_6293_quantized');
    const sharedLstmB = this.requireWeight('onnx::LSTM_6291');
    const sharedLstmOut = this.createEmptyBuffer(totalFrames * 2 * 256, 'shared_lstm_out');
    this.dispatchLSTM(sharedLstmInput, sharedLstmW.buffer, sharedLstmR.buffer, sharedLstmB.buffer, sharedLstmOut, totalFrames, 640, 256, 2);
    predIntermediates.push(sharedLstmInput);

    // Debug: shared LSTM output — matches /shared/LSTM_output_0 [totalFrames, 2, 1, 256]
    await this.captureDebug('/shared/LSTM_output_0', sharedLstmOut, [totalFrames, 2, 1, 256]);

    // Transpose shared LSTM output to channels-first: [totalFrames, 512] → [512, totalFrames]
    const sharedTransposed = this.createEmptyBuffer(512 * totalFrames, 'shared_transposed');
    this.dispatchTranspose(sharedLstmOut, sharedTransposed, totalFrames, 512);
    predIntermediates.push(sharedLstmOut);

    // ── N predictor: 3 AdaIN ResNet blocks on shared LSTM output [512, totalFrames] ──
    this.beginBatch();
    // N.1 pool doubles temporal resolution: totalFrames → 2*totalFrames (e.g. 149→298)
    // N_proj outputs at doubled resolution, then N_conv stride-2 downsamples back
    const baseFrames = totalFrames; // Save original frame count for decoder
    let nFeatures = sharedTransposed; // [512, baseFrames]
    let nChannels = 512;
    let nLength = baseFrames;

    for (let bi = 0; bi < 3; bi++) {
      const prefix = `kmodel.predictor.N.${bi}`;
      const poolConfig = bi === 1 ? { weightName: `kmodel.predictor.N.1.pool.weight`, channels: 512 } : undefined;
      const result = await this.runAdaINResNetBlock(
        nFeatures, stylePred, nChannels, nLength,
        prefix, bi === 1, // block 1 has conv1x1 residual and channel reduction
        bi === 0 ? 512 : 256, // output channels
        poolConfig,
      );
      predIntermediates.push(nFeatures);
      nFeatures = result.output;
      nChannels = result.outChannels;
      nLength = result.outLength;

      // Debug: N predictor block output
      await this.captureDebug(`/N.${bi}/block_output`, nFeatures, [1, nChannels, nLength]);
    }
    console.log(`[KittenTTS] N predictor output: [${nChannels}, ${nLength}]`);

    // N_proj: conv(256→1, k=1) → [1, nLength] (at doubled resolution e.g. 298)
    const nProjWeight = this.requireWeight('kmodel.predictor.N_proj.weight_quantized');
    const nProjBias = this.requireWeight('kmodel.predictor.N_proj.bias');
    const nProjOut = this.createEmptyBuffer(nLength, 'n_proj');
    this.dispatchConv1d(nFeatures, nProjWeight.buffer, nProjBias.buffer, nProjOut, nChannels, 1, 1, nLength, nLength, 0, 1, 1, true);
    predIntermediates.push(nFeatures);

    this.endBatch();
    await this.endStage('Duration + N predictor');

    // ── F0 Prediction: 3 AdaIN ResNet blocks on shared LSTM output [512, baseFrames] ──
    this.startStage();
    this.beginBatch();
    onProgress?.('6/8 F0 predictor'); console.log('[KittenTTS] Running F0 predictor...');
    let f0Features = sharedTransposed; // [512, baseFrames] — same input as N predictor
    let f0Channels = 512;
    let f0Length = baseFrames;

    // F0 predictor: 3 AdaIN ResNet blocks (same structure as N but separate weights)
    // Note: sharedTransposed is already owned by predIntermediates, don't push it again
    const f0Intermediates: GPUBuffer[] = [];
    for (let bi = 0; bi < 3; bi++) {
      const prefix = `kmodel.predictor.F0.${bi}`;
      const poolConfig = bi === 1 ? { weightName: `kmodel.predictor.F0.1.pool.weight`, channels: 512 } : undefined;
      const result = await this.runAdaINResNetBlock(
        f0Features, stylePred, f0Channels, f0Length,
        prefix, bi === 1, // block 1 has conv1x1 + channel reduction
        bi === 0 ? 512 : 256,
        poolConfig,
      );
      // Don't push sharedTransposed (bi=0) — it's already in predIntermediates
      if (bi > 0) f0Intermediates.push(f0Features);
      f0Features = result.output;
      f0Channels = result.outChannels;
      f0Length = result.outLength;

      // Debug: F0 predictor block output
      await this.captureDebug(`/F0.${bi}/block_output`, f0Features, [1, f0Channels, f0Length]);
    }
    console.log(`[KittenTTS] F0 predictor output: [${f0Channels}, ${f0Length}]`);

    // F0_proj: conv(256→1, k=1) → [1, f0Length] (at doubled resolution e.g. 298)
    const f0ProjWeight = this.requireWeight('kmodel.predictor.F0_proj.weight_quantized');
    const f0ProjBias = this.requireWeight('kmodel.predictor.F0_proj.bias');
    const f0ProjOut = this.createEmptyBuffer(f0Length, 'f0_proj');
    this.dispatchConv1d(f0Features, f0ProjWeight.buffer, f0ProjBias.buffer, f0ProjOut, f0Channels, 1, 1, f0Length, f0Length, 0, 1, 1, true);
    f0Intermediates.push(f0Features);

    this.endBatch();
    await this.endStage('F0 predictor');

    // ── 6. Decoder ────────────────────────────────────────────────────────
    this.startStage();
    this.beginBatch(); // Batch all decoder dispatches
    onProgress?.('7/8 Decoder'); console.log('[KittenTTS] Running decoder...');

    // F0_conv: conv1d(1→1, k=3, stride=2, pad=1) — downsamples f0Length→baseFrames
    // CRITICAL: stride=2 downsampling from doubled predictor resolution back to base
    const f0ConvWeight = this.requireWeight('kmodel.decoder.F0_conv.weight');
    const f0ConvBias = this.requireWeight('kmodel.decoder.F0_conv.bias');
    const f0Conv = this.createEmptyBuffer(baseFrames, 'f0_conv');
    this.dispatchConv1d(f0ProjOut, f0ConvWeight.buffer, f0ConvBias.buffer, f0Conv, 1, 1, 3, f0Length, baseFrames, 1, 2, 1, true);
    // f0ProjOut kept alive for sin generator noise injection (destroyed after HiFi-GAN noise section)
    // f0Intermediates.push(f0ProjOut);

    // N_conv: conv1d(1→1, k=3, stride=2, pad=1) — downsamples nLength→baseFrames
    const nConvWeight = this.requireWeight('kmodel.decoder.N_conv.weight');
    const nConvBias = this.requireWeight('kmodel.decoder.N_conv.bias');
    const nConv = this.createEmptyBuffer(baseFrames, 'n_conv');
    this.dispatchConv1d(nProjOut, nConvWeight.buffer, nConvBias.buffer, nConv, 1, 1, 3, nLength, baseFrames, 1, 2, 1, true);

    // Decoder input: concat(expanded_text_features[512], F0_conv[1], N_conv[1]) = [514, baseFrames]
    // CRITICAL: uses expanded text features (MatMul_1 in ONNX), NOT shared LSTM output
    const concat512f0 = this.createEmptyBuffer(513 * baseFrames, 'dec_concat_512f0');
    this.dispatchConcatChannels(expandedTextFeatures, f0Conv, concat512f0, 512, 1, baseFrames);
    const decoderInput = this.createEmptyBuffer(514 * baseFrames, 'decoder_input');
    this.dispatchConcatChannels(concat512f0, nConv, decoderInput, 513, 1, baseFrames);
    concat512f0.destroy();
    await this.captureDebug('/decoder/Concat_output_0', decoderInput, [1, 514, baseFrames]);

    // Cleanup predictor intermediates (no sync needed — GPU handles ordering)
    for (const buf of predIntermediates) buf.destroy();
    for (const buf of f0Intermediates) buf.destroy();
    nProjOut.destroy();
    bertOutput.destroy();

    // ── Decoder encode block ──
    let decFrames = baseFrames; // Track decoder frame count (doubles at decode.3)
    const encodeOut = await this.runDecoderBlock(
      decoderInput, styleDec, 514, 1024, decFrames,
      'kmodel.decoder.encode', true,
    );
    // Debug: decoder encode output (matches ONNX /decoder/encode/Div_output_0)
    await this.captureDebug('/decoder/encode/Div_output_0', encodeOut, [1, 1024, decFrames]);
    decoderInput.destroy();

    // ── Decoder decode blocks ──
    // asr_res: conv1x1(512→64) applied to expanded text features [512, baseFrames]
    const asrResWeight = this.requireWeight('kmodel.decoder.asr_res.0.weight_quantized');
    const asrResBias = this.requireWeight('kmodel.decoder.asr_res.0.bias');
    let asrResOut: GPUBuffer | undefined = this.createEmptyBuffer(64 * baseFrames, 'asr_res');
    this.dispatchConv1d(expandedTextFeatures, asrResWeight.buffer, asrResBias.buffer, asrResOut, 512, 64, 1, baseFrames, baseFrames, 0, 1, 1, true);
    expandedTextFeatures.destroy();

    // Build 1090-channel input for decode blocks (all on GPU)
    let decodeInput2 = this.buildDecodeInput(encodeOut, f0Conv, nConv, asrResOut, 1024, decFrames);
    encodeOut.destroy();

    let decodeOut: GPUBuffer = decodeInput2; // Will be reassigned in loop

    for (let di = 0; di < 4; di++) {
      const prefix = `kmodel.decoder.decode.${di}`;
      const outCh = di < 3 ? 1024 : 512;
      const inCh = 1090; // All decode blocks take 1090

      if (di === 3) {
        // decode.3 has pool: depthwise ConvTranspose inside block, doubles temporal resolution
        decodeOut = await this.runDecoderBlock(
          decodeInput2, styleDec, inCh, outCh, decFrames,
          prefix, true,
          { weightName: 'kmodel.decoder.decode.3.pool.weight', channels: inCh },
        );
        decodeInput2.destroy();
        decFrames = decFrames * 2; // Pool doubles the length
      } else {
        decodeOut = await this.runDecoderBlock(
          decodeInput2, styleDec, inCh, outCh, decFrames,
          prefix, true,
        );
        decodeInput2.destroy();
      }

      // Debug: decoder block output (matches ONNX /decoder/decode.{i}/Div_output_0)
      await this.captureDebug(`/decoder/decode.${di}/Div_output_0`, decodeOut, [1, outCh, decFrames]);

      if (di < 3) {
        // Build next decode input: concat decode output[outCh] + asr_res[64] + F0/N[2]
        decodeInput2 = this.buildDecodeInput(decodeOut, f0Conv, nConv, asrResOut ?? null, outCh, decFrames);
        decodeOut.destroy();
      }
    }

    // Cleanup F0/N conv buffers now that decode loop is done
    f0Conv.destroy();
    nConv.destroy();

    // Assign totalFrames to the new doubled value for the generator
    totalFrames = decFrames;
    // decodeOut is the final decoder output [512, totalFrames] (after pool doubling)
    console.log(`[KittenTTS] Decoder output: [512, ${totalFrames}]`);

    await this.endStage('Decoder (5 blocks)');

    // ── 7. HiFi-GAN Generator ─────────────────────────────────────────────
    this.startStage();
    onProgress?.('8/8 HiFi-GAN'); console.log('[KittenTTS] Running HiFi-GAN...');
    this.beginBatch(); // Batch 1: LeakyReLU + ups.0

    let genFeatures = decodeOut!; // [512, totalFrames]
    let genChannels = 512;
    let genLength = totalFrames;

    // ── LeakyReLU(0.1) before ups.0 (ONNX: LeakyRelu_0 on decoder output) ──
    const preUps0Leaky = this.createEmptyBuffer(genChannels * genLength, 'pre_ups0_leaky');
    this.dispatchLeakyRelu(genFeatures, preUps0Leaky, genChannels * genLength, 0.1);
    genFeatures.destroy();
    genFeatures = preUps0Leaky;

    // ── ups.0: ConvTranspose1d(512→256, k=20, stride=10, pad=5) ──
    const ups0Weight = this.requireWeight('kmodel.decoder.generator.ups.0.weight');
    const ups0Bias = this.requireWeight('kmodel.decoder.generator.ups.0.bias');
    // ONNX: stride=10, kernel=20, pads=[5,5]
    // output_length = (input_length - 1) * stride + kernel - 2*pad = (L-1)*10 + 20 - 10 = L*10
    const ups0Length = genLength * 10;
    const ups0Out = this.createEmptyBuffer(256 * ups0Length, 'ups0');
    this.dispatchConvTranspose1d(genFeatures, ups0Weight.buffer, ups0Bias.buffer, ups0Out, 512, 256, 20, genLength, ups0Length, 10, 5, true);
    genFeatures.destroy();
    genFeatures = ups0Out;
    genChannels = 256;
    genLength = ups0Length;
    console.log(`[KittenTTS] ups.0 output: [${genChannels}, ${genLength}]`);

    this.endBatch(); // Flush ups.0 before CPU sin generator

    // Debug: ups.0 output
    await this.captureDebug('/decoder/generator/ups.0/ConvTranspose_output_0', genFeatures, [1, genChannels, genLength]);

    // ── Noise injection: source excitation from F0 harmonic generator ──
    // Generates 9 harmonics from F0, collapses via linear+tanh, forward STFT → 22 channels.
    // Computed on CPU (cumulative sum is inherently sequential).
    // Weights are cached after first call, so only 1 readBuffer (F0 data) needed.
    const stftLen = genLength * 6 + 1; // = ups1Length + 1 = totalFrames * 60 + 1
    const noiseInput = await this.generateSourceExcitation(f0ProjOut, f0Length, stftLen);
    f0ProjOut.destroy();

    this.beginBatch(); // Batch 2: noise_convs + resblocks + ups.1 + conv_post

    // ── noise_convs.0: Conv1d(22→256, k=12, stride=6, pad=3) → [256, ups0Length] ──
    const nc0Weight = this.requireWeight('kmodel.decoder.generator.noise_convs.0.weight_quantized');
    const nc0Bias = this.requireWeight('kmodel.decoder.generator.noise_convs.0.bias');
    const nc0Len = Math.floor((stftLen + 2 * 3 - 12) / 6) + 1; // should equal genLength (ups0Length)
    const nc0Out = this.createEmptyBuffer(256 * nc0Len, 'noise_convs0');
    this.dispatchConv1d(noiseInput, nc0Weight.buffer, nc0Bias.buffer, nc0Out, 22, 256, 12, stftLen, nc0Len, 3, 6, 1, true);

    // ── noise_res.0: 3-iteration AdaIN ResBlock (same as HiFi-GAN resblocks) ──
    const nr0Out = await this.runHiFiGANResBlock(nc0Out, styleDec, 256, nc0Len, 'kmodel.decoder.generator.noise_res.0');
    nc0Out.destroy();

    // ── Add_3: noise_res.0 output + ups.0 output (noise added BEFORE resblocks) ──
    const noisyUps0 = this.createEmptyBuffer(genChannels * genLength, 'noisy_ups0');
    this.dispatchAdd(genFeatures, nr0Out, noisyUps0, genChannels * genLength);
    genFeatures.destroy();
    nr0Out.destroy();
    genFeatures = noisyUps0;

    // ── resblocks.0 + resblocks.1: parallel residual blocks, output averaged ──
    const resblock0 = await this.runHiFiGANResBlock(genFeatures, styleDec, genChannels, genLength, 'kmodel.decoder.generator.resblocks.0');
    const resblock1 = await this.runHiFiGANResBlock(genFeatures, styleDec, genChannels, genLength, 'kmodel.decoder.generator.resblocks.1');

    // Average: (resblock0 + resblock1) / 2
    const resSum0 = this.createEmptyBuffer(genChannels * genLength, 'res_sum0');
    this.dispatchAdd(resblock0, resblock1, resSum0, genChannels * genLength);
    resblock0.destroy();
    resblock1.destroy();
    const resAvg0 = this.createEmptyBuffer(genChannels * genLength, 'res_avg0');
    this.dispatchScale(resSum0, resAvg0, genChannels * genLength, 0.5);
    resSum0.destroy();
    genFeatures.destroy();
    genFeatures = resAvg0;

    // ── LeakyReLU(0.1) before ups.1 (ONNX: LeakyRelu_1 after resblock average) ──
    const preUps1Leaky = this.createEmptyBuffer(genChannels * genLength, 'pre_ups1_leaky');
    this.dispatchLeakyRelu(genFeatures, preUps1Leaky, genChannels * genLength, 0.1);
    genFeatures.destroy();
    genFeatures = preUps1Leaky;

    // ── ups.1: ConvTranspose1d(256→128, k=12, stride=6, pad=3) ──
    const ups1Weight = this.requireWeight('kmodel.decoder.generator.ups.1.weight');
    const ups1Bias = this.requireWeight('kmodel.decoder.generator.ups.1.bias');
    // ONNX: stride=6, kernel=12, pads=[3,3]
    // output_length = (L-1)*6 + 12 - 6 = L*6
    const ups1Length = genLength * 6;
    const ups1Out = this.createEmptyBuffer(128 * ups1Length, 'ups1');
    this.dispatchConvTranspose1d(genFeatures, ups1Weight.buffer, ups1Bias.buffer, ups1Out, 256, 128, 12, genLength, ups1Length, 6, 3, true);
    genFeatures.destroy();
    genFeatures = ups1Out;
    genChannels = 128;
    genLength = ups1Length;
    console.log(`[KittenTTS] ups.1 output: [${genChannels}, ${genLength}]`);

    // Debug: ups.1 output
    await this.captureDebug('/decoder/generator/ups.1/ConvTranspose_output_0', genFeatures, [1, genChannels, genLength]);

    // ── Reflection pad: +1 sample at start of time dimension ──
    // ONNX: /decoder/generator/reflection_pad/Pad with pads=[0,0,1,0,0,0] mode=reflect
    // Applied AFTER ups.1 and BEFORE resblocks.2+3
    {
      const paddedLength = genLength + 1;
      const paddedOut = this.createEmptyBuffer(genChannels * paddedLength, 'gen_reflected');
      this.dispatchReflectionPad1d(genFeatures, paddedOut, genChannels, genLength, 1, 0);
      genFeatures.destroy();
      genFeatures = paddedOut;
      genLength = paddedLength;
    }
    console.log(`[KittenTTS] After reflection pad: [${genChannels}, ${genLength}]`);

    // ── noise_convs.1: Conv1d(22→128, k=1, stride=1, pad=0) → [128, stftLen] ──
    const nc1Weight = this.requireWeight('kmodel.decoder.generator.noise_convs.1.weight_quantized');
    const nc1Bias = this.requireWeight('kmodel.decoder.generator.noise_convs.1.bias');
    const nc1Out = this.createEmptyBuffer(128 * stftLen, 'noise_convs1');
    this.dispatchConv1d(noiseInput, nc1Weight.buffer, nc1Bias.buffer, nc1Out, 22, 128, 1, stftLen, stftLen, 0, 1, 1, true);
    noiseInput.destroy(); // Done with noise source

    // ── noise_res.1: 3-iteration AdaIN ResBlock ──
    const nr1Out = await this.runHiFiGANResBlock(nc1Out, styleDec, 128, stftLen, 'kmodel.decoder.generator.noise_res.1');
    nc1Out.destroy();

    // ── Add_5: noise_res.1 output + reflection_pad output (noise added BEFORE resblocks) ──
    // genLength should equal stftLen at this point (both are ups1Length + 1)
    const noisyPad = this.createEmptyBuffer(genChannels * genLength, 'noisy_pad');
    this.dispatchAdd(genFeatures, nr1Out, noisyPad, genChannels * genLength);
    genFeatures.destroy();
    nr1Out.destroy();
    genFeatures = noisyPad;

    // ── resblocks.2 + resblocks.3: parallel residual blocks, output averaged ──
    const resblock2 = await this.runHiFiGANResBlock(genFeatures, styleDec, genChannels, genLength, 'kmodel.decoder.generator.resblocks.2');
    const resblock3 = await this.runHiFiGANResBlock(genFeatures, styleDec, genChannels, genLength, 'kmodel.decoder.generator.resblocks.3');

    const resSum1 = this.createEmptyBuffer(genChannels * genLength, 'res_sum1');
    this.dispatchAdd(resblock2, resblock3, resSum1, genChannels * genLength);
    resblock2.destroy();
    resblock3.destroy();
    const resAvg1 = this.createEmptyBuffer(genChannels * genLength, 'res_avg1');
    this.dispatchScale(resSum1, resAvg1, genChannels * genLength, 0.5);
    resSum1.destroy();
    genFeatures.destroy();
    genFeatures = resAvg1;

    // ── LeakyReLU + conv_post: conv(128→22, k=7) ──
    const postLeaky = this.createEmptyBuffer(genChannels * genLength, 'post_leaky');
    this.dispatchLeakyRelu(genFeatures, postLeaky, genChannels * genLength, 0.01);
    genFeatures.destroy();

    const convPostWeight = this.requireWeight('kmodel.decoder.generator.conv_post.weight_quantized');
    const convPostBias = this.requireWeight('kmodel.decoder.generator.conv_post.bias');
    const convPostOut = this.createEmptyBuffer(22 * genLength, 'conv_post');
    this.dispatchConv1d(postLeaky, convPostWeight.buffer, convPostBias.buffer, convPostOut, 128, 22, 7, genLength, genLength, 3, 1, 1, true);
    postLeaky.destroy();

    this.endBatch(); // Flush batch 2: noise + resblocks + ups.1 + conv_post

    // Debug: conv_post output
    await this.captureDebug('/decoder/generator/conv_post/Conv_output_0', convPostOut, [1, 22, genLength]);

    await this.endStage('HiFi-GAN generator');

    this.startStage();
    // ── iSTFT synthesis on GPU: conv_post [22, genLength] → waveform ──
    // Fused gather-based ConvTranspose: each thread computes one output sample
    const stftBins = 11;
    const stftKernel = 20;
    const stftStride = 5;
    const waveformLength = (genLength - 1) * stftStride + stftKernel;

    const stftWeightReal = this.requireWeight('kmodel.decoder.generator.stft.weight_backward_real');
    const stftWeightImag = this.requireWeight('kmodel.decoder.generator.stft.weight_backward_imag');

    const waveformGpu = this.createEmptyBuffer(waveformLength, 'waveform_gpu');
    this.dispatchISTFT(convPostOut, stftWeightReal.buffer, stftWeightImag.buffer, waveformGpu,
      genLength, waveformLength, stftBins, stftKernel, stftStride);
    convPostOut.destroy();

    // Read back and trim: ONNX Slice_3 does starts=[10], ends=[-10] on axis 2
    const waveformData = await this.readBuffer(waveformGpu, waveformLength);
    waveformGpu.destroy();
    const trimStart = 10;
    const trimEnd = waveformLength - 10;
    const finalWaveform = waveformData.slice(trimStart, trimEnd);

    await this.endStage('iSTFT synthesis (GPU)');
    this.printTimings();

    console.log(`[KittenTTS] Waveform: ${finalWaveform.length} samples (${(finalWaveform.length / 24000).toFixed(2)}s)`);

    // Debug: final waveform
    if (this.debugCapture) {
      this.debugActivations.set('waveform', {
        data: finalWaveform,
        shape: [finalWaveform.length],
      });
    }

    // Cleanup — destroy ALL intermediate GPU buffers to prevent memory leaks
    // Note: many buffers are already pushed to predIntermediates/f0Intermediates
    // and destroyed earlier (line ~803-804). Only destroy ones NOT in those arrays.
    inputIdsBuf.destroy();
    styleBuf.destroy();
    speedBuf.destroy();
    tokenTypeIds.destroy();      // leaked: never in predIntermediates
    tokenTypeEmbOut.destroy();   // leaked: never in predIntermediates
    wordPlusTokenType.destroy(); // leaked: never in predIntermediates
    posIds.destroy();
    posEmbOut.destroy();
    embeddingOut.destroy();
    bertEmbedding.destroy();
    stylePred.destroy();
    styleDec.destroy();
    asrResOut?.destroy();
    // bertProjOut — already in predIntermediates[0], destroyed at line ~803
    sharedTransposed.destroy();  // leaked: never in predIntermediates
    // Final flush of any remaining uniform buffers
    this.flushUniformBuffers();

    return { waveform: finalWaveform, duration: durations };
  }

  /** CPU LSTM implementation for debugging/verification. */
  private cpuLSTM(
    input: Float32Array, W: Float32Array, R: Float32Array, bias: Float32Array,
    seqLen: number, inputSize: number, hiddenSize: number, numDirections: number
  ): Float32Array {
    const H = hiddenSize;
    const H4 = H * 4;
    const IS = inputSize;
    const output = new Float32Array(seqLen * numDirections * H);

    for (let dir = 0; dir < numDirections; dir++) {
      const h = new Float32Array(H); // hidden state
      const c = new Float32Array(H); // cell state

      // Bias: [Wb_i, Wb_o, Wb_f, Wb_c, Rb_i, Rb_o, Rb_f, Rb_c] per direction
      const biasBase = dir * 8 * H;

      // Weight bases
      const wBase = dir * IS * H4;
      const rBase = dir * H * H4;

      for (let step = 0; step < seqLen; step++) {
        const t = dir === 0 ? step : seqLen - 1 - step;

        // Save h_prev BEFORE updating (all hidden units must use same h_prev)
        const hPrev = new Float32Array(h);

        // Compute all gates for all hidden units
        const gates_i = new Float32Array(H);
        const gates_o = new Float32Array(H);
        const gates_f = new Float32Array(H);
        const gates_c = new Float32Array(H);

        for (let hi = 0; hi < H; hi++) {
          gates_i[hi] = bias[biasBase + hi] + bias[biasBase + 4 * H + hi];
          gates_o[hi] = bias[biasBase + H + hi] + bias[biasBase + 5 * H + hi];
          gates_f[hi] = bias[biasBase + 2 * H + hi] + bias[biasBase + 6 * H + hi];
          gates_c[hi] = bias[biasBase + 3 * H + hi] + bias[biasBase + 7 * H + hi];
        }

        // Input contribution: W[dir, j, gate*H+h] — layout [IS, 4H]
        for (let j = 0; j < IS; j++) {
          const xVal = input[t * IS + j];
          const wOff = wBase + j * H4;
          for (let hi = 0; hi < H; hi++) {
            gates_i[hi] += xVal * W[wOff + hi];
            gates_o[hi] += xVal * W[wOff + H + hi];
            gates_f[hi] += xVal * W[wOff + 2 * H + hi];
            gates_c[hi] += xVal * W[wOff + 3 * H + hi];
          }
        }

        // Recurrence: R[dir, j, gate*H+h] — layout [H, 4H]
        for (let j = 0; j < H; j++) {
          const hp = hPrev[j];
          const rOff = rBase + j * H4;
          for (let hi = 0; hi < H; hi++) {
            gates_i[hi] += hp * R[rOff + hi];
            gates_o[hi] += hp * R[rOff + H + hi];
            gates_f[hi] += hp * R[rOff + 2 * H + hi];
            gates_c[hi] += hp * R[rOff + 3 * H + hi];
          }
        }

        // Apply gate activations and update states
        for (let hi = 0; hi < H; hi++) {
          const iGate = 1 / (1 + Math.exp(-gates_i[hi]));
          const oGate = 1 / (1 + Math.exp(-gates_o[hi]));
          const fGate = 1 / (1 + Math.exp(-gates_f[hi]));
          const cGate = Math.tanh(gates_c[hi]);

          c[hi] = fGate * c[hi] + iGate * cGate;
          h[hi] = oGate * Math.tanh(c[hi]);
        }

        // Debug: log first timestep, first direction
        if (step === 0 && dir === 0 && this.debugCapture) {
          console.log(`[cpuLSTM] t=0,d=0: gates_i[0:3]=[${gates_i.subarray(0,3).join(',')}]`);
          console.log(`[cpuLSTM] t=0,d=0: gates_o[0:3]=[${gates_o.subarray(0,3).join(',')}]`);
          console.log(`[cpuLSTM] t=0,d=0: gates_f[0:3]=[${gates_f.subarray(0,3).join(',')}]`);
          console.log(`[cpuLSTM] t=0,d=0: gates_c[0:3]=[${gates_c.subarray(0,3).join(',')}]`);
          console.log(`[cpuLSTM] t=0,d=0: h[0:5]=[${h.subarray(0,5).join(',')}]`);
          console.log(`[cpuLSTM] t=0,d=0: c[0:5]=[${c.subarray(0,5).join(',')}]`);
          console.log(`[cpuLSTM] t=0,d=0: input[0:5]=[${input.subarray(0,5).join(',')}]`);
        }

        // Write output: [t, dir, h]
        const outBase = t * numDirections * H + dir * H;
        for (let hi = 0; hi < H; hi++) {
          output[outBase + hi] = h[hi];
        }
      }
    }

    return output;
  }

  // ── GPU Helper Methods ─────────────────────────────────────────────────────

  private createBuffer(data: ArrayBufferView, label: string, usage: number): GPUBuffer {
    const buffer = this.device.createBuffer({
      size: data.byteLength,
      usage: usage | GPUBufferUsage.COPY_DST,
      label,
    });
    this.device.queue.writeBuffer(buffer, 0, data as unknown as ArrayBuffer);
    return buffer;
  }

  private createEmptyBuffer(elements: number, label: string): GPUBuffer {
    return this.device.createBuffer({
      size: elements * 4, // float32
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label,
    });
  }

  private createUniformBuffer(data: ArrayBufferView, label: string): GPUBuffer {
    const size = Math.ceil(data.byteLength / 16) * 16; // Align to 16 bytes
    const alignedData = new Uint8Array(Math.max(size, 16));
    alignedData.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    const buffer = this.device.createBuffer({
      size: alignedData.byteLength,
      usage: GPUBufferUsage.UNIFORM,
      label,
      mappedAtCreation: true,
    });
    new Uint8Array(buffer.getMappedRange()).set(alignedData);
    buffer.unmap();
    // Track for cleanup after submit
    this.pendingUniformBuffers.push(buffer);
    return buffer;
  }

  /** Destroy all pending uniform buffers (call after command submit). */
  private flushUniformBuffers(): void {
    for (const buf of this.pendingUniformBuffers) {
      buf.destroy();
    }
    this.pendingUniformBuffers = [];
  }

  /** Get a weight tensor, throwing a descriptive error if missing. */
  private requireWeight(name: string): GpuTensor {
    const w = this.weights.get(name);
    if (!w) throw new Error(`[KittenTTS] Missing weight: ${name}`);
    return w;
  }

  /** Start batching — flush any pending encoder + uniform buffers. */
  private beginBatch(): void {
    this.flushBatchEncoder();
  }

  /** End batch — flush encoder + uniform buffers. */
  private endBatch(): void {
    this.flushBatchEncoder();
  }

  /** Flush batch (alias). */
  private flushBatch(): void {
    this.flushBatchEncoder();
  }

  /** Submit batch — flush encoder + uniform buffers. */
  private submitBatch(): void {
    this.flushBatchEncoder();
  }

  /**
   * Execute a dispatch on a pipeline. Each dispatch gets its own encoder
   * and is submitted immediately (ensures correct memory barriers and
   * allows buffer destruction right after dispatch).
   */
  private dispatchSingle(
    pipelineName: string,
    bindGroup: GPUBindGroup,
    workgroupsX: number,
    workgroupsY = 1,
    workgroupsZ = 1,
  ): void {
    const { pipeline } = this.pipelines.get(pipelineName)!;

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** Flush uniform buffers (dispatches are submitted immediately now). */
  private flushBatchEncoder(): void {
    this.flushUniformBuffers();
  }

  /** Read buffer contents back to CPU. */
  private async readBuffer(buffer: GPUBuffer, size: number): Promise<Float32Array> {
    // Flush any batched dispatches before reading back
    this.flushBatchEncoder();

    const staging = this.device.createBuffer({
      size: size * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, size * 4);
    this.device.queue.submit([encoder.finish()]);

    // Wait for the GPU copy to complete before mapping — Safari crashes without this
    await this.device.queue.onSubmittedWorkDone();

    await staging.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return result;
  }

  // ── Shader Dispatch Methods ────────────────────────────────────────────────

  private dispatchEmbedding(
    embeddingWeight: GPUBuffer,
    inputIds: GPUBuffer,
    output: GPUBuffer,
    seqLen: number,
    embedDim: number,
    vocabSize: number
  ): void {
    const { bindGroupLayout } = this.pipelines.get('embedding')!;
    const params = this.createUniformBuffer(
      new Uint32Array([seqLen, embedDim, vocabSize]),
      'embedding_params'
    );

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: embeddingWeight } },
        { binding: 1, resource: { buffer: inputIds } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: params } },
      ],
    });

    this.dispatchSingle('embedding', bindGroup, Math.ceil((seqLen * embedDim) / 256));
  }

  private dispatchAdd(a: GPUBuffer, b: GPUBuffer, output: GPUBuffer, size: number): void {
    const { bindGroupLayout } = this.pipelines.get('add')!;
    const params = this.createUniformBuffer(new Uint32Array([size]), 'add_params');

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: a } },
        { binding: 1, resource: { buffer: b } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: params } },
      ],
    });

    this.dispatchSingle('add', bindGroup, Math.ceil(size / 256));
  }

  // ── BERT/ALBERT Encoder ──────────────────────────────────────────────────

  /**
   * Run the BERT/ALBERT encoder.
   *
   * Pipeline:
   * 1. Embedding sum (word + position) is already computed → [seqLen, 128]
   * 2. LayerNorm on embeddings (128-dim)
   * 3. Linear projection 128 → 768 (embedding_hidden_mapping_in)
   * 4. 12 iterations of shared ALBERT layer:
   *    a. Self-attention (Q/K/V projections, scaled dot-product, output projection)
   *    b. Residual + LayerNorm (attention)
   *    c. FFN: Linear 768→2048 (GELU) → Linear 2048→768
   *    d. Residual + LayerNorm (full layer)
   * 5. Return [1, seqLen, 768]
   */
  private runBertEncoder(inputIdsBuf: GPUBuffer, embeddingSum: GPUBuffer, seqLen: number): GPUBuffer {
    const embedDim = 128;
    const hiddenSize = 768;
    const numHeads = 12;
    const headDim = 64;
    const ffnDim = 2048;
    const numLayers = 12;
    const eps = 1e-12; // BERT default epsilon

    // Single command encoder for the entire BERT forward pass.
    const cmdEncoder = this.device.createCommandEncoder({ label: 'bert_encoder' });
    const pass = cmdEncoder.beginComputePass({ label: 'bert_pass' });

    // ── Step 1: LayerNorm on embeddings ──
    const embLnWeight = this.requireWeight('kmodel.bert.embeddings.LayerNorm.weight');
    const embLnBias = this.requireWeight('kmodel.bert.embeddings.LayerNorm.bias');
    const normedEmb = this.createEmptyBuffer(seqLen * embedDim, 'bert_emb_ln');
    this.dispatchLayerNorm(embeddingSum, embLnWeight.buffer, embLnBias.buffer, normedEmb, seqLen, embedDim, eps, pass);

    // ── Step 2: Project 128 → 768 ──
    const projWeight = this.requireWeight('onnx::MatMul_5883_quantized');
    const projBias = this.requireWeight('kmodel.bert.encoder.embedding_hidden_mapping_in.bias');
    let hidden = this.createEmptyBuffer(seqLen * hiddenSize, 'bert_projected');
    this.dispatchMatmul(normedEmb, projWeight.buffer, projBias.buffer, hidden, seqLen, embedDim, hiddenSize, true, pass);

    // ── Step 3: Shared ALBERT layers (12 iterations) ──
    const prefix = 'kmodel.bert.encoder.albert_layer_groups.0.albert_layers.0';

    // Pre-fetch all shared weights (same for all 12 iterations)
    const qWeight = this.requireWeight('onnx::MatMul_5884_quantized');
    const kWeight = this.requireWeight('onnx::MatMul_5887_quantized');
    const vWeight = this.requireWeight('onnx::MatMul_5890_quantized');
    const qBias = this.requireWeight(`${prefix}.attention.query.bias`);
    const kBias = this.requireWeight(`${prefix}.attention.key.bias`);
    const vBias = this.requireWeight(`${prefix}.attention.value.bias`);
    const attnOutWeight = this.requireWeight('onnx::MatMul_5894_quantized');
    const attnOutBias = this.requireWeight(`${prefix}.attention.dense.bias`);
    const attnLnWeight = this.requireWeight(`${prefix}.attention.LayerNorm.weight`);
    const attnLnBias = this.requireWeight(`${prefix}.attention.LayerNorm.bias`);
    const ffnUpWeight = this.requireWeight('onnx::MatMul_5895_quantized');
    const ffnUpBias = this.requireWeight(`${prefix}.ffn.bias`);
    const ffnDownWeight = this.requireWeight('onnx::MatMul_5896_quantized');
    const ffnDownBias = this.requireWeight(`${prefix}.ffn_output.bias`);
    const fullLnWeight = this.requireWeight(`${prefix}.full_layer_layer_norm.weight`);
    const fullLnBias = this.requireWeight(`${prefix}.full_layer_layer_norm.bias`);

    const scale = 1.0 / Math.sqrt(headDim); // 1/sqrt(64) = 0.125

    // Track intermediate buffers for cleanup after submit
    const intermediates: GPUBuffer[] = [normedEmb];
    const debugLayerBufs = new Map<string, GPUBuffer>();

    for (let layer = 0; layer < numLayers; layer++) {
      // ── 3a. Self-Attention: Q/K/V projections ──
      const Q = this.createEmptyBuffer(seqLen * hiddenSize, `Q_${layer}`);
      const K = this.createEmptyBuffer(seqLen * hiddenSize, `K_${layer}`);
      const V = this.createEmptyBuffer(seqLen * hiddenSize, `V_${layer}`);

      this.dispatchMatmul(hidden, qWeight.buffer, qBias.buffer, Q, seqLen, hiddenSize, hiddenSize, true, pass);
      this.dispatchMatmul(hidden, kWeight.buffer, kBias.buffer, K, seqLen, hiddenSize, hiddenSize, true, pass);
      this.dispatchMatmul(hidden, vWeight.buffer, vBias.buffer, V, seqLen, hiddenSize, hiddenSize, true, pass);

      // Multi-head attention: [seqLen, 12heads, 64dim]
      const attnOut = this.createEmptyBuffer(seqLen * hiddenSize, `attn_out_${layer}`);
      this.dispatchMHA(Q, K, V, attnOut, seqLen, numHeads, headDim, scale, pass);
      if (this.debugCapture && layer === 0) {
        debugLayerBufs.set('Q_0', Q);
        debugLayerBufs.set('attn_out_0', attnOut);
        debugLayerBufs.set('ffn_up_weight', ffnUpWeight.buffer);
        debugLayerBufs.set('ffn_up_bias', ffnUpBias.buffer);
      }
      intermediates.push(Q, K, V);

      // Attention output projection
      const attnProjected = this.createEmptyBuffer(seqLen * hiddenSize, `attn_proj_${layer}`);
      this.dispatchMatmul(attnOut, attnOutWeight.buffer, attnOutBias.buffer, attnProjected, seqLen, hiddenSize, hiddenSize, true, pass);
      intermediates.push(attnOut);

      // ── 3b. Residual + LayerNorm ──
      const attnResidual = this.createEmptyBuffer(seqLen * hiddenSize, `attn_res_${layer}`);
      this.dispatchAddPass(hidden, attnProjected, attnResidual, seqLen * hiddenSize, pass);
      intermediates.push(attnProjected);

      const attnNormed = this.createEmptyBuffer(seqLen * hiddenSize, `attn_ln_${layer}`);
      this.dispatchLayerNorm(attnResidual, attnLnWeight.buffer, attnLnBias.buffer, attnNormed, seqLen, hiddenSize, eps, pass);
      if (this.debugCapture && layer === 0) {
        debugLayerBufs.set('attn_ln_0', attnNormed);
      }
      intermediates.push(attnResidual);

      // ── 3c. FFN: up (GELU) then down ──
      const ffnUp = this.createEmptyBuffer(seqLen * ffnDim, `ffn_up_${layer}`);
      this.dispatchMatmulGelu(attnNormed, ffnUpWeight.buffer, ffnUpBias.buffer, ffnUp, seqLen, hiddenSize, ffnDim, pass);

      const ffnDown = this.createEmptyBuffer(seqLen * hiddenSize, `ffn_down_${layer}`);
      this.dispatchMatmul(ffnUp, ffnDownWeight.buffer, ffnDownBias.buffer, ffnDown, seqLen, ffnDim, hiddenSize, true, pass);
      if (this.debugCapture && layer === 0) {
        debugLayerBufs.set('ffn_up_0', ffnUp);
        debugLayerBufs.set('ffn_down_0', ffnDown);
        debugLayerBufs.set('ffn_down_weight', ffnDownWeight.buffer);
        debugLayerBufs.set('ffn_down_bias', ffnDownBias.buffer);
      }
      intermediates.push(ffnUp);

      // ── 3d. Residual + LayerNorm ──
      const ffnResidual = this.createEmptyBuffer(seqLen * hiddenSize, `ffn_res_${layer}`);
      this.dispatchAddPass(attnNormed, ffnDown, ffnResidual, seqLen * hiddenSize, pass);
      intermediates.push(attnNormed, ffnDown);

      const nextHidden = this.createEmptyBuffer(seqLen * hiddenSize, `hidden_${layer + 1}`);
      this.dispatchLayerNorm(ffnResidual, fullLnWeight.buffer, fullLnBias.buffer, nextHidden, seqLen, hiddenSize, eps, pass);
      intermediates.push(ffnResidual);

      // Previous hidden is now an intermediate (except the final output)
      if (this.debugCapture && layer <= 1) {
        debugLayerBufs.set(`hidden_${layer + 1}`, nextHidden);
      }
      intermediates.push(hidden);
      hidden = nextHidden;
    }

    // Submit all dispatches as a single command buffer
    pass.end();
    this.device.queue.submit([cmdEncoder.finish()]);

    // Debug: save key intermediate buffers for inspection
    if (this.debugCapture) {
      this.debugBertBuffers = { normedEmb, projected: intermediates.find(b => b.label === 'bert_projected') || null, hidden, seqLen, embedDim, hiddenSize, layerBuffers: debugLayerBufs };
    }

    // Cleanup all intermediate and uniform buffers
    const keepLabels = this.debugCapture ? new Set([...debugLayerBufs.keys(), 'bert_projected', 'bert_emb_ln']) : null;
    for (const buf of intermediates) {
      if (keepLabels?.has(buf.label)) continue; // keep for debug
      buf.destroy();
    }
    this.flushUniformBuffers();

    return hidden; // [seqLen, 768]
  }

  // ── Text Encoder (CNN + LSTM) ──────────────────────────────────────────────

  /**
   * Run the text encoder.
   *
   * Pipeline:
   * 1. Embedding lookup: input_ids → [seqLen, 512]
   * 2. Transpose to channels-first: [512, seqLen]
   * 3. 3× Conv1d(512, 512, k=5, pad=2) + LayerNorm + LeakyReLU
   * 4. Transpose to [seqLen, 512] for LSTM input
   * 5. Bidirectional LSTM (hidden=256) → [seqLen, 2, 256]
   *
   * Returns buffer with shape [seqLen, 2, 256] (= [seqLen, 512] flattened)
   */
  private async runTextEncoder(inputIdsBuf: GPUBuffer, seqLen: number): Promise<GPUBuffer> {
    const channels = 512;
    const kernelSize = 5;
    const padding = 2;
    const hiddenSize = 256;
    const numDirections = 2;

    const intermediates: GPUBuffer[] = [];

    // ── Step 1: Text encoder embedding lookup ──
    const embWeight = this.requireWeight('kmodel.text_encoder.embedding.weight');
    const embOut = this.createEmptyBuffer(seqLen * channels, 'te_embedding');
    this.dispatchEmbedding(embWeight.buffer, inputIdsBuf, embOut, seqLen, channels, 178);

    // Debug: text encoder embedding — matches /text_encoder/embedding/Gather_output_0 [1, seqLen, 512]
    await this.captureDebug('/text_encoder/embedding/Gather_output_0', embOut, [1, seqLen, channels]);

    // ── Step 2: Transpose [seqLen, 512] → [512, seqLen] for Conv1d ──
    const transposed = this.createEmptyBuffer(seqLen * channels, 'te_transposed');
    this.dispatchTranspose(embOut, transposed, seqLen, channels);
    intermediates.push(embOut);

    // Debug: transposed embedding (conv input) — should match /text_encoder/Transpose_output_0 [1, 512, seqLen]
    await this.captureDebug('/text_encoder/Transpose_output_0', transposed, [1, channels, seqLen]);

    // ── Step 3: 3× Conv1d blocks ──
    let convInput = transposed;

    for (let i = 0; i < 3; i++) {
      const convWeight = this.requireWeight(`kmodel.text_encoder.cnn.${i}.0.weight_quantized`);
      const convBias = this.requireWeight(`kmodel.text_encoder.cnn.${i}.0.bias`);

      // Debug: capture the dequantized weight and bias for the first CNN block
      if (i === 0) {
        await this.captureDebug('debug/cnn0_weight', convWeight.buffer, [512, 512, 5]);
        await this.captureDebug('debug/cnn0_bias', convBias.buffer, [512]);
      }
      const lnGamma = this.requireWeight(`kmodel.text_encoder.cnn.${i}.1.gamma`);
      const lnBeta = this.requireWeight(`kmodel.text_encoder.cnn.${i}.1.beta`);

      // Conv1d: [512, seqLen] → [512, seqLen]
      const convOut = this.createEmptyBuffer(channels * seqLen, `te_conv${i}`);
      this.dispatchConv1d(
        convInput, convWeight.buffer, convBias.buffer, convOut,
        channels, channels, kernelSize, seqLen, seqLen, padding, 1, 1, true
      );
      intermediates.push(convInput);

      // Debug: text encoder CNN output — matches /text_encoder/cnn.{i}/cnn.{i}.0/Conv_output_0quant_scaled_output [1, 512, seqLen]
      await this.captureDebug(`/text_encoder/cnn.${i}/cnn.${i}.0/Conv_output_0quant_scaled_output`, convOut, [1, channels, seqLen]);

      // Transpose [512, seqLen] → [seqLen, 512] for LayerNorm
      const preNorm = this.createEmptyBuffer(seqLen * channels, `te_prenorm${i}`);
      this.dispatchTranspose(convOut, preNorm, channels, seqLen);
      intermediates.push(convOut);

      // LayerNorm over 512-dim features
      const normed = this.createEmptyBuffer(seqLen * channels, `te_ln${i}`);
      this.dispatchLayerNorm(preNorm, lnGamma.buffer, lnBeta.buffer, normed, seqLen, channels, 1e-5);
      intermediates.push(preNorm);

      // Transpose [seqLen, 512] → [512, seqLen] back to channels-first
      const postNorm = this.createEmptyBuffer(channels * seqLen, `te_postnorm${i}`);
      this.dispatchTranspose(normed, postNorm, seqLen, channels);
      intermediates.push(normed);

      // LeakyReLU
      const activated = this.createEmptyBuffer(channels * seqLen, `te_act${i}`);
      this.dispatchLeakyRelu(postNorm, activated, channels * seqLen, 0.2);
      intermediates.push(postNorm);

      convInput = activated;
    }

    // ── Step 4: Transpose [512, seqLen] → [seqLen, 512] for LSTM input ──
    const lstmInput = this.createEmptyBuffer(seqLen * channels, 'te_lstm_in');
    this.dispatchTranspose(convInput, lstmInput, channels, seqLen);
    intermediates.push(convInput);

    // ── Step 5: Bidirectional LSTM ──
    const lstmW = this.requireWeight('onnx::LSTM_5874_quantized');   // [2, input_size, 4*hidden] = [2, 512, 1024]
    const lstmR = this.requireWeight('onnx::LSTM_5875_quantized');   // [2, hidden, 4*hidden] = [2, 256, 1024]
    const lstmB = this.requireWeight('onnx::LSTM_5873');             // [2, 8*hidden] = [2, 2048] bias

    const lstmOut = this.createEmptyBuffer(
      seqLen * numDirections * hiddenSize,
      'te_lstm_out'
    );
    this.dispatchLSTM(
      lstmInput, lstmW.buffer, lstmR.buffer, lstmB.buffer, lstmOut,
      seqLen, channels, hiddenSize, numDirections
    );
    intermediates.push(lstmInput);

    // Debug: text encoder LSTM output — matches /text_encoder/lstm/LSTM_output_0 [seqLen, 2, 1, 256]
    await this.captureDebug('/text_encoder/lstm/LSTM_output_0', lstmOut, [seqLen, 2, 1, hiddenSize]);

    // Cleanup intermediates — defer since buffers may be in pending batch encoder
    for (const buf of intermediates) {
      buf.destroy();
    }

    return lstmOut; // [seqLen, 2, 256] = [seqLen, 512]
  }

  // ── Additional Dispatch Methods ───────────────────────────────────────────
  // All dispatch methods accept an optional GPUComputePassEncoder for batching.
  // When provided, dispatches are recorded into the shared pass (no separate submit).
  // When omitted, a standalone command encoder + pass is created and submitted.

  private dispatchLayerNorm(
    input: GPUBuffer,
    gamma: GPUBuffer,
    beta: GPUBuffer,
    output: GPUBuffer,
    batchSize: number,
    hiddenSize: number,
    eps: number,
    pass?: GPUComputePassEncoder
  ): void {
    const pipeline = this.pipelines.get('layerNorm')!;
    // Struct layout: { batch_size: u32, hidden_size: u32, eps: f32 } = 12 bytes
    const paramBuf = new ArrayBuffer(12);
    new Uint32Array(paramBuf, 0, 2).set([batchSize, hiddenSize]);
    new Float32Array(paramBuf, 8, 1).set([eps]);
    const params = this.createUniformBuffer(new Uint8Array(paramBuf), 'ln_params');

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: gamma } },
        { binding: 2, resource: { buffer: beta } },
        { binding: 3, resource: { buffer: output } },
        { binding: 4, resource: { buffer: params } },
      ],
    });

    if (pass) {
      pass.setPipeline(pipeline.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(batchSize / 256));
    } else {
      this.dispatchSingle('layerNorm', bindGroup, Math.ceil(batchSize / 256));
    }
  }

  private dispatchMatmul(
    A: GPUBuffer,
    B: GPUBuffer,
    bias: GPUBuffer,
    output: GPUBuffer,
    M: number,
    K: number,
    N: number,
    useBias: boolean,
    pass?: GPUComputePassEncoder
  ): void {
    const pipeline = this.pipelines.get('matmul')!;
    const params = this.createUniformBuffer(
      new Uint32Array([M, K, N, useBias ? 1 : 0]),
      'matmul_params'
    );

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: A } },
        { binding: 1, resource: { buffer: B } },
        { binding: 2, resource: { buffer: bias } },
        { binding: 3, resource: { buffer: output } },
        { binding: 4, resource: { buffer: params } },
      ],
    });

    if (pass) {
      pass.setPipeline(pipeline.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(M / 16), Math.ceil(N / 16));
    } else {
      this.dispatchSingle('matmul', bindGroup, Math.ceil(M / 16), Math.ceil(N / 16));
    }
  }

  private dispatchGelu(input: GPUBuffer, output: GPUBuffer, size: number): void {
    const pipeline = this.pipelines.get('gelu')!;
    const params = this.createUniformBuffer(new Uint32Array([size]), 'gelu_params');
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
        { binding: 2, resource: { buffer: params } },
      ],
    });
    this.dispatchSingle('gelu', bindGroup, Math.ceil(size / 256));
  }

  private dispatchMatmulGelu(
    A: GPUBuffer,
    B: GPUBuffer,
    bias: GPUBuffer,
    output: GPUBuffer,
    M: number,
    K: number,
    N: number,
    pass?: GPUComputePassEncoder
  ): void {
    const pipeline = this.pipelines.get('matmulGelu')!;
    const params = this.createUniformBuffer(
      new Uint32Array([M, K, N]),
      'matmul_gelu_params'
    );

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: A } },
        { binding: 1, resource: { buffer: B } },
        { binding: 2, resource: { buffer: bias } },
        { binding: 3, resource: { buffer: output } },
        { binding: 4, resource: { buffer: params } },
      ],
    });

    if (pass) {
      pass.setPipeline(pipeline.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(M / 16), Math.ceil(N / 16));
    } else {
      this.dispatchSingle('matmulGelu', bindGroup, Math.ceil(M / 16), Math.ceil(N / 16));
    }
  }

  private dispatchMHA(
    Q: GPUBuffer,
    K: GPUBuffer,
    V: GPUBuffer,
    output: GPUBuffer,
    seqLen: number,
    numHeads: number,
    headDim: number,
    scale: number,
    pass?: GPUComputePassEncoder
  ): void {
    const pipeline = this.pipelines.get('mha')!;

    // Pack params: seq_len, num_heads, head_dim as u32, scale as f32
    const paramData = new ArrayBuffer(16);
    const u32View = new Uint32Array(paramData);
    const f32View = new Float32Array(paramData);
    u32View[0] = seqLen;
    u32View[1] = numHeads;
    u32View[2] = headDim;
    f32View[3] = scale;

    const params = this.createUniformBuffer(new Uint32Array(paramData), 'mha_params');

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: Q } },
        { binding: 1, resource: { buffer: K } },
        { binding: 2, resource: { buffer: V } },
        { binding: 3, resource: { buffer: output } },
        { binding: 4, resource: { buffer: params } },
      ],
    });

    if (pass) {
      pass.setPipeline(pipeline.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(headDim / 64), numHeads * seqLen);
    } else {
      this.dispatchSingle('mha', bindGroup, Math.ceil(headDim / 64), numHeads * seqLen);
    }
  }

  private dispatchAddPass(
    a: GPUBuffer,
    b: GPUBuffer,
    output: GPUBuffer,
    size: number,
    pass?: GPUComputePassEncoder
  ): void {
    const pipeline = this.pipelines.get('add')!;
    const params = this.createUniformBuffer(new Uint32Array([size]), 'add_params');

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: a } },
        { binding: 1, resource: { buffer: b } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: params } },
      ],
    });

    if (pass) {
      pass.setPipeline(pipeline.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(size / 256));
    } else {
      this.dispatchSingle('add', bindGroup, Math.ceil(size / 256));
    }
  }

  // ── Text Encoder Dispatch Methods ─────────────────────────────────────────

  private dispatchConv1d(
    input: GPUBuffer,
    weight: GPUBuffer,
    bias: GPUBuffer,
    output: GPUBuffer,
    inChannels: number,
    outChannels: number,
    kernelSize: number,
    inputLength: number,
    outputLength: number,
    padding: number,
    stride: number,
    dilation: number,
    useBias: boolean
  ): void {
    const { bindGroupLayout } = this.pipelines.get('conv1d')!;
    const params = this.createUniformBuffer(
      new Uint32Array([inChannels, outChannels, kernelSize, inputLength, outputLength, padding, stride, dilation, useBias ? 1 : 0]),
      'conv1d_params'
    );

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: weight } },
        { binding: 2, resource: { buffer: bias } },
        { binding: 3, resource: { buffer: output } },
        { binding: 4, resource: { buffer: params } },
      ],
    });

    this.dispatchSingle('conv1d', bindGroup, Math.ceil((outChannels * outputLength) / 256));
  }

  private dispatchTranspose(
    input: GPUBuffer,
    output: GPUBuffer,
    rows: number,
    cols: number
  ): void {
    const { bindGroupLayout } = this.pipelines.get('transpose')!;
    const params = this.createUniformBuffer(new Uint32Array([rows, cols]), 'transpose_params');

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
        { binding: 2, resource: { buffer: params } },
      ],
    });

    this.dispatchSingle('transpose', bindGroup, Math.ceil((rows * cols) / 256));
  }

  private dispatchLeakyRelu(
    input: GPUBuffer,
    output: GPUBuffer,
    size: number,
    alpha: number
  ): void {
    const { bindGroupLayout } = this.pipelines.get('leakyRelu')!;

    // LeakyReLU params: { size: u32, alpha: f32 }
    const paramData = new ArrayBuffer(8);
    new Uint32Array(paramData, 0, 1).set([size]);
    new Float32Array(paramData, 4, 1).set([alpha]);
    const params = this.createUniformBuffer(new Uint8Array(paramData), 'leaky_relu_params');

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
        { binding: 2, resource: { buffer: params } },
      ],
    });

    this.dispatchSingle('leakyRelu', bindGroup, Math.ceil(size / 256));
  }

  private dispatchSigmoid(
    input: GPUBuffer,
    output: GPUBuffer,
    size: number
  ): void {
    const { bindGroupLayout } = this.pipelines.get('sigmoid')!;
    const paramData = new ArrayBuffer(4);
    new Uint32Array(paramData, 0, 1).set([size]);
    const params = this.createUniformBuffer(new Uint8Array(paramData), 'sigmoid_params');

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
        { binding: 2, resource: { buffer: params } },
      ],
    });

    this.dispatchSingle('sigmoid', bindGroup, Math.ceil(size / 256));
  }

  private dispatchLSTM(
    input: GPUBuffer,
    W: GPUBuffer,
    R: GPUBuffer,
    bias: GPUBuffer,
    output: GPUBuffer,
    seqLen: number,
    inputSize: number,
    hiddenSize: number,
    numDirections: number
  ): void {
    const pipeline = this.pipelines.get('lstm')!;
    const params = this.createUniformBuffer(
      new Uint32Array([seqLen, inputSize, hiddenSize, numDirections]),
      'lstm_params'
    );

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: W } },
        { binding: 2, resource: { buffer: R } },
        { binding: 3, resource: { buffer: bias } },
        { binding: 4, resource: { buffer: output } },
        { binding: 5, resource: { buffer: params } },
      ],
    });

    this.dispatchSingle('lstm', bindGroup, 1, numDirections);
  }

  // ── AdaIN ResNet Block (used in N predictor, F0 predictor) ──────────────

  /**
   * Run one AdaIN ResNet block.
   *
   * Pattern:
   *   norm1: InstanceNorm(x) → AdaIN(FC(style)) → LeakyReLU
   *   conv1: Conv1d(in→out, k=3, pad=1)
   *   norm2: InstanceNorm → AdaIN(FC(style)) → LeakyReLU
   *   conv2: Conv1d(out→out, k=3, pad=1)
   *   residual: x + conv2_out (with optional conv1x1 for channel change)
   *
   * Some blocks also have sigmoid → divide by 2 (block 0 in N/F0 predictor).
   */
  private async runAdaINResNetBlock(
    input: GPUBuffer,
    style: GPUBuffer, // [128]
    inChannels: number,
    length: number,
    prefix: string,   // e.g. 'kmodel.predictor.N.0'
    hasConv1x1: boolean,  // block 1 has conv1x1 residual projection
    outChannels: number,
    pool?: { weightName: string; channels: number }, // For N.1, F0.1: depthwise ConvTranspose pool
  ): Promise<{ output: GPUBuffer; outChannels: number; outLength: number }> {
    const intermediates: GPUBuffer[] = [];
    let curLength = length;

    // ── norm1: InstanceNorm + AdaIN ──
    const norm1Out = this.createEmptyBuffer(inChannels * curLength, 'adain_norm1');
    this.dispatchInstanceNorm(input, norm1Out, inChannels, curLength, 1e-5);

    // AdaIN FC: style[128] → [2*inChannels] via FC weight/bias
    const norm1FcWeight = this.requireWeight(`${prefix}.norm1.fc.weight_quantized`);
    const norm1FcBias = this.requireWeight(`${prefix}.norm1.fc.bias`);
    const norm1Style = this.createEmptyBuffer(norm1FcBias.size, 'norm1_style');
    this.dispatchMatmul(style, norm1FcWeight.buffer, norm1FcBias.buffer, norm1Style, 1, 128, norm1FcBias.size, true);

    const adain1Out = this.createEmptyBuffer(inChannels * curLength, 'adain1_out');
    this.dispatchAdaIN(norm1Out, norm1Style, adain1Out, inChannels, curLength);
    intermediates.push(norm1Out, norm1Style);

    // LeakyReLU
    const act1 = this.createEmptyBuffer(inChannels * curLength, 'adain_act1');
    this.dispatchLeakyRelu(adain1Out, act1, inChannels * curLength, 0.2);
    intermediates.push(adain1Out);

    // ── Pool: depthwise ConvTranspose1d (stride=2, k=3, pad=1, output_pad=1) ──
    // Inserted between act1 and conv1, doubles temporal resolution
    let conv1Input = act1;
    if (pool) {
      const poolWeight = this.requireWeight(pool.weightName);
      // Output length: (L-1)*2 - 2*1 + 1*(3-1) + 1 + 1 = 2L
      const poolOutLength = curLength * 2;
      const poolOut = this.createEmptyBuffer(pool.channels * poolOutLength, 'pool_out');
      this.dispatchDepthwiseConvTranspose1d(act1, poolWeight.buffer, poolOut, pool.channels, 3, curLength, poolOutLength, 2, 1);
      intermediates.push(act1);
      conv1Input = poolOut;
      curLength = poolOutLength;
    }

    // ── conv1: Conv1d(in→out, k=3, pad=1) ──
    const conv1Weight = this.requireWeight(`${prefix}.conv1.weight_quantized`);
    const conv1Bias = this.requireWeight(`${prefix}.conv1.bias`);
    const conv1Out = this.createEmptyBuffer(outChannels * curLength, 'adain_conv1');
    this.dispatchConv1d(conv1Input, conv1Weight.buffer, conv1Bias.buffer, conv1Out, inChannels, outChannels, 3, curLength, curLength, 1, 1, 1, true);
    intermediates.push(conv1Input);

    // ── norm2: InstanceNorm + AdaIN ──
    const norm2Out = this.createEmptyBuffer(outChannels * curLength, 'adain_norm2');
    this.dispatchInstanceNorm(conv1Out, norm2Out, outChannels, curLength, 1e-5);

    const norm2FcWeight = this.requireWeight(`${prefix}.norm2.fc.weight_quantized`);
    const norm2FcBias = this.requireWeight(`${prefix}.norm2.fc.bias`);
    const norm2Style = this.createEmptyBuffer(norm2FcBias.size, 'norm2_style');
    this.dispatchMatmul(style, norm2FcWeight.buffer, norm2FcBias.buffer, norm2Style, 1, 128, norm2FcBias.size, true);

    const adain2Out = this.createEmptyBuffer(outChannels * curLength, 'adain2_out');
    this.dispatchAdaIN(norm2Out, norm2Style, adain2Out, outChannels, curLength);
    intermediates.push(conv1Out, norm2Out, norm2Style);

    // LeakyReLU
    const act2 = this.createEmptyBuffer(outChannels * curLength, 'adain_act2');
    this.dispatchLeakyRelu(adain2Out, act2, outChannels * curLength, 0.2);
    intermediates.push(adain2Out);

    // ── conv2: Conv1d(out→out, k=3, pad=1) ──
    const conv2Weight = this.requireWeight(`${prefix}.conv2.weight_quantized`);
    const conv2Bias = this.requireWeight(`${prefix}.conv2.bias`);
    const conv2Out = this.createEmptyBuffer(outChannels * curLength, 'adain_conv2');
    this.dispatchConv1d(act2, conv2Weight.buffer, conv2Bias.buffer, conv2Out, outChannels, outChannels, 3, curLength, curLength, 1, 1, 1, true);
    intermediates.push(act2);

    // ── Residual connection ──
    let residual: GPUBuffer;
    if (pool) {
      // Pool blocks: Resize(2x) on input, then conv1x1 for channel projection
      const resized = this.createEmptyBuffer(inChannels * curLength, 'adain_resized');
      this.dispatchResize1d(input, resized, inChannels, length, curLength);
      const conv1x1Weight = this.requireWeight(`${prefix}.conv1x1.weight_quantized`);
      residual = this.createEmptyBuffer(outChannels * curLength, 'adain_res_proj');
      this.dispatchConv1d(resized, conv1x1Weight.buffer, conv1x1Weight.buffer, residual, inChannels, outChannels, 1, curLength, curLength, 0, 1, 1, false);
      intermediates.push(resized);
    } else if (hasConv1x1 && inChannels !== outChannels) {
      const conv1x1Weight = this.requireWeight(`${prefix}.conv1x1.weight_quantized`);
      residual = this.createEmptyBuffer(outChannels * curLength, 'adain_res_proj');
      this.dispatchConv1d(input, conv1x1Weight.buffer, conv1x1Weight.buffer, residual, inChannels, outChannels, 1, curLength, curLength, 0, 1, 1, false);
    } else {
      residual = input; // Same channels, direct residual
    }

    // (conv2 + residual) / √2 — normalized residual connection
    const SQRT2_INV = 1 / Math.SQRT2;
    const rawSum = this.createEmptyBuffer(outChannels * curLength, 'adain_raw_sum');
    this.dispatchAdd(conv2Out, residual, rawSum, outChannels * curLength);
    const output = this.createEmptyBuffer(outChannels * curLength, 'adain_block_out');
    this.dispatchScale(rawSum, output, outChannels * curLength, SQRT2_INV);
    intermediates.push(conv2Out, rawSum);
    if ((pool) || (hasConv1x1 && inChannels !== outChannels)) {
      intermediates.push(residual);
    }

    // Cleanup — defer destruction since buffers may be referenced by pending dispatches
    for (const buf of intermediates) buf.destroy();

    return { output, outChannels, outLength: curLength };
  }

  // ── Decoder Block (encode / decode.0-3) ──────────────────────────────────

  /**
   * Run one decoder AdaIN conv block.
   *
   * Pattern:
   *   norm1: InstanceNorm(input) + AdaIN(FC(style))
   *   conv1: Conv1d(in→out, k=3, pad=1) + LeakyReLU
   *   norm2: InstanceNorm + AdaIN(FC(style))
   *   conv2: Conv1d(out→out, k=3, pad=1) + LeakyReLU
   *   residual: conv1x1(in→out) + output
   */
  private async runDecoderBlock(
    input: GPUBuffer,
    style: GPUBuffer, // [128]
    inChannels: number,
    outChannels: number,
    length: number,
    prefix: string, // e.g. 'kmodel.decoder.encode'
    hasConv1x1: boolean,
    pool?: { weightName: string; channels: number }, // For decode.3: depthwise ConvTranspose pool
  ): Promise<GPUBuffer> {
    const intermediates: GPUBuffer[] = [];
    let curLength = length;

    // ── norm1 + AdaIN ──
    const norm1Out = this.createEmptyBuffer(inChannels * curLength, 'dec_norm1');
    this.dispatchInstanceNorm(input, norm1Out, inChannels, curLength, 1e-5);

    // AdaIN FC
    const norm1FcWeight = this.requireWeight(`${prefix}.norm1.fc.weight_quantized`);
    const norm1FcBias = this.requireWeight(`${prefix}.norm1.fc.bias`);
    const norm1FcOutSize = norm1FcBias.size;
    const norm1Style = this.createEmptyBuffer(norm1FcOutSize, 'dec_norm1_style');
    this.dispatchMatmul(style, norm1FcWeight.buffer, norm1FcBias.buffer, norm1Style, 1, 128, norm1FcOutSize, true);

    const adain1Out = this.createEmptyBuffer(inChannels * curLength, 'dec_adain1');
    this.dispatchAdaIN(norm1Out, norm1Style, adain1Out, inChannels, curLength);
    intermediates.push(norm1Out, norm1Style);

    // LeakyReLU
    const act1 = this.createEmptyBuffer(inChannels * curLength, 'dec_act1');
    this.dispatchLeakyRelu(adain1Out, act1, inChannels * curLength, 0.2);
    intermediates.push(adain1Out);

    // ── Pool: depthwise ConvTranspose1d (stride=2, k=3, pad=1, output_pad=1) ──
    // Inserted between act1 and conv1, doubles temporal resolution
    let conv1Input = act1;
    if (pool) {
      const poolWeight = this.requireWeight(pool.weightName);
      const poolOutLength = curLength * 2;
      const poolOut = this.createEmptyBuffer(pool.channels * poolOutLength, 'dec_pool_out');
      this.dispatchDepthwiseConvTranspose1d(act1, poolWeight.buffer, poolOut, pool.channels, 3, curLength, poolOutLength, 2, 1);
      intermediates.push(act1);
      conv1Input = poolOut;
      curLength = poolOutLength;
    }

    // ── conv1: Conv1d(in→out, k=3, pad=1) ──
    const conv1Weight = this.requireWeight(`${prefix}.conv1.weight_quantized`);
    const conv1Bias = this.requireWeight(`${prefix}.conv1.bias`);
    const conv1Out = this.createEmptyBuffer(outChannels * curLength, 'dec_conv1');
    this.dispatchConv1d(conv1Input, conv1Weight.buffer, conv1Bias.buffer, conv1Out, inChannels, outChannels, 3, curLength, curLength, 1, 1, 1, true);
    intermediates.push(conv1Input);

    // ── norm2 + AdaIN ──
    const norm2Out = this.createEmptyBuffer(outChannels * curLength, 'dec_norm2');
    this.dispatchInstanceNorm(conv1Out, norm2Out, outChannels, curLength, 1e-5);

    const norm2FcWeight = this.requireWeight(`${prefix}.norm2.fc.weight_quantized`);
    const norm2FcBias = this.requireWeight(`${prefix}.norm2.fc.bias`);
    const norm2FcOutSize = norm2FcBias.size;
    const norm2Style = this.createEmptyBuffer(norm2FcOutSize, 'dec_norm2_style');
    this.dispatchMatmul(style, norm2FcWeight.buffer, norm2FcBias.buffer, norm2Style, 1, 128, norm2FcOutSize, true);

    const adain2Out = this.createEmptyBuffer(outChannels * curLength, 'dec_adain2');
    this.dispatchAdaIN(norm2Out, norm2Style, adain2Out, outChannels, curLength);
    intermediates.push(conv1Out, norm2Out, norm2Style);

    // LeakyReLU
    const act2 = this.createEmptyBuffer(outChannels * curLength, 'dec_act2');
    this.dispatchLeakyRelu(adain2Out, act2, outChannels * curLength, 0.2);
    intermediates.push(adain2Out);

    // ── conv2: Conv1d(out→out, k=3, pad=1) ──
    const conv2Weight = this.requireWeight(`${prefix}.conv2.weight_quantized`);
    const conv2Bias = this.requireWeight(`${prefix}.conv2.bias`);
    const conv2Out = this.createEmptyBuffer(outChannels * curLength, 'dec_conv2');
    this.dispatchConv1d(act2, conv2Weight.buffer, conv2Bias.buffer, conv2Out, outChannels, outChannels, 3, curLength, curLength, 1, 1, 1, true);
    intermediates.push(act2);

    // ── Residual: (conv1x1(in→out) + conv2) / √2 ──
    // The √2 normalization keeps activations from growing at each residual block
    const SQRT2_INV = 1 / Math.SQRT2; // 0.7071067811865476
    let rawSum: GPUBuffer;
    if (pool) {
      // Pool blocks: Resize(2x) on input for residual, then conv1x1
      const resized = this.createEmptyBuffer(inChannels * curLength, 'dec_resized');
      this.dispatchResize1d(input, resized, inChannels, length, curLength);
      const conv1x1Weight = this.requireWeight(`${prefix}.conv1x1.weight_quantized`);
      const residual = this.createEmptyBuffer(outChannels * curLength, 'dec_res_proj');
      this.dispatchConv1d(resized, conv1x1Weight.buffer, conv1x1Weight.buffer, residual, inChannels, outChannels, 1, curLength, curLength, 0, 1, 1, false);
      rawSum = this.createEmptyBuffer(outChannels * curLength, 'dec_raw_sum');
      this.dispatchAdd(conv2Out, residual, rawSum, outChannels * curLength);
      intermediates.push(conv2Out, resized, residual);
    } else if (hasConv1x1) {
      const conv1x1Weight = this.requireWeight(`${prefix}.conv1x1.weight_quantized`);
      const residual = this.createEmptyBuffer(outChannels * curLength, 'dec_res_proj');
      this.dispatchConv1d(input, conv1x1Weight.buffer, conv1x1Weight.buffer, residual, inChannels, outChannels, 1, curLength, curLength, 0, 1, 1, false);

      rawSum = this.createEmptyBuffer(outChannels * curLength, 'dec_raw_sum');
      this.dispatchAdd(conv2Out, residual, rawSum, outChannels * curLength);
      intermediates.push(conv2Out, residual);
    } else {
      rawSum = conv2Out;
    }

    // Normalize residual by √2 (ONNX: Div by 1.4142135)
    const output = this.createEmptyBuffer(outChannels * curLength, 'dec_block_out');
    this.dispatchScale(rawSum, output, outChannels * curLength, SQRT2_INV);
    intermediates.push(rawSum);

    for (const buf of intermediates) buf.destroy();
    return output;
  }

  // ── Build decode block input (concat features + asr_res + F0/N) ────────

  private buildDecodeInput(
    features: GPUBuffer,
    f0Conv: GPUBuffer,
    nConv: GPUBuffer,
    asrRes: GPUBuffer | null,
    featureChannels: number,
    length: number,
  ): GPUBuffer {
    // Total channels: featureChannels + 64 (asr_res) + 2 (F0 + N) = featureChannels + 66
    // Chain GPU concat: features+asrRes → concat1, concat1+f0 → concat2, concat2+n → final
    const asrBuf = asrRes ?? this.createEmptyBuffer(64 * length, 'asr_zero');

    // Step 1: features[featureChannels] + asrRes[64]
    const concat1 = this.createEmptyBuffer((featureChannels + 64) * length, 'dec_concat1');
    this.dispatchConcatChannels(features, asrBuf, concat1, featureChannels, 64, length);

    // Step 2: concat1[featureChannels+64] + f0[1]
    const concat2 = this.createEmptyBuffer((featureChannels + 65) * length, 'dec_concat2');
    this.dispatchConcatChannels(concat1, f0Conv, concat2, featureChannels + 64, 1, length);
    concat1.destroy();

    // Step 3: concat2[featureChannels+65] + n[1]
    const output = this.createEmptyBuffer((featureChannels + 66) * length, 'dec_concat3');
    this.dispatchConcatChannels(concat2, nConv, output, featureChannels + 65, 1, length);
    concat2.destroy();

    if (!asrRes) asrBuf.destroy();

    return output;
  }

  // ── HiFi-GAN ResBlock ───────────────────────────────────────────────────

  /**
   * Run one HiFi-GAN residual block with AdaIN and Snake activation.
   *
   * ONNX pattern per iteration i:
   *   InstanceNorm(input) → AdaIN1(style) → Snake(alpha1) → conv1(dilated)
   *   → InstanceNorm → AdaIN2(style) → Snake(alpha2) → conv2
   *   → Add(conv2, input)  [simple residual, no alpha scaling]
   *
   * Snake activation: x + (1/alpha) * sin²(alpha * x)
   * alpha1/alpha2 are Snake parameters [1, C, 1], NOT residual weights.
   */
  private async runHiFiGANResBlock(
    input: GPUBuffer,
    style: GPUBuffer, // [128]
    channels: number,
    length: number,
    prefix: string, // e.g. 'kmodel.decoder.generator.resblocks.0'
  ): Promise<GPUBuffer> {
    let current = input;

    for (let i = 0; i < 3; i++) {
      // Destroy intermediates per-iteration to reduce peak GPU memory.
      // At 128×17881 (ups.1 resolution), each buffer is ~9MB.
      // Accumulating all 3 iterations = ~275MB; per-iteration = ~91MB peak. Critical for iOS.
      const iterBufs: GPUBuffer[] = [];

      // ── InstanceNorm → AdaIN1 → Snake(alpha1) ──
      const norm1 = this.createEmptyBuffer(channels * length, `hifi_norm1_${i}`);
      this.dispatchInstanceNorm(current, norm1, channels, length, 1e-5);

      const adain1FcWeight = this.requireWeight(`${prefix}.adain1.${i}.fc.weight_quantized`);
      const adain1FcBias = this.requireWeight(`${prefix}.adain1.${i}.fc.bias`);
      const adain1Style = this.createEmptyBuffer(adain1FcBias.size, `hifi_adain1_style_${i}`);
      this.dispatchMatmul(style, adain1FcWeight.buffer, adain1FcBias.buffer, adain1Style, 1, 128, adain1FcBias.size, true);

      const adain1Out = this.createEmptyBuffer(channels * length, `hifi_adain1_${i}`);
      this.dispatchAdaIN(norm1, adain1Style, adain1Out, channels, length);
      iterBufs.push(norm1, adain1Style);

      // Snake activation: x + (1/alpha) * sin²(alpha * x)
      const alpha1Weight = this.requireWeight(`${prefix}.alpha1.${i}`);
      const snake1Out = this.createEmptyBuffer(channels * length, `hifi_snake1_${i}`);
      this.dispatchSnake(adain1Out, alpha1Weight.buffer, snake1Out, channels, length);
      iterBufs.push(adain1Out);

      // ── conv1 (dilated) ──
      const conv1Weight = this.requireWeight(`${prefix}.convs1.${i}.weight_quantized`);
      const conv1Bias = this.requireWeight(`${prefix}.convs1.${i}.bias`);
      const kernelSize = conv1Weight.shape[2];
      const dilation = [1, 3, 5][i];
      const padding = Math.floor((kernelSize * dilation - dilation) / 2);
      const conv1Out = this.createEmptyBuffer(channels * length, `hifi_conv1_${i}`);
      this.dispatchConv1d(snake1Out, conv1Weight.buffer, conv1Bias.buffer, conv1Out, channels, channels, kernelSize, length, length, padding, 1, dilation, true);
      iterBufs.push(snake1Out);

      // ── InstanceNorm → AdaIN2 → Snake(alpha2) ──
      const norm2 = this.createEmptyBuffer(channels * length, `hifi_norm2_${i}`);
      this.dispatchInstanceNorm(conv1Out, norm2, channels, length, 1e-5);

      const adain2FcWeight = this.requireWeight(`${prefix}.adain2.${i}.fc.weight_quantized`);
      const adain2FcBias = this.requireWeight(`${prefix}.adain2.${i}.fc.bias`);
      const adain2Style = this.createEmptyBuffer(adain2FcBias.size, `hifi_adain2_style_${i}`);
      this.dispatchMatmul(style, adain2FcWeight.buffer, adain2FcBias.buffer, adain2Style, 1, 128, adain2FcBias.size, true);

      const adain2Out = this.createEmptyBuffer(channels * length, `hifi_adain2_${i}`);
      this.dispatchAdaIN(norm2, adain2Style, adain2Out, channels, length);
      iterBufs.push(conv1Out, norm2, adain2Style);

      // Snake activation
      const alpha2Weight = this.requireWeight(`${prefix}.alpha2.${i}`);
      const snake2Out = this.createEmptyBuffer(channels * length, `hifi_snake2_${i}`);
      this.dispatchSnake(adain2Out, alpha2Weight.buffer, snake2Out, channels, length);
      iterBufs.push(adain2Out);

      // ── conv2 ──
      const conv2Weight = this.requireWeight(`${prefix}.convs2.${i}.weight_quantized`);
      const conv2Bias = this.requireWeight(`${prefix}.convs2.${i}.bias`);
      const k2 = conv2Weight.shape[2];
      const pad2 = Math.floor((k2 - 1) / 2);
      const conv2Out = this.createEmptyBuffer(channels * length, `hifi_conv2_${i}`);
      this.dispatchConv1d(snake2Out, conv2Weight.buffer, conv2Bias.buffer, conv2Out, channels, channels, k2, length, length, pad2, 1, 1, true);
      iterBufs.push(snake2Out);

      // ── Simple residual: output = conv2 + input ──
      const resOut = this.createEmptyBuffer(channels * length, `hifi_res_${i}`);
      this.dispatchAdd(conv2Out, current, resOut, channels * length);
      iterBufs.push(conv2Out);
      if (current !== input) iterBufs.push(current);
      current = resOut;

      // Defer destruction — buffers may be referenced by pending dispatches in batch encoder
      for (const buf of iterBufs) buf.destroy();
    }

    return current;
  }

  // ── Additional Dispatch Methods ─────────────────────────────────────────

  private dispatchSnake(
    input: GPUBuffer,
    alpha: GPUBuffer, // [C] (flattened from [1, C, 1])
    output: GPUBuffer,
    channels: number,
    length: number,
  ): void {
    const { bindGroupLayout } = this.pipelines.get('snake')!;
    const params = this.createUniformBuffer(new Uint32Array([channels, length]), 'snake_params');

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: alpha } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: params } },
      ],
    });

    this.dispatchSingle('snake', bindGroup, Math.ceil((channels * length) / 256));
  }

  private dispatchInstanceNorm(
    input: GPUBuffer,
    output: GPUBuffer,
    channels: number,
    length: number,
    eps: number,
  ): void {
    const { bindGroupLayout } = this.pipelines.get('instanceNorm')!;
    const paramData = new ArrayBuffer(12);
    new Uint32Array(paramData, 0, 2).set([channels, length]);
    new Float32Array(paramData, 8, 1).set([eps]);
    const params = this.createUniformBuffer(new Uint8Array(paramData), 'instnorm_params');

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
        { binding: 2, resource: { buffer: params } },
      ],
    });

    this.dispatchSingle('instanceNorm', bindGroup, Math.ceil(channels / 256));
  }

  private dispatchAdaIN(
    normed: GPUBuffer,
    styleFc: GPUBuffer, // [2*channels] — first half is scale, second half is bias
    output: GPUBuffer,
    channels: number,
    length: number,
  ): void {
    // styleFc layout: [scale_0..scale_{C-1}, bias_0..bias_{C-1}]
    // The shader reads from a single buffer and splits internally
    // (avoids buffer binding offset alignment issues with non-power-of-2 channel counts)
    const { bindGroupLayout } = this.pipelines.get('adain')!;
    const params = this.createUniformBuffer(new Uint32Array([channels, length]), 'adain_params');

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: normed } },
        { binding: 1, resource: { buffer: styleFc } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: params } },
      ],
    });

    this.dispatchSingle('adain', bindGroup, Math.ceil((channels * length) / 256));
  }

  private dispatchConvTranspose1d(
    input: GPUBuffer,
    weight: GPUBuffer,
    bias: GPUBuffer,
    output: GPUBuffer,
    inChannels: number,
    outChannels: number,
    kernelSize: number,
    inputLength: number,
    outputLength: number,
    stride: number,
    padding: number,
    useBias: boolean,
  ): void {
    const { bindGroupLayout } = this.pipelines.get('convTranspose1d')!;
    const params = this.createUniformBuffer(
      new Uint32Array([inChannels, outChannels, kernelSize, inputLength, outputLength, stride, padding, useBias ? 1 : 0]),
      'conv_transpose_params'
    );

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: weight } },
        { binding: 2, resource: { buffer: bias } },
        { binding: 3, resource: { buffer: output } },
        { binding: 4, resource: { buffer: params } },
      ],
    });

    this.dispatchSingle('convTranspose1d', bindGroup, Math.ceil((outChannels * outputLength) / 256));
  }

  /**
   * Depthwise ConvTranspose1d: each channel processed independently (groups=channels).
   * Used for pool layers in N.1, F0.1, decode.3 that double temporal resolution.
   * Weight shape: [channels, 1, kernel_size] stored as [channels * kernel_size].
   */
  private dispatchDepthwiseConvTranspose1d(
    input: GPUBuffer,
    weight: GPUBuffer,
    output: GPUBuffer,
    channels: number,
    kernelSize: number,
    inputLength: number,
    outputLength: number,
    stride: number,
    padding: number,
  ): void {
    const { bindGroupLayout } = this.pipelines.get('depthwiseConvTranspose1d')!;
    const params = this.createUniformBuffer(
      new Uint32Array([channels, kernelSize, inputLength, outputLength, stride, padding]),
      'dw_conv_transpose_params'
    );

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: weight } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: params } },
      ],
    });

    this.dispatchSingle('depthwiseConvTranspose1d', bindGroup, Math.ceil((channels * outputLength) / 256));
  }

  /**
   * Resize 1D: nearest-neighbor interpolation (typically 2x upsampling).
   * Used on the residual path of blocks with pools.
   */
  private dispatchResize1d(
    input: GPUBuffer,
    output: GPUBuffer,
    channels: number,
    inputLength: number,
    outputLength: number,
  ): void {
    const { bindGroupLayout } = this.pipelines.get('resize1d')!;
    const params = this.createUniformBuffer(
      new Uint32Array([channels, inputLength, outputLength]),
      'resize1d_params'
    );

    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
        { binding: 2, resource: { buffer: params } },
      ],
    });

    this.dispatchSingle('resize1d', bindGroup, Math.ceil((channels * outputLength) / 256));
  }

  /** Multiply all elements by a constant scale factor. */
  private dispatchScale(input: GPUBuffer, output: GPUBuffer, size: number, scale: number): void {
    const { bindGroupLayout } = this.pipelines.get('scale')!;
    const paramData = new ArrayBuffer(16);
    new Uint32Array(paramData, 0, 2).set([size, 0]);
    new Float32Array(paramData, 8, 1).set([scale]);
    const params = this.createUniformBuffer(new Uint8Array(paramData), 'scale_params');
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
        { binding: 2, resource: { buffer: params } },
      ],
    });
    this.dispatchSingle('scale', bindGroup, Math.ceil(size / 256));
  }

  /** Concatenate two channel-first tensors along channel dimension. */
  private dispatchConcatChannels(
    a: GPUBuffer, b: GPUBuffer, output: GPUBuffer,
    channelsA: number, channelsB: number, length: number,
  ): void {
    const { bindGroupLayout } = this.pipelines.get('concatChannels')!;
    const params = this.createUniformBuffer(
      new Uint32Array([channelsA, channelsB, length]), 'concat_params'
    );
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: a } },
        { binding: 1, resource: { buffer: b } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: params } },
      ],
    });
    this.dispatchSingle('concatChannels', bindGroup, Math.ceil(((channelsA + channelsB) * length) / 256));
  }

  /** AdaIN row-major: normed[rows, C] + style_fc[2*C] → output[rows, C]. */
  private dispatchAdaINRowMajor(
    normed: GPUBuffer, styleFc: GPUBuffer, output: GPUBuffer,
    channels: number, rows: number,
  ): void {
    const { bindGroupLayout } = this.pipelines.get('adainRowMajor')!;
    const total = channels * rows;
    const params = this.createUniformBuffer(new Uint32Array([channels, total]), 'adain_rm_params');
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: normed } },
        { binding: 1, resource: { buffer: styleFc } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: params } },
      ],
    });
    this.dispatchSingle('adainRowMajor', bindGroup, Math.ceil(total / 256));
  }

  /** Concat row-major A[rows, colsA] with broadcast B[colsB] → [rows, colsA+colsB]. */
  private dispatchConcatBroadcast(
    a: GPUBuffer, b: GPUBuffer, output: GPUBuffer,
    rows: number, colsA: number, colsB: number,
  ): void {
    const { bindGroupLayout } = this.pipelines.get('concatBroadcast')!;
    const params = this.createUniformBuffer(
      new Uint32Array([rows, colsA, colsB]), 'concat_bc_params'
    );
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: a } },
        { binding: 1, resource: { buffer: b } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: params } },
      ],
    });
    this.dispatchSingle('concatBroadcast', bindGroup, Math.ceil((rows * (colsA + colsB)) / 256));
  }

  /** GPU length expansion: [seqLen, D] → [totalFrames, D] using duration cumsum. */
  private dispatchExpandRowMajor(
    input: GPUBuffer, cumsum: GPUBuffer, output: GPUBuffer,
    seqLen: number, dim: number, totalFrames: number,
  ): void {
    const { bindGroupLayout } = this.pipelines.get('expandRowMajor')!;
    const params = this.createUniformBuffer(new Uint32Array([seqLen, dim, totalFrames]), 'expand_rm_params');
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: cumsum } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: params } },
      ],
    });
    this.dispatchSingle('expandRowMajor', bindGroup, Math.ceil((totalFrames * dim) / 256));
  }

  /** GPU length expansion with transpose: [seqLen, D] row-major → [D, totalFrames] channel-first. */
  private dispatchExpandChannelFirst(
    input: GPUBuffer, cumsum: GPUBuffer, output: GPUBuffer,
    seqLen: number, dim: number, totalFrames: number,
  ): void {
    const { bindGroupLayout } = this.pipelines.get('expandChannelFirst')!;
    const params = this.createUniformBuffer(new Uint32Array([seqLen, dim, totalFrames]), 'expand_cf_params');
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: cumsum } },
        { binding: 2, resource: { buffer: output } },
        { binding: 3, resource: { buffer: params } },
      ],
    });
    this.dispatchSingle('expandChannelFirst', bindGroup, Math.ceil((totalFrames * dim) / 256));
  }

  /** GPU iSTFT synthesis: conv_post [22, genLen] → waveform [waveformLen]. */
  private dispatchISTFT(
    convPost: GPUBuffer, weightReal: GPUBuffer, weightImag: GPUBuffer,
    output: GPUBuffer, genLength: number, waveformLength: number,
    bins: number, kernelSize: number, stride: number,
  ): void {
    const { bindGroupLayout } = this.pipelines.get('istft')!;
    const params = this.createUniformBuffer(
      new Uint32Array([genLength, waveformLength, bins, kernelSize, stride]), 'istft_params'
    );
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: convPost } },
        { binding: 1, resource: { buffer: weightReal } },
        { binding: 2, resource: { buffer: weightImag } },
        { binding: 3, resource: { buffer: output } },
        { binding: 4, resource: { buffer: params } },
      ],
    });
    this.dispatchSingle('istft', bindGroup, Math.ceil(waveformLength / 256));
  }

  /** Reflection pad 1D along time dimension. */
  private dispatchReflectionPad1d(
    input: GPUBuffer, output: GPUBuffer,
    channels: number, inputLength: number, padLeft: number, padRight: number,
  ): void {
    const { bindGroupLayout } = this.pipelines.get('reflectionPad1d')!;
    const params = this.createUniformBuffer(
      new Uint32Array([channels, inputLength, padLeft, padRight]), 'reflection_pad_params'
    );
    const outLength = inputLength + padLeft + padRight;
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
        { binding: 2, resource: { buffer: params } },
      ],
    });
    this.dispatchSingle('reflectionPad1d', bindGroup, Math.ceil((channels * outLength) / 256));
  }

  /** Alpha-weighted residual: output = current + alpha[ch] * residual */
  private dispatchAlphaResidual(
    current: GPUBuffer, residual: GPUBuffer, alpha: GPUBuffer, output: GPUBuffer,
    channels: number, length: number,
  ): void {
    const pipeline = this.pipelines.get('alphaResidual')!;
    const params = this.createUniformBuffer(
      new Uint32Array([channels, length]), 'alpha_res_params'
    );
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: current } },
        { binding: 1, resource: { buffer: residual } },
        { binding: 2, resource: { buffer: alpha } },
        { binding: 3, resource: { buffer: output } },
        { binding: 4, resource: { buffer: params } },
      ],
    });
    this.dispatchSingle('alphaResidual', bindGroup, Math.ceil((channels * length) / 256));
  }

  // ── Shader Compilation ─────────────────────────────────────────────────────

  private compileShaders(): void {
    const shaderMap: Record<string, string> = {
      embedding: embeddingShader,
      layerNorm: layerNormShader,
      matmul: matmulShader,
      conv1d: conv1dShader,
      instanceNorm: instanceNormShader,
      adain: adainShader,
      adainRowMajor: adainRowMajorShader,
      convTranspose1d: convTranspose1dShader,
      depthwiseConvTranspose1d: depthwiseConvTranspose1dShader,
      resize1d: resize1dShader,
      scale: scaleShader,
      concatChannels: concatChannelsShader,
      concatBroadcast: concatBroadcastShader,
      reflectionPad1d: reflectionPad1dShader,
      alphaResidual: alphaResidualShader,
      snake: snakeShader,
      leakyRelu: leakyReluShader,
      gelu: geluShader,
      tanh: tanhShader,
      sigmoid: sigmoidShader,
      softmax: softmaxShader,
      add: addShader,
      mha: mhaShader,
      matmulGelu: matmulGeluShader,
      transpose: transposeShader,
      lstm: lstmShader,
      istft: istftShader,
      expandRowMajor: expandRowMajorShader,
      expandChannelFirst: expandChannelFirstShader,
    };

    for (const [name, code] of Object.entries(shaderMap)) {
      const module = this.device.createShaderModule({ code, label: name });
      const bindGroupLayout = this.device.createBindGroupLayout({
        entries: this.inferBindGroupLayout(code),
        label: `${name}_layout`,
      });

      const pipeline = this.device.createComputePipeline({
        layout: this.device.createPipelineLayout({
          bindGroupLayouts: [bindGroupLayout],
        }),
        compute: { module, entryPoint: 'main' },
        label: name,
      });

      this.pipelines.set(name, { pipeline, bindGroupLayout });
    }

    console.log(`[KittenTTS] Compiled ${this.pipelines.size} shader pipelines`);
  }

  /** Infer bind group layout from WGSL source by parsing @binding annotations. */
  private inferBindGroupLayout(wgsl: string): GPUBindGroupLayoutEntry[] {
    const entries: GPUBindGroupLayoutEntry[] = [];
    const bindingRegex = /@group\(0\)\s+@binding\((\d+)\)\s+var<(\w+)(?:,\s*(\w+))?>/g;

    let match;
    while ((match = bindingRegex.exec(wgsl)) !== null) {
      const binding = parseInt(match[1]);
      const addressSpace = match[2];
      const accessMode = match[3];

      if (addressSpace === 'uniform') {
        entries.push({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' },
        });
      } else if (addressSpace === 'storage') {
        entries.push({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: accessMode === 'read_write' ? 'storage' : 'read-only-storage' },
        });
      }
    }

    return entries;
  }

  /**
   * Generate source excitation signal for noise injection.
   *
   * Pipeline (computed on CPU — cumulative sum is sequential):
   * 1. Read F0_proj from GPU [1, f0Length]
   * 2. Upsample F0 to waveform rate via nearest-neighbor [waveLen]
   * 3. For 9 harmonics: cumsum(k * f0 / sr) → sin(2π * phase)
   *    Voiced (f0 > 0): scaled sin wave. Unvoiced: Gaussian noise.
   * 4. Linear(9→1) + bias + tanh → [waveLen, 1]
   * 5. Edge-pad 10 → [waveLen + 20]
   * 6. Forward STFT via Conv1d(1→11, k=20, s=5) for real + imag → magnitude + phase
   * 7. Concat [magnitude, phase] → [22, stftLen]
   * 8. Upload to GPU
   */
  private async generateSourceExcitation(
    f0ProjBuf: GPUBuffer,
    f0Length: number,
    stftLen: number,
  ): Promise<GPUBuffer> {
    const SAMPLE_RATE = 24000;
    const NUM_HARMONICS = 9;

    // 1. Read F0_proj from GPU
    const f0Data = await this.readBuffer(f0ProjBuf, f0Length);

    // 2. Upsample F0 to waveform rate (nearest-neighbor)
    // Waveform length from STFT: (stftLen - 1) * 5 + 20 = raw, then edge-pad adds 20
    // So: padded = waveLen + 20, stftLen = (padded - 20) / 5 + 1 = waveLen / 5 + 1
    // => waveLen = (stftLen - 1) * 5
    const waveLen = (stftLen - 1) * 5;
    const f0Upsampled = new Float32Array(waveLen);
    const upsampleRatio = waveLen / f0Length;
    for (let i = 0; i < waveLen; i++) {
      const srcIdx = Math.min(Math.floor(i / upsampleRatio), f0Length - 1);
      f0Upsampled[i] = f0Data[srcIdx];
    }

    // 3. Generate 9 harmonics via cumulative phase
    const harmonics = new Float32Array(waveLen * NUM_HARMONICS);
    const VOICED_SCALE = 0.1;
    const UNVOICED_SCALE = 0.003; // sqrt(0.003) ≈ 0.055
    for (let k = 0; k < NUM_HARMONICS; k++) {
      const harmIdx = k + 1; // harmonics 1-9
      let phase = 0;
      for (let t = 0; t < waveLen; t++) {
        const f0 = f0Upsampled[t];
        const voiced = f0 > 10; // 10 Hz threshold
        if (voiced) {
          phase += f0 * harmIdx / SAMPLE_RATE;
          // Wrap phase to prevent precision loss
          phase -= Math.floor(phase);
          harmonics[t * NUM_HARMONICS + k] = Math.sin(2 * Math.PI * phase) * VOICED_SCALE;
        } else {
          // Gaussian noise for unvoiced regions
          const u1 = Math.random();
          const u2 = Math.random();
          harmonics[t * NUM_HARMONICS + k] = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2) * UNVOICED_SCALE;
          phase = 0; // Reset phase for unvoiced
        }
      }
    }

    // 4. Linear layer: [waveLen, 9] × [9, 1] + bias → tanh → [waveLen]
    // Cache weights on first call to avoid GPU readBuffer round-trips
    if (!this.sinGenWeights) {
      const linearWeight = this.requireWeight('onnx::MatMul_6388');
      const linearBias = this.requireWeight('kmodel.decoder.generator.m_source.l_linear.bias');
      const fwdReal = this.requireWeight('kmodel.decoder.generator.stft.weight_forward_real');
      const fwdImag = this.requireWeight('kmodel.decoder.generator.stft.weight_forward_imag');
      this.sinGenWeights = {
        linearWeight: await this.readBuffer(linearWeight.buffer, 9),
        linearBias: await this.readBuffer(linearBias.buffer, 1),
        fwdReal: await this.readBuffer(fwdReal.buffer, 11 * 20),
        fwdImag: await this.readBuffer(fwdImag.buffer, 11 * 20),
      };
    }
    const { linearWeight: weightData, linearBias: biasData } = this.sinGenWeights;
    const waveSignal = new Float32Array(waveLen);
    for (let t = 0; t < waveLen; t++) {
      let sum = biasData[0];
      for (let k = 0; k < NUM_HARMONICS; k++) {
        sum += harmonics[t * NUM_HARMONICS + k] * weightData[k];
      }
      waveSignal[t] = Math.tanh(sum);
    }

    // 5. Edge-pad 10 on each side → [waveLen + 20]
    const padded = new Float32Array(waveLen + 20);
    for (let i = 0; i < 10; i++) {
      padded[i] = waveSignal[0]; // edge pad left
    }
    for (let i = 0; i < waveLen; i++) {
      padded[i + 10] = waveSignal[i];
    }
    for (let i = 0; i < 10; i++) {
      padded[waveLen + 10 + i] = waveSignal[waveLen - 1]; // edge pad right
    }

    // 6. Forward STFT: two Conv1d(1→11, k=20, stride=5)
    const { fwdReal: fwdRealData, fwdImag: fwdImagData } = this.sinGenWeights;

    const realOut = new Float32Array(11 * stftLen);
    const imagOut = new Float32Array(11 * stftLen);
    const STRIDE = 5;
    const KERNEL = 20;
    for (let bin = 0; bin < 11; bin++) {
      for (let t = 0; t < stftLen; t++) {
        let sumR = 0, sumI = 0;
        const offset = t * STRIDE;
        for (let k = 0; k < KERNEL; k++) {
          const val = padded[offset + k];
          sumR += val * fwdRealData[bin * KERNEL + k];
          sumI += val * fwdImagData[bin * KERNEL + k];
        }
        realOut[bin * stftLen + t] = sumR;
        imagOut[bin * stftLen + t] = sumI;
      }
    }

    // 7. Magnitude + Phase → [22, stftLen]
    const noiseData = new Float32Array(22 * stftLen);
    const EPS = 1e-14;
    for (let bin = 0; bin < 11; bin++) {
      for (let t = 0; t < stftLen; t++) {
        const r = realOut[bin * stftLen + t];
        const im = imagOut[bin * stftLen + t];
        // Magnitude: sqrt(r² + i² + eps)
        noiseData[bin * stftLen + t] = Math.sqrt(r * r + im * im + EPS);
        // Phase: atan2(imag, real)
        noiseData[(11 + bin) * stftLen + t] = Math.atan2(im, r);
      }
    }

    // 8. Upload to GPU
    return this.createBuffer(noiseData, 'noise_source',
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  }

  /** Free GPU weight buffers to release VRAM (~75MB).
   *  Call after generation to prevent iOS Safari jetsam crashes.
   *  Weights must be reloaded via loadModel() before next generate(). */
  freeWeights(): void {
    let freed = 0;
    for (const tensor of this.weights.values()) {
      freed += tensor.buffer.size;
      tensor.buffer.destroy();
    }
    this.weights.clear();
    // Also clear cached sin generator weights (CPU-side)
    this.sinGenWeights = null;
    console.log(`[KittenTTS] Freed ${(freed / 1024 / 1024).toFixed(1)}MB GPU weight buffers`);
  }

  /** Check if model weights are loaded (GPU buffers alive). */
  get weightsLoaded(): boolean {
    return this.weights.size > 0;
  }

  /** Destroy all GPU resources. */
  destroy(): void {
    for (const tensor of this.weights.values()) {
      tensor.buffer.destroy();
    }
    this.weights.clear();
    this.voices.clear();
  }
}
