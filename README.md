<div align="center">

# Nemotron 3.5 ASR Streaming, ONNX

**Whisper large-v3-class multilingual ASR that actually streams, on your CPU.**

[![Hugging Face](https://img.shields.io/badge/ONNX%20weights-Hugging%20Face-yellow?logo=huggingface)](https://huggingface.co/codavidgarcia/nemotron-3.5-asr-streaming-0.6b-onnx)
[![License: Apache 2.0](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![Weights: OpenMDW-1.1](https://img.shields.io/badge/weights-OpenMDW--1.1-green)](https://openmdw.ai/license/1-1/)
[![ONNX opset 17](https://img.shields.io/badge/ONNX-opset_17-005CED?logo=onnx)](https://onnx.ai/)
[![Runtime](https://img.shields.io/badge/runtime-onnxruntime%20CPU%20%7C%20CUDA-lightgrey)](https://onnxruntime.ai/)

[Model weights](https://huggingface.co/codavidgarcia/nemotron-3.5-asr-streaming-0.6b-onnx) ·
[Base model (NVIDIA)](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) ·
[Validation](#validation) ·
[Making of](#the-making-of-the-bug-that-only-appeared-after-the-first-second)

<img src="assets/demo_streaming.gif" alt="Live streaming demo: partial transcripts appearing in real time from the ONNX fp16 engine on CPU" width="850"/>

</div>

Export NVIDIA's [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b),
a multilingual (40 locales) cache-aware FastConformer-RNNT **streaming** ASR
model, to ONNX. Run it with a small streaming engine that only needs
**numpy + onnxruntime**.

**📦 Weights (fp16, ready to download):**
[`codavidgarcia/nemotron-3.5-asr-streaming-0.6b-onnx`](https://huggingface.co/codavidgarcia/nemotron-3.5-asr-streaming-0.6b-onnx)

- 🌍 One 600M model, 40 language-locales, automatic language detection (`<xx-XX>` tags)
- ⚡ True cache-aware streaming: 80 ms to 1.12 s configurable chunks, native punctuation/casing
- 💻 About 3.8x real-time on CPU (RTF 0.26 fp32). No PyTorch, no GPU required.
- 🎯 Measured parity vs the original: **WER 0.0137** (fp16) / **0.0082** (fp32)
- 📉 Whisper large-v3-class multilingual quality that **actually streams**, at about 1/3 of the parameters, on your CPU

> Unofficial community export by [@codavidgarcia](https://huggingface.co/codavidgarcia).
> Weights remain NVIDIA's under [OpenMDW-1.1](https://openmdw.ai/license/1-1/).
> All code here is Apache-2.0 (see `LICENSE`).

## Quickstart

Use the prebuilt ONNX package from Hugging Face:

```bash
pip install numpy onnxruntime soundfile scipy
huggingface-cli download codavidgarcia/nemotron-3.5-asr-streaming-0.6b-onnx --local-dir ./onnx-out

python engine/nemotron_onnx_streaming.py ./onnx-out meeting.wav \
    --language auto --chunk-ms 320 --precision fp16
# text: Actually, the masterminds behind this ...
# detected language: <en-US>
# RTF: 0.287
```

In Python:

```python
from nemotron_onnx_streaming import NemotronOnnxStreaming

engine = NemotronOnnxStreaming("./onnx-out", language="auto", chunk_ms=320, precision="fp16")
engine.accept_waveform(pcm_float32_16k)   # feed any number of samples
print(engine.get_partial())               # live hypothesis
print(engine.get_final())                 # flush + final transcript
print(engine.detected_language, engine.rtf)
```

Regenerate the ONNX graphs yourself (any chunk size, fp32/fp16/int8):

```bash
pip install -r requirements.txt   # torch (CPU ok) + transformers>=5.13 + onnx
python export/export_onnx.py --output-dir ./onnx-out --chunk-ms 80,320,1120 --validate
python export/quantize.py --model-dir ./onnx-out --fp16
python validation/parity.py --wav-dir ./test_wavs --model-dir ./onnx-out --language auto
```

## How it works

<p align="center">
  <img src="assets/model_architecture.png" alt="Architecture: FastConformer encoder, language-ID encoding concatenated and projected into the RNNT decoder" width="850"/>
  <br/><sub>Architecture (image: NVIDIA). The FastConformer encoder output is concatenated with the language-ID encoding and projected into the RNNT decoder.</sub>
</p>

Grounded in the transformers implementation (`transformers.models.nemotron3_5_asr`
+ `nemotron_asr_streaming`, v5.13.0):

- **Encoder** (`encoder_{ms}ms[_first].onnx`): one cache-aware FastConformer
  streaming step with **flat tensor I/O**. Mel features in, encoder frames
  out, and every piece of streaming state explicit: 56 left-context attention
  K/V frames per layer (24 layers), subsampling Conv2d left-context, and
  conformer depthwise-conv left-context. First-chunk and steady-state
  variants (the first chunk prepends NeMo's `init_pad` zero frame). The
  language prompt (128-dim one-hot over `prompt_ids`, fused by
  `prompt_projector`) is baked into the graph, so one encoder serves all
  languages and `auto` (index 101).
- **Decoder** (`decoder.onnx`): RNNT prediction network (embedding + 2-layer
  LSTM) as a single-token step with LSTM state in/out. State is committed
  only when a non-blank token is consumed (HF masked-cache semantics).
- **Joiner** (`joiner.onnx`): `logits = Linear(relu(enc + dec))`.
- **Engine** (`engine/nemotron_onnx_streaming.py`): numpy log-mel extractor
  bit-compatible with HF's (max diff 3.7e-9 vs librosa), chunk scheduling
  matching the HF processor (`1+8*r` mel frames first, `8*(r+1)` steady),
  RNNT greedy decode with `max_symbols_per_step=10`, language-tag handling,
  partial/final results. About 600 lines.

## Validation

6 real meeting-audio files (3 to 60 s, `language=auto`), jiwer,
torch 2.13 CPU + onnxruntime 1.27, opset 17:

| Precision | WER vs HF reference | RTF (CPU) | Size (320 ms) |
| --- | --- | --- | --- |
| **fp16 (shipped)** | **0.0137** | 0.315 | ~2.5 GB |
| fp32 | 0.0082 | 0.263 | ~4.7 GB |
| int8 (dynamic) | 0.189 | 0.148 | ~0.7 GB |

| Chunk | Latency | RTF (fp32, CPU) |
| --- | --- | --- |
| 80 ms | ~0.08 s | 0.653 |
| 320 ms | ~0.32 s | 0.263 |
| 1120 ms | ~1.12 s | 0.106 |

Graph-level parity vs the HF cache-aware streaming path stays below 1.5e-05
over first + 5 steady chunks. Text-level parity is near-verbatim at 320 ms
and 1120 ms (cross-checked against HF streaming at the same lookahead).
"Parity" here means fidelity of the conversion. For absolute ASR accuracy,
see NVIDIA's FLEURS tables on the base model card.

## The making of: the bug that only appeared after the first second

The most instructive part of this export was a bug that produced **perfect
transcriptions for the first chunk and garbage afterwards**. Exactly the
kind of bug that survives naive testing, because smoke tests pass.

The exported encoder threads 96 cache tensors (K + V across 24 layers, plus
conv caches) through the ONNX graph as positional inputs/outputs. The first
version built the cache name list **interleaved** (`k0, v0, k1, v1, ...`)
while the graph body unpacked them **blocked** (`caches[0:24]` for K,
`caches[24:48]` for V). Every layer from 1 up received another layer's K as
its V.

**Why chunk 0 was flawless:** all caches start as zeros, and any permutation
of zeros is still zeros. The bug was invisible until the first cache
carry-over, and it poisoned the decoder state permanently. A 60-second file
produced just *"Actually, that's a lot."*

**Why the obvious suspect was innocent:** before touching anything, we tested
the Transformer-XL relative-position theory numerically (warm-up
`cached_frames` growing 4, 8, ... vs a fixed window of 56). The warm-up
encoding turned out to be an exact centered slice of the full one, diff 0.0.
That clean counter-example saved hours of fixing the wrong thing. The real
culprit surfaced by localizing the first divergent layer, recording the
actual attention inputs, and finding V off by 430 while K matched to 3e-5.

The fix was one loop (`_cache_shapes()` in `export/export_onnx.py`). The
debugging probes live in `tools/` (layer-localized parity, cache sweep,
attention replay, input recorder), kept for the next divergence.

## Repo layout

```
export/export_onnx.py      HF checkpoint -> ONNX (encoder first/steady, decoder, joiner) + --validate
export/quantize.py         int8 dynamic quant + custom fp16 cast (>2 GiB-safe)
engine/nemotron_onnx_streaming.py   numpy+onnxruntime streaming engine / CLI
validation/parity.py       WER-parity harness (HF reference vs ONNX, per-file + aggregate)
validation/make_reference.py        generate HF reference transcripts
validation/hf_streaming_reference.py  HF streaming reference at a given lookahead
tools/                     debugging probes from the making-of
outputs/                   (git-ignored) exported graphs
```

## Related work

[onnx-community's int4 build](https://huggingface.co/onnx-community/nemotron-3.5-asr-streaming-0.6b-onnx-int4)
targets the smallest footprint with onnxruntime-genai at a fixed 560 ms
chunk. This repo covers the complementary case: the model's original
accuracy preserved (fp16) on plain onnxruntime, plus the tooling to
reproduce and verify everything.

## Roadmap

- [ ] int8 static quantization with calibration (dynamic int8 measurably degrades WER)
- [ ] Deduplicate encoder weights across chunk sizes (one shared external-data store)
- [ ] sherpa-onnx C++ port ([sherpa-onnx#3573](https://github.com/k2-fsa/sherpa-onnx/issues/3573)). The engine is meant as the reference implementation
- [ ] Word-level timestamps, batched streaming, CUDA-EP tuning

## Credits and license

Export, engine, fp16/int8 conversion and validation by
[@codavidgarcia](https://huggingface.co/codavidgarcia). If this saved you
time, a star helps others find it.

- **Code**: Apache-2.0 (`LICENSE`).
- **Weights**: NVIDIA, [OpenMDW-1.1](https://openmdw.ai/license/1-1/). Not
  affiliated with or endorsed by NVIDIA. See the model repo for the
  redistributable package.
