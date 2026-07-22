---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
library_name: onnx
pipeline_tag: automatic-speech-recognition
base_model: nvidia/nemotron-3.5-asr-streaming-0.6b
tags:
  - onnx
  - streaming-asr
  - automatic-speech-recognition
  - rnnt
  - fastconformer
  - multilingual
  - cache-aware ASR
  - unofficial
language:
  - en
  - es
  - de
  - fr
  - it
  - pt
  - nl
  - tr
  - ru
  - ar
  - hi
  - ja
  - ko
  - vi
  - uk
  - zh
---

# Nemotron 3.5 ASR Streaming 0.6B — ONNX (unofficial community export)

> **Unofficial community ONNX export** of NVIDIA's
> [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b).
> Not affiliated with or endorsed by NVIDIA. Weights are a derivative of
> NVIDIA's checkpoint and remain under the
> [OpenMDW-1.1 license](https://openmdw.ai/license/1-1/) — **attribution to
> NVIDIA is required** (see [License](#license)).

Multilingual (40 language-locales), cache-aware FastConformer-RNNT streaming
ASR, exported to ONNX for dependency-light inference with `onnxruntime`
(numpy + onnxruntime only at runtime). Supports the same runtime-selectable
chunk sizes as the base model: **80 / 160 / 320 / 560 / 1120 ms**, and the
same language-ID prompt conditioning (`target_lang` or `auto` with `<xx-XX>`
language-tag emission).

## Contents

| File | Description |
| --- | --- |
| `encoder_{80,160,320,560,1120}ms_first.onnx` | first-chunk streaming encoder step (prepends NeMo `init_pad` zero frame) |
| `encoder_{80,160,320,560,1120}ms.onnx` | steady-state streaming encoder step (caches in/out) |
| `decoder.onnx` | RNNT prediction network (embedding + 2×LSTM-640 + projector), single-token step |
| `joiner.onnx` | RNNT joint network, single encoder frame |
| `*_int8.onnx` | dynamic int8 variants (weights only; caches remain fp32) |
| `tokens.txt` | id → token-piece map (SentencePiece-style detokenization) |
| `nemotron_onnx_config.json` | blank/vocab ids, cache shapes, prompt dictionary, feature params |

Encoder I/O: mel features for one chunk + `prompt_ids` (language index) +
additive attention cache mask + streaming caches (per-layer attention K/V
`[1, 8, 56, 128]`, 3 subsampling Conv2d left-contexts, per-layer depthwise
Conv1d left-contexts `[1, 1024, 8]`) → RNNT-ready frames `[1, r+1, 640]` +
updated caches. Language one-hot fusion (`prompt_projector`) and the
`encoder_projector` are inside the graph.

## Usage

Reference engine (pure onnxruntime):
[`nemotron_onnx_streaming.py`](https://huggingface.co/<community-org>/nemotron-3.5-asr-streaming-0.6b-onnx)
— see the linked code repo for `NemotronOnnxStreaming` (chunk scheduling,
RNNT greedy decode, `auto` language tags, partial/final results).

```python
from nemotron_onnx_streaming import NemotronOnnxStreaming

engine = NemotronOnnxStreaming(".", language="auto", chunk_ms=320)
text = engine.transcribe_file("sample.wav")
```

## Intended use

Low-latency streaming transcription (voice agents, live captioning) and
offline transcription across the 40 supported locales, in environments where
a full PyTorch/NeMo stack is undesirable (edge, mobile, serverless, C#
/C++/Rust via onnxruntime bindings). For English-only use cases, NVIDIA
recommends the smaller-latency-tuned English sister model; this export tracks
the multilingual checkpoint only.

## Validation results

WER parity vs the 🤗 Transformers reference (offline `generate`), measured
with `validation/parity.py` (jiwer, lowercase + punctuation-stripped):

| Precision | WER vs HF reference | RTF (CPU) | Package size (320 ms) | Notes |
| --- | --- | --- | --- | --- |
| **fp16 (shipped)** | **micro 0.0137 / macro 0.0108** | 0.315 | **~2.5 GB** | near-parity with fp32; the distributable build |
| fp32 | micro 0.0082 / macro 0.0034 | 0.263 | ~4.7 GB | reference export |
| int8 (dynamic, MatMul/Gemm) | micro 0.189 | 0.148 | ~0.7 GB | 1.8× faster than fp32 on CPU but measurable WER cost — not the shipped default |

Chunk-size sweep (fp32, graph-level parity ~1e-6 vs HF streaming on all):

| Chunk size | Latency | RTF (fp32, CPU) | Notes |
| --- | --- | --- | --- |
| 80 ms | ~0.08 s | 0.653 | fluent output, slightly noisier decoding (least right context) |
| 320 ms | ~0.32 s | 0.263 | sweet spot: near-verbatim parity with the transformers offline reference on all 6 files |
| 1120 ms | ~1.12 s | 0.106 | best decode quality; near-verbatim vs HF **streaming** at the same lookahead (cross-checked on 60 s real audio) |

Parity was measured as WER between this ONNX export (streaming, chunked)
and the transformers reference on the same files — i.e. fidelity of the
export, not absolute ASR quality. Numbers: 6 real meeting-audio files
(3–60 s, auto language), torch 2.13 CPU + onnxruntime 1.27, opset 17.
WER values against the *offline* reference differ per chunk size because
each streaming configuration legitimately decodes slightly differently;
the meaningful check is parity against HF at the *same* lookahead
(verified: graph-level ~1e-6 on all sizes; text-level near-verbatim at
320 ms and 1120 ms).

_Base-model FLEURS results (NVIDIA): average LangID WER 10.38 / 10.00 / 9.49 /
9.12 / 8.84 across 80 ms–1.12 s chunks on the 19 transcription-ready locales._

## Export provenance

- Base checkpoint revision: `f3d333391852ba876df169dcc9ba902d25b6ab0b`
- Exported with `transformers>=5.13.0` (`Nemotron3_5AsrForRNNT`),
  `torch.onnx.export` (dynamo=False, opset 17)
- Graph numerics verified against the HF cache-aware streaming path
  (per-chunk pooler max-abs-diff < 1e-3 at export time)

## License

[OpenMDW-1.1](https://openmdw.ai/license/1-1/). The model weights (including
these ONNX derivatives) are licensed by NVIDIA under OpenMDW-1.1, which
requires attribution: **"NVIDIA Nemotron 3.5 ASR"** with a link to the
[base model](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b).
The export/conversion tooling code is Apache-2.0 (see the code repository).
