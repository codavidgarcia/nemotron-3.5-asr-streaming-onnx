#!/usr/bin/env python3
"""WER-parity validation: transformers reference vs the ONNX streaming engine.

For every 16 kHz mono WAV in a directory, runs
  1. the 🤗 Transformers reference (offline ``generate``), and
  2. the pure-onnxruntime streaming engine (``NemotronOnnxStreaming``),
then computes per-file and aggregate WER (jiwer) plus the ONNX engine RTF,
and writes ``report_<timestamp>.json`` next to this script.

The transformers import is guarded: without torch/transformers the script
still runs the ONNX-only side (hypotheses + RTF), and can score against
``<stem>.txt`` reference transcripts placed next to the WAVs (--ref txt).

Example:
    python parity.py --wav-dir ./test_wavs --model-dir ./onnx-out \
        --language auto --chunk-ms 320
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))


def _normalize(text: str, enabled: bool) -> str:
    """Lowercase + strip punctuation + collapse whitespace (configurable)."""
    if not enabled:
        return " ".join(text.split())
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    # also drop unicode punctuation jiwer-style normalizers would remove
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return " ".join(text.split())


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    pcm, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return pcm.mean(axis=1), sr


class TransformersReference:
    """Offline HF reference. Import is lazy so ONNX-only runs work without torch."""

    def __init__(self, model_id: str, language: str):
        from transformers import AutoModelForRNNT, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        try:
            self.model = AutoModelForRNNT.from_pretrained(model_id, device_map="auto")
        except Exception:
            self.model = AutoModelForRNNT.from_pretrained(model_id)
        self.model.eval()
        self.language = language

    def transcribe(self, pcm: np.ndarray, sampling_rate: int) -> str:
        import torch

        inputs = self.processor(
            pcm, sampling_rate=sampling_rate, language=self.language, return_tensors="pt"
        )
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        with torch.no_grad():
            output = self.model.generate(**inputs)
        sequences = getattr(output, "sequences", output)
        return self.processor.batch_decode(sequences, skip_special_tokens=True)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav-dir", required=True, type=Path, help="directory of 16 kHz mono WAVs")
    parser.add_argument("--model-dir", required=True, type=Path, help="exported ONNX model directory")
    parser.add_argument("--hf-model-id", default="nvidia/nemotron-3.5-asr-streaming-0.6b")
    parser.add_argument("--language", default="auto", help="locale / bare code / 'auto'")
    parser.add_argument("--chunk-ms", type=int, default=320, choices=[80, 160, 320, 560, 1120])
    parser.add_argument("--precision", default="fp32", choices=["fp32", "int8", "fp16"])
    parser.add_argument("--ref", default="transformers", choices=["transformers", "txt"],
                        help="'txt' reads <stem>.txt references instead of running HF")
    parser.add_argument("--no-normalize", action="store_true",
                        help="disable the lowercase/strip-punctuation WER normalizer")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()

    try:
        import jiwer
    except ImportError:
        jiwer = None
        print("[parity] jiwer not installed: WER will be skipped", file=sys.stderr)

    from nemotron_onnx_streaming import NemotronOnnxStreaming

    reference = None
    if args.ref == "transformers":
        try:
            reference = TransformersReference(args.hf_model_id, args.language)
        except ImportError:
            print("[parity] torch/transformers unavailable: running ONNX-only side "
                  "(use --ref txt with <stem>.txt transcripts to get WER)", file=sys.stderr)

    engine = NemotronOnnxStreaming(
        args.model_dir, language=args.language, chunk_ms=args.chunk_ms,
        num_threads=args.num_threads, precision=args.precision,
    )

    wavs = sorted(args.wav_dir.glob("*.wav"))[: args.limit]
    if not wavs:
        parser.error(f"no .wav files in {args.wav_dir}")

    normalize = not args.no_normalize
    files, refs, hyps = [], [], []
    for wav in wavs:
        pcm, sr = _load_wav(wav)
        duration = len(pcm) / sr

        engine.reset()
        t0 = time.perf_counter()
        block = 2 * 16000
        for i in range(0, len(pcm), block):
            engine.accept_waveform(pcm[i : i + block])
        hyp = engine.get_final()
        wall = time.perf_counter() - t0
        rtf = wall / duration if duration else 0.0

        ref = None
        if reference is not None:
            ref = reference.transcribe(pcm, sr)
        elif args.ref == "txt":
            txt = wav.with_suffix(".txt")
            if txt.exists():
                ref = txt.read_text(encoding="utf-8").strip()

        record = {
            "file": wav.name,
            "duration_s": round(duration, 3),
            "rtf": round(rtf, 4),
            "hypothesis": hyp,
            "reference": ref,
            "detected_language": engine.detected_language,
        }
        if ref is not None and jiwer is not None:
            r, h = _normalize(ref, normalize), _normalize(hyp, normalize)
            record["wer"] = round(jiwer.wer(r, h), 4)
            refs.append(r)
            hyps.append(h)
        files.append(record)
        print(f"[{wav.name}] rtf={rtf:.3f} wer={record.get('wer', 'n/a')}\n  hyp: {hyp}")

    aggregate = {
        "num_files": len(files),
        "mean_rtf": round(float(np.mean([f["rtf"] for f in files])), 4),
        "scored_files": len(refs),
    }
    if refs and jiwer is not None:
        aggregate["wer_micro"] = round(jiwer.wer(refs, hyps), 4)
        aggregate["wer_macro"] = round(float(np.mean([f["wer"] for f in files if "wer" in f])), 4)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "hf_model_id": args.hf_model_id,
            "model_dir": str(args.model_dir),
            "language": args.language,
            "chunk_ms": args.chunk_ms,
            "precision": args.precision,
            "reference": args.ref,
            "normalized": normalize,
        },
        "aggregate": aggregate,
        "files": files,
    }
    out_path = Path(__file__).parent / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[parity] aggregate: {json.dumps(aggregate)}")
    print(f"[parity] wrote {out_path}")


if __name__ == "__main__":
    main()
