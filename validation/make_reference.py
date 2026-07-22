"""Generate transformers reference transcripts for the parity harness.

Runs the HF checkpoint (offline generate, language=auto) over every WAV in
validation/samples/ and writes validation/reference.json:
  { "<file>": {"text": ..., "text_with_tags": ...} }
"""
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForRNNT, AutoProcessor

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
SAMPLES = Path(__file__).parent / "samples"
OUT = Path(__file__).parent / "reference.json"


def main() -> None:
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForRNNT.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()

    results = {}
    for wav in sorted(SAMPLES.glob("*.wav")):
        audio, sr = sf.read(wav)
        assert sr == 16000, f"{wav}: expected 16kHz, got {sr}"
        inputs = proc(audio, sampling_rate=16000, language="auto", return_tensors="pt")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, return_dict_in_generate=True)
        dt = time.perf_counter() - t0
        results[wav.name] = {
            "text": proc.decode(out.sequences[0], skip_special_tokens=True),
            "text_with_tags": proc.decode(out.sequences[0], skip_special_tokens=False),
            "duration_s": round(len(audio) / sr, 2),
            "ref_inference_s": round(dt, 2),
        }
        print(f"{wav.name}: {dt:.1f}s -> {results[wav.name]['text'][:80]}")

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
