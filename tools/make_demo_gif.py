"""Record a real streaming run and render it as a terminal-style GIF.

Step 1 runs NemotronOnnxStreaming on a sample WAV, feeding 320 ms blocks and
capturing every partial-hypothesis change with its timestamp. Step 2 replays
those exact events into terminal-styled frames (Pillow) and writes a GIF.

Usage:
    python tools/make_demo_gif.py <model_dir> <wav> <out.gif> [--speed 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

import numpy as np
import soundfile as sf


def record_events(model_dir: str, wav: str) -> tuple[list[tuple[float, str]], float, str]:
    from nemotron_onnx_streaming import NemotronOnnxStreaming

    engine = NemotronOnnxStreaming(model_dir, language="auto", chunk_ms=320, precision="fp16")
    pcm, sr = sf.read(wav, dtype="float32")
    block = int(0.320 * sr)

    events: list[tuple[float, str]] = []
    t0 = time.perf_counter()
    last = ""
    for i in range(0, len(pcm), block):
        engine.accept_waveform(pcm[i : i + block])
        text = engine.get_partial()
        if text != last:
            events.append((time.perf_counter() - t0, text))
            last = text
    final = engine.get_final()
    if final != last:
        events.append((time.perf_counter() - t0, final))
    lang = engine.detected_language or ""
    return events, engine.rtf, lang


def render_gif(events, rtf, lang, out_path: Path, speed: float, fps: int = 10, args_wav: str = "demo.wav") -> None:
    from PIL import Image, ImageDraw, ImageFont

    W, H = 980, 560
    BG, FG, DIM, ACCENT = (30, 30, 30), (235, 235, 235), (130, 130, 130), (80, 220, 120)
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 19)
    small = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 16)

    total_t = events[-1][0] if events else 1.0
    n_frames = max(2, int(total_t / speed * fps))

    header = f"$ nemotron_onnx_streaming.py . {Path(args_wav).name} --language auto --precision fp16"
    status = f"ONNX fp16 | 320 ms chunks | CPU | RTF {rtf:.2f} | detected {lang}"

    def wrap(text: str, width: int = 88) -> list[str]:
        words, lines, cur = text.split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 > width:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        if cur:
            lines.append(cur)
        return lines

    frames = []
    for f in range(n_frames):
        t = f / fps * speed
        shown = ""
        for et, text in events:
            if et <= t:
                shown = text
            else:
                break
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, 46], fill=(45, 45, 45))
        d.ellipse([14, 16, 28, 30], fill=(255, 95, 86))
        d.ellipse([34, 16, 48, 30], fill=(255, 189, 46))
        d.ellipse([54, 16, 68, 30], fill=(39, 201, 63))
        d.text((84, 12), "nemotron-onnx, live streaming demo", font=small, fill=DIM)
        y = 62
        d.text((20, y), header, font=font, fill=ACCENT)
        y += 34
        for line in wrap(shown)[-12:]:
            d.text((20, y), line, font=font, fill=FG)
            y += 28
        if f % (fps // 2 or 1) < (fps // 4 or 1):
            d.rectangle([20, y + 2, 32, y + 24], fill=FG)  # cursor block
        d.text((20, H - 30), status, font=small, fill=DIM)
        frames.append(img)

    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=int(1000 / fps), loop=0, optimize=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("wav")
    ap.add_argument("out", type=Path)
    ap.add_argument("--speed", type=float, default=4.0, help="playback speed vs realtime")
    args = ap.parse_args()

    events, rtf, lang = record_events(args.model_dir, args.wav)
    Path(args.out).with_suffix(".events.json").write_text(
        json.dumps({"rtf": rtf, "lang": lang, "events": events}), encoding="utf-8"
    )
    render_gif(events, rtf, lang, args.out, args.speed, args_wav=args.wav)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {len(events)} events, RTF {rtf:.2f})")


if __name__ == "__main__":
    main()
