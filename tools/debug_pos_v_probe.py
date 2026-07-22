#!/usr/bin/env python3
"""Debug 6: verify the last two unverified attention inputs in the wrapper's
real chunk-1 run: the V cache after chunk 0, and the pos_embed content
(spy on relative_k_proj input).
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "export"))

from export_onnx import StreamingEncoderStep, CHUNK_MS_TO_LOOKAHEAD, MODEL_ID, _encoder_io  # noqa: E402

CHUNK_MS = 320


def main():
    from transformers import AutoModelForRNNT

    torch.manual_seed(0)
    model = AutoModelForRNNT.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    enc_cfg = model.config.encoder_config
    lookahead = CHUNK_MS_TO_LOOKAHEAD[CHUNK_MS]
    left = enc_cfg.sliding_window - 1
    n_layers = enc_cfg.num_hidden_layers
    sub = enc_cfg.subsampling_factor
    n_mels = enc_cfg.num_mel_bins
    C = lookahead + 1

    first = torch.randn(1, 1 + sub * lookahead, n_mels)
    steady = torch.randn(1, sub * (lookahead + 1), n_mels)
    prompt = torch.tensor([model.config.default_prompt_id], dtype=torch.long)

    with torch.no_grad():
        out = model.get_audio_features(
            input_features=first, prompt_ids=prompt, num_lookahead_tokens=lookahead,
            use_cache=True, output_attention_mask=False,
        )
        hf_v0 = out.past_key_values.layers[0].values.clone()
        hf_k0 = out.past_key_values.layers[0].keys.clone()

    cache_shapes, cache_names, _, output_names = _encoder_io(model, left, n_layers)
    caches = {n: torch.zeros(*cache_shapes[n]) for n in cache_names}
    valid = 0

    first_w = StreamingEncoderStep(model, left, n_layers, first_chunk=True).eval()
    steady_w = StreamingEncoderStep(model, left, n_layers, first_chunk=False).eval()

    def run_wrapper(wrapper, feats):
        nonlocal valid
        mask = torch.zeros(1, 1, 1, left + C)
        mask[..., : left - valid] = -1e9
        ordered = [caches[n] for n in cache_names]
        with torch.no_grad():
            outs = wrapper(feats, prompt, mask, *ordered)
        valid = min(left, valid + C)
        for name, value in zip(output_names[1:], outs[1:]):
            caches[name.replace("_out_", "_")] = value
        return outs[0]

    run_wrapper(first_w, first)
    v0 = hf_v0.shape[2]
    print(f"V cache after chunk 0 diff: {(hf_v0[0] - caches['v_cache_0'][0, :, -v0:]).abs().max().item():.3e}")
    print(f"K cache after chunk 0 diff: {(hf_k0[0] - caches['k_cache_0'][0, :, -v0:]).abs().max().item():.3e}")

    # spy pos_embed in the real chunk-1 run
    pos_real = {}
    rkp = steady_w.layers[0].self_attn.relative_k_proj
    orig = rkp.forward

    def spy(x):
        pos_real["x"] = x.clone()
        return orig(x)

    rkp.forward = spy
    run_wrapper(steady_w, steady)
    rkp.forward = orig

    with torch.no_grad():
        pos_expected = model.encoder.encode_positions(torch.zeros(1, C, enc_cfg.hidden_size),
                                                      cached_frames=left)
    print(f"pos_embed shape: real {tuple(pos_real['x'].shape)} expected {tuple(pos_expected.shape)}")
    print(f"pos_embed diff: {(pos_real['x'] - pos_expected).abs().max().item():.3e}")

    # full manual replay of real-run layer-0 attention with the REAL captured inputs
    k56 = caches["k_cache_0"]  # NOTE: already updated by chunk 1! rebuild from snapshots
    print("(note: caches dict now holds post-chunk-1 values)")


if __name__ == "__main__":
    main()
