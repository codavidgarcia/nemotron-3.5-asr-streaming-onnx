#!/usr/bin/env python3
"""Debug 3: find the exact layer where the steady chunk diverges.

Runs HF and wrapper through chunks 0 and 1, then compares per-layer K caches
(written from each layer's attention *input*, so the first diverging layer
marks the layer whose predecessor produced a wrong output).
Also replays layer-0 attention with HF-captured tensors through wrapper math.
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

    # capture HF layer-0 attention I/O during chunk 1
    captured = {}

    def attn_hook(mod, args, kwargs, output):
        captured["attn_input"] = kwargs["hidden_states"].clone()
        captured["pos_embed"] = kwargs["position_embeddings"].clone()
        captured["attn_output"] = output[0].clone()

    # ---- HF chunks 0 and 1
    with torch.no_grad():
        out = model.get_audio_features(
            input_features=first, prompt_ids=prompt, num_lookahead_tokens=lookahead,
            use_cache=True, output_attention_mask=False,
        )
        kv, pc = out.past_key_values, out.padding_cache
        k_prev_l0 = kv.layers[0].keys.clone()
        v_prev_l0 = kv.layers[0].values.clone()
        h = model.encoder.layers[0].self_attn.register_forward_hook(attn_hook, with_kwargs=True)
        model.get_audio_features(
            input_features=steady, prompt_ids=prompt, num_lookahead_tokens=lookahead,
            use_cache=True, output_attention_mask=False,
            past_key_values=kv, padding_cache=pc,
        )
        h.remove()
    hf_k_after1 = [kv.layers[i].keys.clone() for i in range(n_layers)]

    # ---- wrapper chunks 0 and 1
    cache_shapes, cache_names, _, output_names = _encoder_io(model, left, n_layers)
    caches = {n: torch.zeros(*cache_shapes[n]) for n in cache_names}
    valid = 0

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

    first_w = StreamingEncoderStep(model, left, n_layers, first_chunk=True).eval()
    steady_w = StreamingEncoderStep(model, left, n_layers, first_chunk=False).eval()
    run_wrapper(first_w, first)
    run_wrapper(steady_w, steady)

    print("per-layer K-cache diff after chunk 1 (right-aligned real frames):")
    for i in range(n_layers):
        hf_k = hf_k_after1[i]  # (1, 8, v1+v2, 128)
        v = hf_k.shape[2]
        my_k = caches[f"k_cache_{i}"][0, :, -v:, :]
        d = (hf_k[0] - my_k).abs().max().item()
        flag = "  <-- first divergence" if d > 1e-4 and i > 0 and (
            hf_k_after1[i - 1][0] - caches[f"k_cache_{i-1}"][0, :, -hf_k_after1[i-1].shape[2]:, :]
        ).abs().max().item() <= 1e-4 else ""
        print(f"  layer {i:2d}: {d:.3e}{flag}")

    # ---- replay layer-0 attention with HF tensors through wrapper math
    wrapper = steady_w
    with torch.no_grad():
        out_w, _, _ = wrapper._attention(
            model.encoder.layers[0].self_attn, captured["attn_input"],
            captured["pos_embed"], k_prev_l0, v_prev_l0,
            torch.zeros(1, 1, 1, k_prev_l0.shape[2] + C),
        )
    print(f"layer0 attn replay (HF pos, HF kv): {(out_w - captured['attn_output']).abs().max().item():.3e}")

    with torch.no_grad():
        pos_full = model.encoder.encode_positions(captured["attn_input"], cached_frames=left)
        v0 = k_prev_l0.shape[2]
        k56 = torch.zeros(1, 8, left, 128); k56[0, :, -v0:] = k_prev_l0[0]
        v56 = torch.zeros(1, 8, left, 128); v56[0, :, -v0:] = v_prev_l0[0]
        mask = torch.zeros(1, 1, 1, left + C); mask[..., : left - v0] = -1e9
        out_w2, _, _ = wrapper._attention(
            model.encoder.layers[0].self_attn, captured["attn_input"],
            pos_full, k56, v56, mask,
        )
    print(f"layer0 attn replay (fixed-56 pos, padded kv): {(out_w2 - captured['attn_output']).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
