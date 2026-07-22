#!/usr/bin/env python3
"""Debug 2: capture HF layer-0 attention I/O at chunk 1 (steady, warm-up) and
replay it through the export wrapper's attention to find the exact divergence.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "export"))

from export_onnx import StreamingEncoderStep, CHUNK_MS_TO_LOOKAHEAD, MODEL_ID  # noqa: E402

CHUNK_MS = 320


def main():
    from transformers import AutoModelForRNNT

    torch.manual_seed(0)
    model = AutoModelForRNNT.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    enc_cfg = model.config.encoder_config
    lookahead = CHUNK_MS_TO_LOOKAHEAD[CHUNK_MS]
    left = enc_cfg.sliding_window - 1
    sub = enc_cfg.subsampling_factor
    n_mels = enc_cfg.num_mel_bins
    C = lookahead + 1

    first = torch.randn(1, 1 + sub * lookahead, n_mels)
    steady = torch.randn(1, sub * (lookahead + 1), n_mels)
    prompt = torch.tensor([model.config.default_prompt_id], dtype=torch.long)

    captured = {}

    def attn_hook(mod, args, kwargs, output):
        captured["attn_input"] = kwargs["hidden_states"].clone()
        captured["pos_embed"] = kwargs["position_embeddings"].clone()
        captured["attention_mask"] = (
            kwargs["attention_mask"].clone() if kwargs["attention_mask"] is not None else None
        )
        captured["attn_output"] = output[0].clone()

    h = model.encoder.layers[0].self_attn.register_forward_hook(attn_hook, with_kwargs=True)

    with torch.no_grad():
        out = model.get_audio_features(
            input_features=first, prompt_ids=prompt, num_lookahead_tokens=lookahead,
            use_cache=True, output_attention_mask=False,
        )
        kv, pc = out.past_key_values, out.padding_cache
        # snapshot layer-0 K/V BEFORE chunk 1
        k_prev = kv.layers[0].keys.clone()
        v_prev = kv.layers[0].values.clone()
        # also snapshot encoder position_embeddings for chunk 1 via hook
        out1 = model.get_audio_features(
            input_features=steady, prompt_ids=prompt, num_lookahead_tokens=lookahead,
            use_cache=True, output_attention_mask=False,
            past_key_values=kv, padding_cache=pc,
        )
    h.remove()

    print("captured attn input:", captured["attn_input"].shape)
    print("captured pos_embed:", captured["pos_embed"].shape)
    am = captured["attention_mask"]
    print("attention_mask:", None if am is None else (am.shape, am.dtype))
    if am is not None:
        if am.dtype == torch.bool:
            print("  mask: any False (masked out)?", bool((~am).any()))
        else:
            print("  mask min/max:", am.min().item(), am.max().item())
            print("  num -inf:", bool((am == float("-inf")).any()))

    # replay through wrapper attention
    wrapper = StreamingEncoderStep(model, left, enc_cfg.num_hidden_layers, first_chunk=False).eval()
    v = k_prev.shape[2]
    k_cache = torch.zeros(1, 8, left, 128)
    v_cache = torch.zeros(1, 8, left, 128)
    k_cache[0, :, -v:] = k_prev[0]
    v_cache[0, :, -v:] = v_prev[0]
    mask = torch.zeros(1, 1, 1, left + C)
    mask[..., : left - v] = -1e9
    with torch.no_grad():
        out_w, _, _ = wrapper._attention(
            model.encoder.layers[0].self_attn, captured["attn_input"],
            captured["pos_embed"], k_cache, v_cache, mask,
        )
    print(f"attn output diff (wrapper vs HF, same pos_embed): "
          f"{(out_w - captured['attn_output']).abs().max().item():.3e}")

    # and with wrapper's own fixed-window pos_embed
    with torch.no_grad():
        pos_full = model.encoder.encode_positions(captured["attn_input"], cached_frames=left)
        out_w2, _, _ = wrapper._attention(
            model.encoder.layers[0].self_attn, captured["attn_input"],
            pos_full, k_cache, v_cache, mask,
        )
    print(f"attn output diff (wrapper fixed-window pos):   "
          f"{(out_w2 - captured['attn_output']).abs().max().item():.3e}")

    # what pos_embed length did HF actually use?
    print(f"HF pos_embed length: {captured['pos_embed'].shape[1]} "
          f"(2L-1 => L={(captured['pos_embed'].shape[1] + 1) // 2}, "
          f"=> cached_frames={(captured['pos_embed'].shape[1] + 1) // 2 - C})")


if __name__ == "__main__":
    main()
