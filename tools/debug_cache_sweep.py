#!/usr/bin/env python3
"""Debug 4: sweep ALL streaming caches (attention K/V, subsampling conv2d,
conformer conv1d) after chunk 0 and after chunk 1, HF vs wrapper.
The first diverging cache in pipeline order localizes the bug.
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

    hf_pc, hf_kv = {}, {}
    with torch.no_grad():
        out = model.get_audio_features(
            input_features=first, prompt_ids=prompt, num_lookahead_tokens=lookahead,
            use_cache=True, output_attention_mask=False,
        )
        hf_pc[0] = {k: l.cache.clone() for k, l in out.padding_cache.layers.items()}
        hf_kv[0] = [out.past_key_values.layers[i].keys.clone() for i in range(n_layers)]
        out = model.get_audio_features(
            input_features=steady, prompt_ids=prompt, num_lookahead_tokens=lookahead,
            use_cache=True, output_attention_mask=False,
            past_key_values=out.past_key_values, padding_cache=out.padding_cache,
        )
        hf_pc[1] = {k: l.cache.clone() for k, l in out.padding_cache.layers.items()}
        hf_kv[1] = [out.past_key_values.layers[i].keys.clone() for i in range(n_layers)]

    cache_shapes, cache_names, _, output_names = _encoder_io(model, left, n_layers)
    caches = {n: torch.zeros(*cache_shapes[n]) for n in cache_names}
    valid = 0
    my_snap = {}

    def run_wrapper(wrapper, feats, tag):
        nonlocal valid
        mask = torch.zeros(1, 1, 1, left + C)
        mask[..., : left - valid] = -1e9
        ordered = [caches[n] for n in cache_names]
        with torch.no_grad():
            outs = wrapper(feats, prompt, mask, *ordered)
        valid = min(left, valid + C)
        for name, value in zip(output_names[1:], outs[1:]):
            caches[name.replace("_out_", "_")] = value
        my_snap[tag] = {n: v.clone() for n, v in caches.items()}

    first_w = StreamingEncoderStep(model, left, n_layers, first_chunk=True).eval()
    steady_w = StreamingEncoderStep(model, left, n_layers, first_chunk=False).eval()
    run_wrapper(first_w, first, 0)
    run_wrapper(steady_w, steady, 1)

    def cmp(hf, my, name):
        # right-align: HF caches hold only real frames; mine are full-width
        t = hf.shape[2] if hf.dim() == 4 else hf.shape[-1]
        if hf.dim() == 4:  # conv2d (B, C, T, F)
            my_slice = my[:, :, -t:, :]
        elif hf.dim() == 3 and hf.shape[0] == 1 and hf.shape[1] == 8:  # attn K (B, H, T, D)
            my_slice = my[:, :, -t:, :]
        else:  # conv1d (B, C, T)
            my_slice = my[..., -t:]
        d = (hf - my_slice).abs().max().item()
        print(f"    {name:16s} hf{tuple(hf.shape)} my{tuple(my.shape)} diff {d:.3e}")

    for chunk in (0, 1):
        print(f"== caches after chunk {chunk}")
        for i in range(3):
            cmp(hf_pc[chunk][f"subsampling.{i}"], my_snap[chunk][f"conv2d_cache_{i}"], f"subsampling.{i}")
        for i in range(n_layers):
            cmp(hf_pc[chunk][f"conv.{i}"], my_snap[chunk][f"conv1d_cache_{i}"], f"conv.{i}")
        for i in range(n_layers):
            cmp(hf_kv[chunk][i], my_snap[chunk][f"k_cache_{i}"], f"attn_k.{i}")


if __name__ == "__main__":
    main()
