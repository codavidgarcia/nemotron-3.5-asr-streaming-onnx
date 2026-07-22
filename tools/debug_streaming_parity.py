#!/usr/bin/env python3
"""Debug: localize the steady-chunk divergence between HF cache-aware streaming
and the export wrapper (first + steady), tensor by tensor.

Run with the nemonv venv python. Prints max-abs diffs for: pooler output,
pre-prompt encoder hidden, attention K/V caches after each chunk, and the
relative-position score matrix (matrix_bd) computed both with HF's growing
`cached_frames` and the wrapper's fixed full window.
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
    sub = enc_cfg.subsampling_factor
    n_mels = enc_cfg.num_mel_bins
    C = lookahead + 1

    first = torch.randn(1, 1 + sub * lookahead, n_mels)
    steady = torch.randn(1, sub * (lookahead + 1), n_mels)
    prompt = torch.tensor([model.config.default_prompt_id], dtype=torch.long)

    # ---------------- HF path ----------------
    hf_outs = []
    with torch.no_grad():
        out = model.get_audio_features(
            input_features=first, prompt_ids=prompt, num_lookahead_tokens=lookahead,
            use_cache=True, output_attention_mask=False,
        )
        hf_outs.append(out)
        kv, pc = out.past_key_values, out.padding_cache
        for _ in range(2):
            out = model.get_audio_features(
                input_features=steady, prompt_ids=prompt, num_lookahead_tokens=lookahead,
                use_cache=True, output_attention_mask=False,
                past_key_values=kv, padding_cache=pc,
            )
            hf_outs.append(out)
            kv, pc = out.past_key_values, out.padding_cache

    # ---------------- wrapper path ----------------
    cache_shapes, cache_names, _, output_names = _encoder_io(model, left, enc_cfg.num_hidden_layers)
    caches = {n: torch.zeros(*cache_shapes[n]) for n in cache_names}
    valid = 0
    captured = {}

    def run_wrapper(wrapper, feats, tag):
        nonlocal valid
        mask = torch.zeros(1, 1, 1, left + C)
        mask[..., : left - valid] = -1e9
        ordered = [caches[n] for n in cache_names]

        # capture pre-prompt hidden
        def hook(mod, args):
            captured[tag] = args[0][..., : enc_cfg.hidden_size].clone()

        h = wrapper.prompt_projector.register_forward_pre_hook(hook)
        with torch.no_grad():
            outs = wrapper(feats, prompt, mask, *ordered)
        h.remove()
        valid = min(left, valid + C)
        for name, value in zip(output_names[1:], outs[1:]):
            caches[name.replace("_out_", "_")] = value
        return outs[0]

    first_w = StreamingEncoderStep(model, left, enc_cfg.num_hidden_layers, first_chunk=True).eval()
    steady_w = StreamingEncoderStep(model, left, enc_cfg.num_hidden_layers, first_chunk=False).eval()
    wr_pool = [run_wrapper(first_w, first, "chunk0")]
    for i in range(2):
        wr_pool.append(run_wrapper(steady_w, steady, f"chunk{i+1}"))

    for i, (hf, wp) in enumerate(zip(hf_outs, wr_pool)):
        print(f"chunk {i}: pooler diff = {(hf.pooler_output - wp).abs().max().item():.3e}   "
              f"hidden diff = {(hf.last_hidden_state - captured[f'chunk{i}']).abs().max().item():.3e}")

    # ---------------- attention cache compare after chunk 1 (HF layer 0) ----
    layer0 = kv.layers[0]
    hf_k = layer0.keys  # (1, heads, v, 128) real frames only
    my_k = caches["k_cache_0"][0, :, -hf_k.shape[2]:, :]
    print(f"layer0 K cache after 3 chunks: hf len {hf_k.shape[2]}, "
          f"diff (right-aligned) = {(hf_k[0] - my_k).abs().max().item():.3e}")

    # ---------------- matrix_bd: growing-L (HF) vs fixed-L (wrapper) --------
    attn = model.encoder.layers[0].self_attn
    pos_mod = model.encoder.encode_positions
    v, chunk = 4, C  # warm-up state at chunk 1
    hstate = torch.randn(1, chunk, enc_cfg.hidden_size)
    with torch.no_grad():
        pos_small = pos_mod(hstate, cached_frames=v)        # HF warm-up: L = v + C
        pos_full = pos_mod(hstate, cached_frames=left)      # wrapper: L = 56 + C

    # centered-slice equivalence claim
    off = left - v
    sliced = pos_full[:, off : off + 2 * (v + chunk) - 1]
    print(f"pos_embed centered-slice diff: {(pos_small - sliced).abs().max().item():.3e}")

    def matrix_bd(pos, total_kv):
        rel_k = attn.relative_k_proj(pos).view(1, -1, attn.config.num_attention_heads, attn.head_dim)
        q = torch.randn(1, chunk, attn.config.num_attention_heads, attn.head_dim).transpose(1, 2)
        m = q @ rel_k.permute(0, 2, 3, 1)
        m = torch.nn.functional.pad(m, (1, 0))
        m = m.view(1, 8, -1, chunk)
        m = m[:, :, 1:].view(1, 8, chunk, -1)
        return m[..., :total_kv]

    torch.manual_seed(1)
    q = torch.randn(1, 8, chunk, attn.head_dim)

    def matrix_bd_q(pos, total_kv):
        rel_k = attn.relative_k_proj(pos).view(1, -1, 8, attn.head_dim)
        m = q @ rel_k.permute(0, 2, 3, 1)
        m = torch.nn.functional.pad(m, (1, 0))
        m = m.view(1, 8, -1, chunk)
        m = m[:, :, 1:].view(1, 8, chunk, -1)
        return m[..., :total_kv]

    with torch.no_grad():
        bd_small = matrix_bd_q(pos_small, v + chunk)      # (1,8,C,v+C)
        bd_full = matrix_bd_q(pos_full, left + chunk)     # (1,8,C,60)
    # right-aligned slot comparison
    diff = (bd_small - bd_full[..., off:]).abs().max().item()
    print(f"matrix_bd small-L vs full-L (right-aligned) diff: {diff:.3e}")


if __name__ == "__main__":
    main()
