#!/usr/bin/env python3
"""Debug 7: record ALL inputs that StreamingEncoderStep._attention actually
receives at chunk 1 layer 0 in the real wrapper run, and compare each against
the HF/replay tensors. Definitive.
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

    # ---- HF: capture chunk-1 layer-0 attention I/O
    hf = {}

    def hook(mod, args, kwargs, output):
        hf["attn_in"] = kwargs["hidden_states"].clone()
        hf["pos"] = kwargs["position_embeddings"].clone()
        hf["mask"] = kwargs["attention_mask"]
        hf["out"] = output[0].clone()

    with torch.no_grad():
        out = model.get_audio_features(
            input_features=first, prompt_ids=prompt, num_lookahead_tokens=lookahead,
            use_cache=True, output_attention_mask=False,
        )
        kv, pc = out.past_key_values, out.padding_cache
        hf_k0 = kv.layers[0].keys.clone()
        hf_v0 = kv.layers[0].values.clone()
        h = model.encoder.layers[0].self_attn.register_forward_hook(hook, with_kwargs=True)
        model.get_audio_features(
            input_features=steady, prompt_ids=prompt, num_lookahead_tokens=lookahead,
            use_cache=True, output_attention_mask=False,
            past_key_values=kv, padding_cache=pc,
        )
        h.remove()

    # ---- wrapper: record real _attention inputs at chunk 1
    cache_shapes, cache_names, _, output_names = _encoder_io(model, left, n_layers)
    caches = {n: torch.zeros(*cache_shapes[n]) for n in cache_names}
    valid = 0
    rec = {}

    first_w = StreamingEncoderStep(model, left, n_layers, first_chunk=True).eval()
    steady_w = StreamingEncoderStep(model, left, n_layers, first_chunk=False).eval()

    orig_attn = steady_w._attention
    call_count = {"n": 0}

    def spy_attn(attn, hidden, pos_embed, k_cache, v_cache, cache_mask):
        call_count["n"] += 1
        if call_count["n"] == 1:  # layer 0
            rec.update(hidden=hidden.clone(), pos=pos_embed.clone(),
                       k_cache=k_cache.clone(), v_cache=v_cache.clone(),
                       mask=cache_mask.clone())
        return orig_attn(attn, hidden, pos_embed, k_cache, v_cache, cache_mask)

    steady_w._attention = spy_attn

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
    wr_out1 = run_wrapper(steady_w, steady)
    steady_w._attention = orig_attn

    # ---- compare recorded real-run inputs vs HF/replay tensors
    print(f"attn input:      {(hf['attn_in'] - rec['hidden']).abs().max().item():.3e}")
    v0 = hf_k0.shape[2]
    print(f"k_cache (right-aligned): {(hf_k0[0] - rec['k_cache'][0, :, -v0:]).abs().max().item():.3e}")
    print(f"v_cache (right-aligned): {(hf_v0[0] - rec['v_cache'][0, :, -v0:]).abs().max().item():.3e}")
    print(f"k_cache invalid region (should be zeros): {rec['k_cache'][0, :, :-v0].abs().max().item():.3e}")
    print(f"pos_embed: HF len {hf['pos'].shape[1]} vs rec len {rec['pos'].shape[1]}")
    off = (rec["pos"].shape[1] - hf["pos"].shape[1]) // 2
    print(f"pos (centered slice):    {(hf['pos'] - rec['pos'][:, off:off + hf['pos'].shape[1]]).abs().max().item():.3e}")
    print(f"rec mask: min {rec['mask'].min().item():.1e} max {rec['mask'].max().item():.1e} "
          f"shape {tuple(rec['mask'].shape)}")
    print(f"HF mask: {hf['mask'].shape} dtype {hf['mask'].dtype} "
          f"min {hf['mask'].min().item()} max {hf['mask'].max().item()}")

    # ---- replay with the RECORDED inputs
    with torch.no_grad():
        out_replay, _, _ = orig_attn(
            model.encoder.layers[0].self_attn, rec["hidden"], rec["pos"],
            rec["k_cache"], rec["v_cache"], rec["mask"],
        )
    print(f"replay(recorded inputs) vs HF attn out: {(out_replay - hf['out']).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
