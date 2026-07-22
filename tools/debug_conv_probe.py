#!/usr/bin/env python3
"""Debug 5: direct I/O comparison of layer-0 submodules at chunk 1, HF vs wrapper,
each in its own real run (no replay). Spy points: attn input, attn output,
conv input (pre-norm_conv), conv output.
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

    hf_cap = {}
    layer0 = model.encoder.layers[0]

    def spy_out(cap, key):
        def hook(mod, args, kwargs, output):
            cap[key] = (output[0] if isinstance(output, tuple) else output).clone()
        return hook

    def reg(mod, cap, key):
        return mod.register_forward_hook(spy_out(cap, key), with_kwargs=True)

    with torch.no_grad():
        out = model.get_audio_features(
            input_features=first, prompt_ids=prompt, num_lookahead_tokens=lookahead,
            use_cache=True, output_attention_mask=False,
        )
        kv, pc = out.past_key_values, out.padding_cache
        hooks = [
            reg(layer0.self_attn, hf_cap, "attn_out"),
            layer0.self_attn.register_forward_pre_hook(
                lambda m, a, k: hf_cap.__setitem__("attn_in", k["hidden_states"].clone()),
                with_kwargs=True),
            layer0.conv.register_forward_pre_hook(
                lambda m, a, k: hf_cap.__setitem__(
                    "conv_in", (k["hidden_states"] if "hidden_states" in k else a[0]).clone()),
                with_kwargs=True),
            reg(layer0.conv, hf_cap, "conv_out"),
        ]
        model.get_audio_features(
            input_features=steady, prompt_ids=prompt, num_lookahead_tokens=lookahead,
            use_cache=True, output_attention_mask=False,
            past_key_values=kv, padding_cache=pc,
        )
        for h in hooks:
            h.remove()

    # ---- wrapper run with spies on the same points
    cache_shapes, cache_names, _, output_names = _encoder_io(model, left, n_layers)
    caches = {n: torch.zeros(*cache_shapes[n]) for n in cache_names}
    valid = 0
    wr_cap = {}

    steady_w = StreamingEncoderStep(model, left, n_layers, first_chunk=False).eval()
    first_w = StreamingEncoderStep(model, left, n_layers, first_chunk=True).eval()
    wlayer0 = steady_w.layers[0]

    def wrap_linear(mod, key, capture_input=False):
        orig = mod.forward
        def spy(x):
            if capture_input:
                wr_cap[key] = x.clone()
            out = orig(x)
            if not capture_input:
                wr_cap[key] = out.clone()
            return out
        mod.forward = spy
        return orig

    restores = [
        (wlayer0.self_attn.q_proj, wrap_linear(wlayer0.self_attn.q_proj, "attn_in", True)),
        (wlayer0.self_attn.o_proj, wrap_linear(wlayer0.self_attn.o_proj, "attn_out")),
        (wlayer0.norm_conv, wrap_linear(wlayer0.norm_conv, "conv_in", True)),
        (wlayer0.conv.pointwise_conv2, wrap_linear(wlayer0.conv.pointwise_conv2, "conv_pw2_out")),
    ]

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
    run_wrapper(steady_w, steady)
    for mod, orig in restores:
        mod.forward = orig

    # wrapper conv_out = pointwise_conv2 output transposed back
    wr_conv_out = wr_cap["conv_pw2_out"].transpose(1, 2)

    print(f"layer0 attn input diff:  {(hf_cap['attn_in'] - wr_cap['attn_in']).abs().max().item():.3e}")
    print(f"layer0 attn output diff: {(hf_cap['attn_out'] - wr_cap['attn_out']).abs().max().item():.3e}")
    print(f"layer0 conv input diff:  {(hf_cap['conv_in'] - wr_cap['conv_in']).abs().max().item():.3e}")
    print(f"layer0 conv output diff: {(hf_cap['conv_out'] - wr_conv_out).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
