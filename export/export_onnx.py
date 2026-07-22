#!/usr/bin/env python3
"""Export NVIDIA `nvidia/nemotron-3.5-asr-streaming-0.6b` to streaming ONNX graphs.

Produces these ONNX models per selected chunk size:

  1. ``encoder_{chunk_ms}ms_first.onnx`` / ``encoder_{chunk_ms}ms.onnx`` — one
     cache-aware FastConformer streaming step, first-chunk and steady-state
     variants (the first chunk prepends NeMo's ``init_pad`` zero frame at each
     subsampling conv). Inputs: mel features for one chunk, the language-prompt
     index, an additive attention cache mask, and all streaming caches
     (attention K/V per layer, subsampling Conv2d left-context, conformer
     depthwise Conv1d left-context). Outputs: RNNT-ready encoder frames
     (batch, chunk_frames, 640) and the updated caches. The graph already
     includes the language-ID prompt fusion (``prompt_projector``) and the
     ``encoder_projector`` so its output feeds the joiner directly.
  2. ``decoder.onnx`` — the RNNT prediction network (embedding + 2-layer LSTM
     + projector) for a single token step, with LSTM state in/out.
  3. ``joiner.onnx`` — the RNNT joint network for a single encoder frame:
     ``logits = head(relu(enc_frame + dec_out))``.

The exported encoder step re-implements the streaming forward of
``transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming``
with flat tensor I/O, because the HF modules thread caches through Python
cache objects (``DynamicCache`` / ``NemotronAsrStreamingEncoderCausalConvPaddingCache``)
that ``torch.onnx.export`` cannot trace. All math is reused from the loaded
submodules (weights are shared, not copied).

Language prompt choice: the HF model applies the language one-hot inside
``Nemotron3_5AsrForRNNT.get_audio_features`` (one-hot of ``prompt_ids`` over
``config.num_prompts=128`` slots, broadcast over time, concatenated to the
encoder hidden states and fused by ``prompt_projector``). We export the
encoder with ``prompt_ids`` as an explicit ``int64[1]`` graph input and keep
the one-hot + fusion inside the graph, so a single ONNX file serves all
languages and ``auto`` (index 101).

Verified against transformers ``main`` (model lands in v5.13.0), module
``transformers.models.nemotron3_5_asr`` / ``nemotron_asr_streaming``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Constants grounded in the HF checkpoint (config.json / processor_config.json
# of nvidia/nemotron-3.5-asr-streaming-0.6b, revision f3d33339).
# ---------------------------------------------------------------------------

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"

# chunk_ms -> right attention context (num_lookahead_tokens), 80ms encoder frames.
CHUNK_MS_TO_LOOKAHEAD = {80: 0, 160: 1, 320: 3, 560: 6, 1120: 13}

# Feature extraction (processor_config.json -> NemotronAsrStreamingFeatureExtractor).
FEATURE_PARAMS = {
    "sampling_rate": 16000,
    "n_fft": 512,
    "hop_length": 160,
    "win_length": 400,
    "n_mels": 128,
    "preemphasis": 0.97,
    "log_zero_guard": 2**-24,
    "mel_norm": "slaney",
    "mel_scale": "slaney",
    "fmin": 0.0,
    "fmax": 8000.0,
}

# Language prompt dictionary (processor_config.json / processing_nemotron3_5_asr.py).
# locale-or-code -> prompt index. "auto" = automatic language detection.
PROMPT_DICTIONARY = {
    "af-ZA": 54, "am-ET": 49, "ar": 7, "ar-AR": 7, "auto": 101, "ay-BO": 81,
    "az-AZ": 66, "bg": 30, "bg-BG": 30, "bn-IN": 36, "cs": 22, "cs-CZ": 22,
    "da": 25, "da-DK": 25, "de": 9, "de-DE": 9, "el": 21, "el-GR": 21,
    "en": 0, "en-GB": 1, "en-US": 0, "enGB": 1, "es": 3, "es-ES": 2,
    "es-US": 3, "esES": 2, "et": 60, "et-EE": 60, "fa-IR": 38, "fi": 26,
    "fi-FI": 26, "fr": 8, "fr-CA": 100, "fr-FR": 8, "gn-PY": 82, "gu-IN": 42,
    "ha-NG": 50, "haw-US": 97, "he-IL": 64, "hi": 6, "hi-HI": 6, "hi-IN": 6,
    "hr": 29, "hr-HR": 29, "hu": 23, "hu-HU": 23, "hy-AM": 68, "id-ID": 34,
    "ig-NG": 53, "it": 15, "it-IT": 15, "ja-JA": 10, "ja-JP": 10, "ka-GE": 67,
    "km-KH": 47, "kn-IN": 43, "ko": 14, "ko-KO": 14, "ko-KR": 14, "ku-TR": 65,
    "ky-KG": 71, "ln-CD": 58, "lt": 31, "lt-LT": 31, "lv": 61, "lv-LV": 61,
    "mi-NZ": 96, "ml-IN": 44, "mr-IN": 41, "ms-MY": 35, "mt-MT": 102,
    "nah-MX": 83, "nb": 103, "nb-NO": 103, "ne-NP": 46, "nl": 16, "nl-NL": 16,
    "nn": 104, "nn-NO": 104, "no": 27, "no-NO": 27, "ny-MW": 57, "or-KE": 59,
    "pl": 17, "pl-PL": 17, "pt": 13, "pt-BR": 12, "pt-PT": 13, "qu-PE": 80,
    "ro": 20, "ro-RO": 20, "ru": 11, "ru-RU": 11, "rw-RW": 55, "si-LK": 45,
    "sk": 28, "sk-SK": 28, "sl": 62, "sl-SI": 62, "sm-WS": 98, "so-SO": 56,
    "sv": 24, "sv-SE": 24, "sw-KE": 48, "ta-IN": 39, "te-IN": 40, "tg-TJ": 70,
    "th-TH": 32, "to-TO": 99, "tr": 18, "tr-TR": 18, "uk": 19, "uk-UA": 19,
    "ur-PK": 37, "uz-UZ": 69, "vi-VN": 33, "yo-NG": 52, "zh-CN": 4,
    "zh-TW": 5, "zh-ZH": 4, "zu-ZA": 51,
}


def _rel_shift(scores: torch.Tensor) -> torch.Tensor:
    """Transformer-XL relative shift, as in NemotronAsrStreamingEncoderAttention._rel_shift."""
    batch, heads, q_len, p_len = scores.shape
    scores = F.pad(scores, (1, 0))
    scores = scores.view(batch, heads, -1, q_len)
    return scores[:, :, 1:].view(batch, heads, q_len, p_len)


class StreamingEncoderStep(nn.Module):
    """One cache-aware streaming encoder step with flat tensor I/O.

    Mirrors, for fixed chunk shapes and no padding mask (the HF streaming path
    passes ``attention_mask=None``):

      * ``NemotronAsrStreamingEncoderSubsamplingConv2D`` with the causal Conv2d
        left-context caches threaded explicitly (steady-state ``left_pad`` =
        ``kernel - stride`` = 1 frame per conv; the first-chunk ``init_pad``
        zero frame is handled by the caller, see export notes below),
      * each ``NemotronAsrStreamingEncoderBlock`` (0.5-scaled FFs, relative
        position attention with sliding-window K/V cache, causal depthwise
        conv module),
      * language-prompt fusion + projection from
        ``Nemotron3_5AsrForRNNT.get_audio_features``.

    The attention K/V cache holds exactly ``left_context = sliding_window - 1``
    (56) frames per layer; each step concatenates the cached K/V with the new
    chunk's K/V and keeps the last 56 frames. Because the cache width equals
    the attention left context, the chunked-limited mask reduces to an
    additive mask over cache validity only (engine right-aligns real frames
    and passes ``-inf`` on not-yet-filled slots, reproducing HF's growing
    ``DynamicCache`` during the first chunks).
    """

    def __init__(self, model: nn.Module, left_context: int, num_layers: int, first_chunk: bool = False):
        super().__init__()
        self.first_chunk = first_chunk
        enc = model.encoder
        self.subsampling = enc.subsampling
        self.encode_positions = enc.encode_positions
        self.layers = enc.layers
        self.prompt_projector = model.prompt_projector
        self.encoder_projector = model.encoder_projector
        self.input_scale = enc.input_scale
        self.num_prompts = model.config.num_prompts
        self.left_context = left_context
        self.num_layers = num_layers
        enc_cfg = model.config.encoder_config
        self.num_heads = enc_cfg.num_attention_heads
        self.head_dim = enc_cfg.hidden_size // enc_cfg.num_attention_heads
        self.conv_left_pad = enc_cfg.conv_kernel_size - 1  # stride 1 depthwise

    # -- causal conv helpers (bypass the HF cache-object forward paths) ------

    @staticmethod
    def _conv2d_no_pad(module: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x, module.weight, module.bias,
            stride=module.stride, padding=0, dilation=module.dilation, groups=module.groups,
        )

    def _subsampling(self, input_features: torch.Tensor, conv2d_caches: list[torch.Tensor]):
        sub = self.subsampling
        kernel, stride = sub.conv_in.kernel_size[0], sub.conv_in.stride[0]
        freq_pad = (kernel - 1, stride - 1)  # NeMo CausalConv2D frequency padding

        new_caches = []
        x = input_features.unsqueeze(1)  # (B, 1, T, F)

        # stem conv. On the first chunk NeMo's CausalConv2D cache prepends an
        # extra `init_pad = (kernel-1) - left_pad` zero frame (see
        # NemotronAsrStreamingEncoderCausalConv2dCacheLayer.update); replicate
        # it with a constant zero frame so steady-state caches stay 1 wide.
        x = F.pad(x, freq_pad)  # pad last (freq) dim
        if self.first_chunk:
            x = torch.cat([torch.zeros_like(conv2d_caches[0]), conv2d_caches[0], x], dim=2)
        else:
            x = torch.cat([conv2d_caches[0], x], dim=2)  # prepend cached left time frames
        new_caches.append(x[:, :, -conv2d_caches[0].shape[2]:, :])
        x = sub.act_fn(self._conv2d_no_pad(sub.conv_in, x))

        # depthwise-separable stages
        for layer, cache in zip(sub.layers, conv2d_caches[1:]):
            x = F.pad(x, freq_pad)
            if self.first_chunk:
                x = torch.cat([torch.zeros_like(cache), cache, x], dim=2)
            else:
                x = torch.cat([cache, x], dim=2)
            new_caches.append(x[:, :, -cache.shape[2]:, :])
            x = self._conv2d_no_pad(layer.depthwise_conv, x)
            x = self._conv2d_no_pad(layer.pointwise_conv, x)
            x = sub.act_fn(x)

        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)
        return sub.linear(x), new_caches

    def _attention(
        self,
        attn: nn.Module,
        hidden: torch.Tensor,
        pos_embed: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_mask: torch.Tensor,
    ):
        batch, q_len, _ = hidden.shape
        shape = (batch, q_len, self.num_heads, self.head_dim)
        q = attn.q_proj(hidden).view(shape).transpose(1, 2)
        k = attn.k_proj(hidden).view(shape).transpose(1, 2)
        v = attn.v_proj(hidden).view(shape).transpose(1, 2)

        k_all = torch.cat([k_cache, k], dim=2)
        v_all = torch.cat([v_cache, v], dim=2)
        k_new = k_all[:, :, -self.left_context:, :]
        v_new = v_all[:, :, -self.left_context:, :]
        total_kv = k_all.shape[2]

        q_u = q + attn.bias_u.view(1, self.num_heads, 1, self.head_dim)
        q_v = q + attn.bias_v.view(1, self.num_heads, 1, self.head_dim)

        rel_k = attn.relative_k_proj(pos_embed).view(batch, -1, self.num_heads, self.head_dim)
        matrix_bd = q_v @ rel_k.permute(0, 2, 3, 1)
        matrix_bd = _rel_shift(matrix_bd)[..., :total_kv] * attn.scaling

        weights = (q_u @ k_all.transpose(2, 3)) * attn.scaling + matrix_bd
        weights = weights + cache_mask  # additive -inf on invalid cache slots
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(q.dtype)

        out = (weights @ v_all).transpose(1, 2).reshape(batch, q_len, -1)
        return attn.o_proj(out), k_new, v_new

    def forward(
        self,
        input_features: torch.Tensor,   # (1, T_mel, n_mels)
        prompt_ids: torch.Tensor,       # (1,) int64
        cache_mask: torch.Tensor,       # (1, 1, 1, left_context + T_enc) additive float
        *caches: torch.Tensor,
    ):
        n = self.num_layers
        k_caches = list(caches[0:n])
        v_caches = list(caches[n:2 * n])
        conv2d_caches = list(caches[2 * n:2 * n + 3])
        conv1d_caches = list(caches[2 * n + 3:3 * n + 3])

        hidden, new_conv2d = self._subsampling(input_features, conv2d_caches)
        hidden = hidden * self.input_scale

        pos_embed = self.encode_positions(hidden, cached_frames=self.left_context)

        new_k, new_v, new_conv1d = [], [], []
        for i, layer in enumerate(self.layers):
            hidden = hidden + 0.5 * layer.feed_forward1(layer.norm_feed_forward1(hidden))

            attn_out, k_new_i, v_new_i = self._attention(
                layer.self_attn, layer.norm_self_att(hidden), pos_embed,
                k_caches[i], v_caches[i], cache_mask,
            )
            new_k.append(k_new_i)
            new_v.append(v_new_i)
            hidden = hidden + attn_out

            # conformer convolution module (causal depthwise with left-context cache)
            conv = layer.conv
            h = conv.pointwise_conv1(layer.norm_conv(hidden).transpose(1, 2))
            h = F.glu(h, dim=1)
            h = torch.cat([conv1d_caches[i], h], dim=-1)
            new_conv1d.append(h[:, :, -self.conv_left_pad:])
            h = F.conv1d(
                h, conv.depthwise_conv.weight, conv.depthwise_conv.bias,
                stride=1, padding=0, dilation=1, groups=conv.depthwise_conv.groups,
            )
            h = conv.norm(h.transpose(1, 2)).transpose(1, 2)
            h = conv.activation(h)
            h = conv.pointwise_conv2(h)
            hidden = hidden + h.transpose(1, 2)

            hidden = hidden + 0.5 * layer.feed_forward2(layer.norm_feed_forward2(hidden))
            hidden = layer.norm_out(hidden)

        # language-ID prompt fusion (Nemotron3_5AsrForRNNT.get_audio_features)
        one_hot = F.one_hot(prompt_ids, num_classes=self.num_prompts).to(hidden.dtype)
        one_hot = one_hot[:, None, :].expand(-1, hidden.shape[1], -1)
        fused = self.prompt_projector(torch.cat([hidden, one_hot], dim=-1))
        enc_out = self.encoder_projector(fused)  # (1, T_enc, decoder_hidden)

        return (enc_out, *new_k, *new_v, *new_conv2d, *new_conv1d)


class DecoderStep(nn.Module):
    """RNNT prediction network, single token step, LSTM state in/out.

    Replicates ``Nemotron3_5AsrRNNTDecoder`` minus the Python blank fast-path:
    the graph always updates state; the calling engine only commits the new
    state when a non-blank token was consumed (matching the masked
    ``ParakeetRNNTDecoderCache.update`` semantics).
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.embedding = decoder.embedding
        self.lstm = decoder.lstm
        self.decoder_projector = decoder.decoder_projector

    def forward(self, token: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        # token: (1, 1) int64; h/c: (num_layers, 1, hidden)
        emb = self.embedding(token)
        out, (h_new, c_new) = self.lstm(emb, (h, c))
        dec_out = self.decoder_projector(out[:, -1])  # (1, hidden)
        return dec_out, h_new, c_new


class JoinerStep(nn.Module):
    """RNNT joint network for one encoder frame: logits = head(act(enc + dec))."""

    def __init__(self, joint: nn.Module):
        super().__init__()
        self.activation = joint.activation
        self.head = joint.head

    def forward(self, enc_frame: torch.Tensor, dec_out: torch.Tensor):
        # enc_frame: (1, joint_hidden); dec_out: (1, joint_hidden)
        return self.head(self.activation(enc_frame + dec_out))  # (1, vocab)


# ---------------------------------------------------------------------------
# Export driver
# ---------------------------------------------------------------------------


def _load_model(model_id: str, device: str):
    from transformers import AutoModelForRNNT, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForRNNT.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    model.to(device)
    return model, processor


def _cache_shapes(model, left_context: int) -> dict:
    """Static cache shapes for the exported graphs (batch = 1, fp32)."""
    enc_cfg = model.config.encoder_config
    n_mels = enc_cfg.num_mel_bins
    ch = enc_cfg.subsampling_conv_channels
    kernel, stride = enc_cfg.subsampling_conv_kernel_size, enc_cfg.subsampling_conv_stride
    freq_pad = (kernel - 1) + (stride - 1)

    shapes: dict[str, tuple] = {}
    heads = enc_cfg.num_attention_heads
    head_dim = enc_cfg.hidden_size // heads
    # Block order: all K caches, then all V caches — MUST match the positional
    # unpacking in StreamingEncoderStep.forward (caches[0:n] = K, caches[n:2n] = V)
    # and the output_names order in _encoder_io. (Interleaving k/v here misaligns
    # the named graph inputs with the wrapper's positional slots: chunk 0 works
    # because all caches are zero, every steady chunk reads the wrong tensors.)
    for i in range(enc_cfg.num_hidden_layers):
        shapes[f"k_cache_{i}"] = (1, heads, left_context, head_dim)
    for i in range(enc_cfg.num_hidden_layers):
        shapes[f"v_cache_{i}"] = (1, heads, left_context, head_dim)

    # subsampling Conv2d caches: post-freq-pad activations, left_pad = kernel - stride = 1 frame.
    freq = n_mels
    conv2d_shapes = []
    freq_in = freq + freq_pad
    conv2d_shapes.append((1, 1, kernel - stride, freq_in))           # stem, in_channels=1
    freq = (freq_in - kernel) // stride + 1
    freq_in = freq + freq_pad
    conv2d_shapes.append((1, ch, kernel - stride, freq_in))          # dw-sep stage 1
    freq = (freq_in - kernel) // stride + 1
    freq_in = freq + freq_pad
    conv2d_shapes.append((1, ch, kernel - stride, freq_in))          # dw-sep stage 2
    for i, s in enumerate(conv2d_shapes):
        shapes[f"conv2d_cache_{i}"] = s

    for i in range(enc_cfg.num_hidden_layers):
        shapes[f"conv1d_cache_{i}"] = (1, enc_cfg.hidden_size, enc_cfg.conv_kernel_size - 1)
    return shapes


def _encoder_io(model, left_context: int, n_layers: int):
    cache_shapes = _cache_shapes(model, left_context)
    cache_names = list(cache_shapes.keys())
    input_names = ["input_features", "prompt_ids", "cache_mask"] + cache_names
    output_names = (
        ["encoder_out"]
        + [f"k_cache_out_{i}" for i in range(n_layers)]
        + [f"v_cache_out_{i}" for i in range(n_layers)]
        + [f"conv2d_cache_out_{i}" for i in range(3)]
        + [f"conv1d_cache_out_{i}" for i in range(n_layers)]
    )
    return cache_shapes, cache_names, input_names, output_names


def _export_encoder_variant(
    model, out_dir: Path, chunk_ms: int, first_chunk: bool, opset: int, validate: bool, device: str,
):
    enc_cfg = model.config.encoder_config
    lookahead = CHUNK_MS_TO_LOOKAHEAD[chunk_ms]
    if lookahead not in enc_cfg.supported_num_lookahead_tokens:
        raise ValueError(
            f"num_lookahead_tokens={lookahead} not in supported set "
            f"{enc_cfg.supported_num_lookahead_tokens}"
        )
    left_context = enc_cfg.sliding_window - 1
    n_layers = enc_cfg.num_hidden_layers
    subsampling = enc_cfg.subsampling_factor
    n_mels = enc_cfg.num_mel_bins

    # first chunk: 1 + subsampling * lookahead mel frames; steady: subsampling * (lookahead + 1)
    mel_frames = (1 + subsampling * lookahead) if first_chunk else subsampling * (lookahead + 1)
    enc_frames = lookahead + 1

    wrapper = StreamingEncoderStep(
        model, left_context=left_context, num_layers=n_layers, first_chunk=first_chunk
    ).eval()
    wrapper.to(device)

    cache_shapes, cache_names, input_names, output_names = _encoder_io(model, left_context, n_layers)

    input_features = torch.randn(1, mel_frames, n_mels, device=device)
    prompt_ids = torch.tensor([model.config.default_prompt_id], dtype=torch.long, device=device)
    cache_mask = torch.zeros(1, 1, 1, left_context + enc_frames, device=device)
    if first_chunk:
        # nothing valid in the caches yet: mask out all cached attention slots
        cache_mask[..., :left_context] = float("-1e9")
    caches = [torch.zeros(*cache_shapes[name], device=device) for name in cache_names]

    suffix = "_first" if first_chunk else ""
    # Each encoder variant gets its own subdirectory: torch writes external
    # weights (>2GiB protobuf limit) with auto-generated file names that would
    # otherwise collide between variants sharing an output directory.
    stem = f"encoder_{chunk_ms}ms{suffix}"
    (out_dir / stem).mkdir(parents=True, exist_ok=True)
    out_path = out_dir / stem / f"{stem}.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_features, prompt_ids, cache_mask, *caches),
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            dynamo=False,
            do_constant_folding=True,
        )
    print(f"[encoder] wrote {out_path}")

    if validate:
        _check_onnx(out_path)
        _smoke_compare(wrapper, (input_features, prompt_ids, cache_mask, *caches),
                       str(out_path), input_names, "encoder_out")
    return out_path, wrapper


def _parity_check_hf(model, first_wrapper, steady_wrapper, chunk_ms: int, device: str,
                     num_steady: int = 5):
    """Compare the export wrappers against HF's own cache-aware streaming path.

    Runs first + ``num_steady`` streaming steps through both
    ``Nemotron3_5AsrForRNNT.get_audio_features`` (HF cache objects) and the
    export wrappers (flat tensors), printing max abs diffs. This pins down the
    cache-width / init-pad / relative-position details against ground truth.
    """
    enc_cfg = model.config.encoder_config
    lookahead = CHUNK_MS_TO_LOOKAHEAD[chunk_ms]
    left_context = enc_cfg.sliding_window - 1
    n_layers = enc_cfg.num_hidden_layers
    sub = enc_cfg.subsampling_factor
    n_mels = enc_cfg.num_mel_bins
    enc_frames = lookahead + 1

    first = torch.randn(1, 1 + sub * lookahead, n_mels, device=device)
    steady = torch.randn(1, sub * (lookahead + 1), n_mels, device=device)
    prompt_ids = torch.tensor([model.config.default_prompt_id], dtype=torch.long, device=device)

    # --- HF reference: streaming with HF cache objects ---
    hf_pooler = []
    with torch.no_grad():
        out = model.get_audio_features(
            input_features=first, prompt_ids=prompt_ids,
            num_lookahead_tokens=lookahead, use_cache=True, output_attention_mask=False,
        )
        hf_pooler.append(out.pooler_output)
        kv, conv_cache = out.past_key_values, out.padding_cache
        for _ in range(num_steady):
            out = model.get_audio_features(
                input_features=steady, prompt_ids=prompt_ids,
                num_lookahead_tokens=lookahead, use_cache=True, output_attention_mask=False,
                past_key_values=kv, padding_cache=conv_cache,
            )
            hf_pooler.append(out.pooler_output)
            kv, conv_cache = out.past_key_values, out.padding_cache

    # --- wrappers: flat caches, right-aligned with additive validity mask ---
    cache_shapes, cache_names, _, output_names = _encoder_io(model, left_context, n_layers)
    caches = {name: torch.zeros(*cache_shapes[name], device=device) for name in cache_names}
    valid = 0  # number of right-aligned valid attention cache frames

    def run_wrapper(wrapper, feats):
        nonlocal valid
        mask = torch.zeros(1, 1, 1, left_context + enc_frames, device=device)
        mask[..., : left_context - valid] = float("-1e9")
        ordered = [caches[name] for name in cache_names]
        with torch.no_grad():
            outs = wrapper(feats, prompt_ids, mask, *ordered)
        valid = min(left_context, valid + enc_frames)
        for name, value in zip(output_names[1:], outs[1:]):
            # graph outputs are named e.g. k_cache_out_3; map back to input names
            caches[name.replace("_out_", "_")] = value
        return outs[0]

    wr_pooler = [run_wrapper(first_wrapper, first)]
    for _ in range(num_steady):
        wr_pooler.append(run_wrapper(steady_wrapper, steady))
    diffs = []
    for step, (hf, wr) in enumerate(zip(hf_pooler, wr_pooler)):
        diffs.append((hf - wr).abs().max().item())
        print(f"[parity-hf] chunk {step} ({'first' if step == 0 else 'steady'}): "
              f"pooler max abs diff = {diffs[-1]:.3e}")
    if max(diffs) > 1e-3:
        print("[parity-hf] WARNING: wrapper diverges from HF streaming path beyond 1e-3",
              file=sys.stderr)


def export_encoder(model, out_dir: Path, chunk_ms: int, opset: int, validate: bool, device: str):
    _, first_wrapper = _export_encoder_variant(
        model, out_dir, chunk_ms, True, opset, validate, device
    )
    _, steady_wrapper = _export_encoder_variant(
        model, out_dir, chunk_ms, False, opset, validate, device
    )
    if validate:
        _parity_check_hf(model, first_wrapper, steady_wrapper, chunk_ms, device)


def export_decoder(model, out_dir: Path, opset: int, validate: bool, device: str):
    cfg = model.config
    wrapper = DecoderStep(model.decoder).eval().to(device)
    token = torch.full((1, 1), cfg.blank_token_id, dtype=torch.long, device=device)
    h = torch.zeros(cfg.num_decoder_layers, 1, cfg.decoder_hidden_size, device=device)
    c = torch.zeros_like(h)

    out_path = out_dir / "decoder.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (token, h, c), out_path,
            input_names=["token", "h_in", "c_in"],
            output_names=["decoder_out", "h_out", "c_out"],
            opset_version=opset, dynamo=False, do_constant_folding=True,
        )
    print(f"[decoder] wrote {out_path}")
    if validate:
        _check_onnx(out_path)
        _smoke_compare(wrapper, (token, h, c), out_path,
                       ["token", "h_in", "c_in"], "decoder_out")


def export_joiner(model, out_dir: Path, opset: int, validate: bool, device: str):
    cfg = model.config
    wrapper = JoinerStep(model.joint).eval().to(device)
    enc_frame = torch.randn(1, cfg.decoder_hidden_size, device=device)
    dec_out = torch.randn(1, cfg.decoder_hidden_size, device=device)

    out_path = out_dir / "joiner.onnx"
    with torch.no_grad():
        torch.onnx.export(
            wrapper, (enc_frame, dec_out), out_path,
            input_names=["encoder_frame", "decoder_out"],
            output_names=["logits"],
            opset_version=opset, dynamo=False, do_constant_folding=True,
        )
    print(f"[joiner] wrote {out_path}")
    if validate:
        _check_onnx(out_path)
        _smoke_compare(wrapper, (enc_frame, dec_out), out_path,
                       ["encoder_frame", "decoder_out"], "logits")


def _check_onnx(path: Path):
    import onnx

    # >2GiB models can't be serialized in-memory; check via the path-based API.
    onnx.checker.check_model(str(path))
    model = onnx.load(str(path), load_external_data=False)
    print(f"[validate] onnx.checker OK: {path.name} (opset {model.opset_import[0].version})")


def _smoke_compare(wrapper, torch_inputs, onnx_path: Path, input_names: list[str], probe_output: str):
    import onnxruntime as ort

    onnx_path = Path(onnx_path)  # tolerate str callers

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = wrapper(*torch_inputs)
    ref_map = ref if isinstance(ref, (tuple, list)) else (ref,)
    feed = {name: t.cpu().numpy() for name, t in zip(input_names, torch_inputs)}
    onnx_outs = sess.run(None, feed)
    out_names = [o.name for o in sess.get_outputs()]
    idx = out_names.index(probe_output)
    diff = abs(onnx_outs[idx] - ref_map[0].cpu().numpy()).max()
    print(f"[validate] {onnx_path.name} '{probe_output}' max abs diff vs PyTorch: {diff:.3e}")
    if diff > 1e-3:
        print(f"[validate] WARNING: diff {diff:.3e} exceeds 1e-3 tolerance", file=sys.stderr)


def _write_metadata(model, processor, out_dir: Path, chunk_sizes: list[int]):
    cfg = model.config
    enc_cfg = cfg.encoder_config
    left_context = enc_cfg.sliding_window - 1

    meta = {
        "base_model": MODEL_ID,
        "model_type": "nemotron3_5_asr",
        "vocab_size": cfg.vocab_size,
        "blank_token_id": cfg.blank_token_id,
        "pad_token_id": cfg.pad_token_id,
        "decoder_hidden_size": cfg.decoder_hidden_size,
        "num_decoder_layers": cfg.num_decoder_layers,
        "max_symbols_per_step": cfg.max_symbols_per_step,
        "num_prompts": cfg.num_prompts,
        "default_prompt_id": cfg.default_prompt_id,
        "prompt_dictionary": PROMPT_DICTIONARY,
        "encoder": {
            "hidden_size": enc_cfg.hidden_size,
            "num_hidden_layers": enc_cfg.num_hidden_layers,
            "num_attention_heads": enc_cfg.num_attention_heads,
            "subsampling_factor": enc_cfg.subsampling_factor,
            "left_context_frames": left_context,
            "frame_ms": enc_cfg.subsampling_factor * FEATURE_PARAMS["hop_length"]
            / FEATURE_PARAMS["sampling_rate"] * 1000,
        },
        "features": FEATURE_PARAMS,
        "chunk_ms_to_lookahead": {str(k): v for k, v in CHUNK_MS_TO_LOOKAHEAD.items()},
        "exported_chunk_ms": chunk_sizes,
        "cache_shapes": {k: list(v) for k, v in _cache_shapes(model, left_context).items()},
    }
    (out_dir / "nemotron_onnx_config.json").write_text(json.dumps(meta, indent=2))
    print(f"[meta] wrote {out_dir / 'nemotron_onnx_config.json'}")

    # id -> token piece map so the runtime engine needs no tokenizer library.
    tokenizer = processor.tokenizer
    with open(out_dir / "tokens.txt", "w", encoding="utf-8") as f:
        for i in range(cfg.vocab_size):
            piece = tokenizer.convert_ids_to_tokens(i)
            f.write(f"{i}\t{piece if piece is not None else ''}\n")
    print(f"[meta] wrote {out_dir / 'tokens.txt'} ({cfg.vocab_size} pieces)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=MODEL_ID, help="HF model id or local checkpoint dir")
    parser.add_argument("--output-dir", required=True, type=Path, help="directory for ONNX artifacts")
    parser.add_argument(
        "--chunk-ms", default="320",
        help="comma-separated chunk sizes in ms from {80,160,320,560,1120}, or 'all'",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--validate", action="store_true",
                        help="run onnx.checker and an ONNX-vs-PyTorch smoke comparison")
    parser.add_argument("--encoders-only", action="store_true",
                        help="skip decoder/joiner/metadata export (re-export encoders only)")
    args = parser.parse_args()

    chunk_sizes = sorted(CHUNK_MS_TO_LOOKAHEAD) if args.chunk_ms == "all" else [
        int(x) for x in args.chunk_ms.split(",")
    ]
    for cs in chunk_sizes:
        if cs not in CHUNK_MS_TO_LOOKAHEAD:
            parser.error(f"invalid --chunk-ms {cs}; choose from {sorted(CHUNK_MS_TO_LOOKAHEAD)} or 'all'")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, processor = _load_model(args.model_id, args.device)

    for cs in chunk_sizes:
        export_encoder(model, args.output_dir, cs, args.opset, args.validate, args.device)
    if args.encoders_only:
        print("Done (encoders only).")
        return
    export_decoder(model, args.output_dir, args.opset, args.validate, args.device)
    export_joiner(model, args.output_dir, args.opset, args.validate, args.device)
    _write_metadata(model, processor, args.output_dir, chunk_sizes)
    print("Done.")


if __name__ == "__main__":
    main()
