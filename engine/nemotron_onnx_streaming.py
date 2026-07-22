#!/usr/bin/env python3
"""Pure-onnxruntime streaming inference engine for the Nemotron 3.5 ASR ONNX export.

Dependencies: numpy + onnxruntime only (soundfile/scipy are used lazily by
``transcribe_file``). Implements:

* log-mel feature extraction identical to HF ``NemotronAsrStreamingFeatureExtractor``
  (preemphasis 0.97, STFT n_fft=512 / hop=160 / win=400 periodic Hann,
  slaney mel 128 bins, ``log(mel + 2**-24)``, no normalization),
* the cache-aware streaming schedule of the HF processor: first chunk of
  ``1 + 8*r`` mel frames, then chunks of ``8*(r+1)`` mel frames, where ``r``
  is the right attention context (lookahead) selected by ``chunk_ms``,
* RNNT greedy decoding matching ``ParakeetRNNTGenerationMixin``: the encoder
  frame pointer advances on blank (or after ``max_symbols_per_step`` non-blank
  emissions at one frame); the prediction-network state is only updated when a
  non-blank token is consumed,
* language-ID prompt conditioning via the exported ``prompt_ids`` encoder
  input, including ``auto`` mode with optional ``<xx-XX>`` tag stripping.

Grounded in transformers ``models/nemotron3_5_asr`` + ``nemotron_asr_streaming``
(v5.13.0) and the model card of nvidia/nemotron-3.5-asr-streaming-0.6b.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError as e:  # pragma: no cover
    raise ImportError("nemotron_onnx_streaming requires onnxruntime") from e

# ---------------------------------------------------------------------------
# Constants (HF checkpoint config.json / processor_config.json)
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 128
PREEMPHASIS = 0.97
LOG_ZERO_GUARD = 2**-24

SUBSAMPLING = 8
LEFT_CONTEXT = 56  # sliding_window (57) - 1, in subsampled encoder frames

CHUNK_MS_TO_LOOKAHEAD = {80: 0, 160: 1, 320: 3, 560: 6, 1120: 13}

BLANK_TOKEN_ID = 13087
VOCAB_SIZE = 13088
DECODER_LAYERS = 2
DECODER_HIDDEN = 640
MAX_SYMBOLS_PER_STEP = 10
NUM_PROMPTS = 128
AUTO_PROMPT_ID = 101

# locale-or-code -> prompt index ("auto" = automatic language detection)
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

_LANG_TAG_RE = re.compile(r"^<[a-z]{2,3}-[A-Z]{2,3}>$")
_BYTE_TOKEN_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


# ---------------------------------------------------------------------------
# Log-mel feature extraction (numpy replica of NemotronAsrStreamingFeatureExtractor)
# ---------------------------------------------------------------------------


def _slaney_hz_to_mel(freqs: np.ndarray) -> np.ndarray:
    """librosa 'slaney' (non-HTK) mel scale: linear below 1 kHz, log above."""
    freqs = np.atleast_1d(np.asarray(freqs, dtype=np.float64))
    f_sp = 200.0 / 3
    mels = freqs / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = freqs >= min_log_hz
    mels[log_region] = min_log_mel + np.log(freqs[log_region] / min_log_hz) / logstep
    return mels


def _slaney_mel_to_hz(mels: np.ndarray) -> np.ndarray:
    mels = np.atleast_1d(np.asarray(mels, dtype=np.float64))
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = mels >= min_log_mel
    freqs[log_region] = min_log_hz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return freqs


def slaney_mel_filters(
    sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS,
    fmin: float = 0.0, fmax: float | None = None,
) -> np.ndarray:
    """Mel filter bank matching ``librosa.filters.mel(norm='slaney')``.

    Verified against librosa 0.11: max abs diff 3.7e-9 (the HF extractor
    itself uses librosa under the hood).
    """
    if fmax is None:
        fmax = sr / 2
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_f = _slaney_mel_to_hz(np.linspace(
        _slaney_hz_to_mel(fmin).item(), _slaney_hz_to_mel(fmax).item(), n_mels + 2
    ))

    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fft_freqs[None, :]
    weights = np.zeros((n_mels, len(fft_freqs)))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    # slaney area normalization
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, None]
    return weights.astype(np.float32)


class LogMelExtractor:
    """Numpy replica of HF ``NemotronAsrStreamingFeatureExtractor``.

    Preemphasis is applied to the raw waveform *before* the STFT (as in HF),
    the STFT uses a periodic=False Hann window of ``win_length`` centered in
    the ``n_fft`` frame, power spectrum, slaney mel, ``log(mel + 2**-24)``.
    Features are NOT normalized (NemotronAsrStreaming never normalizes).
    """

    def __init__(self):
        # torch.hann_window(win_length, periodic=False) == np.hanning(win_length)
        hann = np.hanning(WIN_LENGTH).astype(np.float32)
        # torch.stft centers the window inside the n_fft frame
        pad = (N_FFT - WIN_LENGTH) // 2
        self.window = np.pad(hann, (pad, N_FFT - WIN_LENGTH - pad))
        self.mel_filters = slaney_mel_filters()

    def __call__(self, pcm: np.ndarray, center: bool) -> np.ndarray:
        """pcm: 1-D float32 16 kHz. Returns (num_frames, N_MELS) float32.

        ``center=True`` (offline / first chunk) pads n_fft//2 on both sides;
        ``center=False`` (subsequent chunks) does not, so feeding
        ``audio[frame*hop - n_fft//2 : ...]`` reproduces the frames a single
        centered pass over the whole utterance would have produced.
        """
        x = np.asarray(pcm, dtype=np.float32)
        if PREEMPHASIS:
            x = np.concatenate([x[:1], x[1:] - PREEMPHASIS * x[:-1]])
        if center:
            x = np.pad(x, N_FFT // 2)
        if len(x) < N_FFT:
            return np.zeros((0, N_MELS), dtype=np.float32)
        num_frames = 1 + (len(x) - N_FFT) // HOP_LENGTH
        # strided framing (num_frames, n_fft)
        strides = (x.strides[0] * HOP_LENGTH, x.strides[0])
        frames = np.lib.stride_tricks.as_strided(x, (num_frames, N_FFT), strides)
        spec = np.abs(np.fft.rfft(frames * self.window, n=N_FFT, axis=1)) ** 2
        mel = spec @ self.mel_filters.T
        return np.log(mel + LOG_ZERO_GUARD).astype(np.float32)


# ---------------------------------------------------------------------------
# Token decoding (SentencePiece-style pieces dumped by the export script)
# ---------------------------------------------------------------------------


class PieceDecoder:
    """id -> text using ``tokens.txt`` (id<TAB>piece) written by export_onnx.py.

    The NeMo multilingual tokenizer is SentencePiece-style: pieces concatenate
    directly and ``▁`` (U+2581) marks word starts. ``<0xNN>`` byte-fallback
    pieces are decoded through a byte buffer.
    """

    def __init__(self, tokens_path: Path | None, vocab_size: int):
        self.pieces: list[str] = [""] * vocab_size
        if tokens_path is not None and tokens_path.exists():
            for line in tokens_path.read_text(encoding="utf-8").splitlines():
                idx, _, piece = line.partition("\t")
                if idx.strip().isdigit() and int(idx) < vocab_size:
                    self.pieces[int(idx)] = piece

    def decode(self, token_ids: list[int], skip_special: bool = True) -> str:
        out: list[str] = []
        byte_buf = bytearray()

        def flush_bytes():
            if byte_buf:
                out.append(byte_buf.decode("utf-8", errors="replace"))
                byte_buf.clear()

        for tid in token_ids:
            piece = self.pieces[tid] if 0 <= tid < len(self.pieces) else ""
            if not piece:
                continue
            m = _BYTE_TOKEN_RE.match(piece)
            if m:
                byte_buf.append(int(m.group(1), 16))
                continue
            flush_bytes()
            if skip_special and piece.startswith("<") and piece.endswith(">"):
                continue
            out.append(piece)
        flush_bytes()
        # VERIFY: SentencePiece detokenization assumed (▁ -> space). If the
        # checkpoint ships a different tokenizer scheme, adjust here.
        return "".join(out).replace("▁", " ").strip()


# ---------------------------------------------------------------------------
# Streaming engine
# ---------------------------------------------------------------------------


class NemotronOnnxStreaming:
    """Cache-aware streaming RNNT ASR over the exported ONNX graphs.

    Parameters
    ----------
    model_dir:
        Directory with ``encoder_{chunk_ms}ms[_first].onnx``, ``decoder.onnx``,
        ``joiner.onnx``, ``tokens.txt`` and ``nemotron_onnx_config.json``.
    language:
        Locale (``"es-ES"``), bare code (``"de"``) or ``"auto"`` for automatic
        language detection. Resolved to the exported ``prompt_ids`` input.
    chunk_ms:
        Streaming chunk size: 80 / 160 / 320 / 560 / 1120 (latency vs accuracy).
    strip_lang_tags:
        In ``auto`` mode the model appends an ``<xx-XX>`` language tag after the
        terminal punctuation; True strips it from the text (still available as
        ``detected_language``), False keeps it.
    precision:
        "fp32" (default), "int8" or "fp16" — selects the file suffix produced
        by export/quantize.py. Falls back to fp32 if the variant is missing.
    """

    def __init__(
        self,
        model_dir: str | Path,
        language: str = "auto",
        chunk_ms: int = 320,
        num_threads: int = 4,
        strip_lang_tags: bool = True,
        precision: str = "fp32",
        providers: list[str] | None = None,
    ):
        self.model_dir = Path(model_dir)
        if chunk_ms not in CHUNK_MS_TO_LOOKAHEAD:
            raise ValueError(f"chunk_ms must be one of {sorted(CHUNK_MS_TO_LOOKAHEAD)}")
        self.chunk_ms = chunk_ms
        self.lookahead = CHUNK_MS_TO_LOOKAHEAD[chunk_ms]
        self.enc_frames_per_chunk = self.lookahead + 1
        self.mel_frames_first = 1 + SUBSAMPLING * self.lookahead
        self.mel_frames_steady = SUBSAMPLING * (self.lookahead + 1)
        # raw-sample window sizes (HF processor: num_samples_first/per_audio_chunk)
        self.samples_first = (self.mel_frames_first - 1) * HOP_LENGTH + WIN_LENGTH // 2
        self.samples_steady = self.mel_frames_steady * HOP_LENGTH + WIN_LENGTH

        self._load_metadata()
        if language not in self.prompt_dictionary:
            raise ValueError(
                f"unknown language {language!r}; use a locale (es-ES), a bare code (de), or 'auto'"
            )
        self.language = language
        self.prompt_id = self.prompt_dictionary[language]
        self.strip_lang_tags = strip_lang_tags

        self.extractor = LogMelExtractor()
        self._init_sessions(num_threads, precision, providers)
        self.reset()

    # -- setup ---------------------------------------------------------------

    def _load_metadata(self) -> None:
        meta_path = self.model_dir / "nemotron_onnx_config.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.vocab_size = meta["vocab_size"]
            self.blank_id = meta["blank_token_id"]
            self.max_symbols = meta["max_symbols_per_step"]
            self.prompt_dictionary = meta.get("prompt_dictionary", PROMPT_DICTIONARY)
            cache_shapes = meta["cache_shapes"]
        else:
            self.vocab_size = VOCAB_SIZE
            self.blank_id = BLANK_TOKEN_ID
            self.max_symbols = MAX_SYMBOLS_PER_STEP
            self.prompt_dictionary = PROMPT_DICTIONARY
            # derive the default cache shapes (600M checkpoint geometry)
            heads, head_dim = 8, 128
            cache_shapes = {}
            for i in range(24):
                cache_shapes[f"k_cache_{i}"] = [1, heads, LEFT_CONTEXT, head_dim]
                cache_shapes[f"v_cache_{i}"] = [1, heads, LEFT_CONTEXT, head_dim]
            cache_shapes["conv2d_cache_0"] = [1, 1, 1, N_MELS + 3]
            cache_shapes["conv2d_cache_1"] = [1, 256, 1, 68]
            cache_shapes["conv2d_cache_2"] = [1, 256, 1, 36]
            for i in range(24):
                cache_shapes[f"conv1d_cache_{i}"] = [1, 1024, 8]
        self.cache_shapes = {k: tuple(v) for k, v in cache_shapes.items()}
        self.piece_decoder = PieceDecoder(self.model_dir / "tokens.txt", self.vocab_size)

    def _resolve(self, stem: str, precision: str) -> Path:
        suffix = "" if precision == "fp32" else f"_{precision}"
        candidates = [
            self.model_dir / stem / f"{stem}{suffix}.onnx",  # per-variant subdir
            self.model_dir / f"{stem}{suffix}.onnx",         # flat layout
        ]
        if suffix:
            candidates += [
                self.model_dir / stem / f"{stem}.onnx",      # graceful fp32 fallback
                self.model_dir / f"{stem}.onnx",
            ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"missing ONNX graph for {stem!r} in {self.model_dir}")

    def _init_sessions(self, num_threads: int, precision: str, providers: list[str] | None) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        if providers is None:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in ort.get_available_providers()
                else ["CPUExecutionProvider"]
            )
        stem = f"encoder_{self.chunk_ms}ms"
        self.enc_first = ort.InferenceSession(
            str(self._resolve(stem + "_first", precision)), sess_options=opts, providers=providers
        )
        self.enc_steady = ort.InferenceSession(
            str(self._resolve(stem, precision)), sess_options=opts, providers=providers
        )
        self.decoder = ort.InferenceSession(
            str(self._resolve("decoder", precision)), sess_options=opts, providers=providers
        )
        self.joiner = ort.InferenceSession(
            str(self._resolve("joiner", precision)), sess_options=opts, providers=providers
        )
        self._enc_input_names = [i.name for i in self.enc_steady.get_inputs()]
        self._enc_output_names = [o.name for o in self.enc_steady.get_outputs()]
        # map "k_cache_out_3" -> "k_cache_3" etc. for cache carry-over
        self._cache_carry = {
            out: out.replace("_out_", "_") for out in self._enc_output_names[1:]
        }

    # -- state ----------------------------------------------------------------

    def reset(self) -> None:
        """Reset all streaming state (caches, buffers, hypothesis)."""
        self.caches = {
            name: np.zeros(shape, dtype=np.float32) for name, shape in self.cache_shapes.items()
        }
        self.cache_valid = 0  # right-aligned valid attention cache frames
        self._pcm = np.zeros(0, dtype=np.float32)
        self._pcm_start = 0  # absolute sample index of _pcm[0]
        self._mel_idx = 0    # next mel frame index to produce
        self._started = False
        self._flushed = False
        # decoder (prediction network) state; primed with the blank token below
        self._dec_h = np.zeros((DECODER_LAYERS, 1, DECODER_HIDDEN), dtype=np.float32)
        self._dec_c = np.zeros_like(self._dec_h)
        self._dec_out = self._run_decoder(self.blank_id)
        self._tokens: list[int] = []
        self._lang_tags: list[str] = []
        self.audio_seconds = 0.0
        self.inference_seconds = 0.0

    # -- decoder / joiner -------------------------------------------------------

    def _run_decoder(self, token_id: int) -> np.ndarray:
        """Advance the prediction network with one (non-blank or initial blank)
        token and commit the LSTM state. Returns the new decoder output."""
        out, h, c = self.decoder.run(
            None,
            {
                "token": np.array([[token_id]], dtype=np.int64),
                "h_in": self._dec_h,
                "c_in": self._dec_c,
            },
        )
        self._dec_h, self._dec_c = h, c
        return out

    def _greedy_decode_chunk(self, enc_out: np.ndarray, num_frames: int) -> None:
        """RNNT greedy decode over ``num_frames`` encoder frames of one chunk.

        Mirrors ParakeetRNNTGenerationMixin: advance on blank; on non-blank,
        emit the token and step the prediction network; force-advance after
        ``max_symbols`` consecutive non-blank emissions at the same frame.
        """
        for t in range(num_frames):
            enc_frame = enc_out[0, t : t + 1]  # (1, decoder_hidden)
            symbols = 0
            while True:
                logits = self.joiner.run(None, {"encoder_frame": enc_frame,
                                                "decoder_out": self._dec_out})[0]
                token = int(np.argmax(logits, axis=-1)[0])
                if token == self.blank_id:
                    break  # advance frame; prediction state untouched
                piece = self.piece_decoder.pieces[token]
                if _LANG_TAG_RE.match(piece):
                    self._lang_tags.append(piece)
                    if not self.strip_lang_tags:
                        self._tokens.append(token)
                else:
                    self._tokens.append(token)
                self._dec_out = self._run_decoder(token)
                symbols += 1
                if symbols >= self.max_symbols:
                    break  # forced advance

    # -- encoder step -----------------------------------------------------------

    def _run_encoder(self, features: np.ndarray, first: bool) -> np.ndarray:
        """features: (num_mel_frames, N_MELS). Returns (1, enc_frames, 640)."""
        expected = self.mel_frames_first if first else self.mel_frames_steady
        if features.shape[0] != expected:
            raise ValueError(f"chunk has {features.shape[0]} mel frames, expected {expected}")
        left = LEFT_CONTEXT
        mask = np.zeros((1, 1, 1, left + self.enc_frames_per_chunk), dtype=np.float32)
        invalid = left - self.cache_valid
        if invalid > 0:
            mask[..., :invalid] = -1e9
        feed = {
            "input_features": features[None].astype(np.float32),
            "prompt_ids": np.array([self.prompt_id], dtype=np.int64),
            "cache_mask": mask,
        }
        feed.update(self.caches)
        session = self.enc_first if first else self.enc_steady
        t0 = time.perf_counter()
        outs = session.run(None, feed)
        self.inference_seconds += time.perf_counter() - t0
        for name, value in zip(self._enc_output_names[1:], outs[1:]):
            self.caches[self._cache_carry[name]] = value
        self.cache_valid = min(left, self.cache_valid + self.enc_frames_per_chunk)
        return outs[0]

    # -- chunk scheduling --------------------------------------------------------

    def _produce_first_chunk(self, decode_frames: int | None = None) -> None:
        pcm = self._pcm[: self.samples_first]
        feats = self.extractor(pcm, center=True)[: self.mel_frames_first]
        if feats.shape[0] < self.mel_frames_first:  # short audio: zero-pad features
            pad = np.zeros((self.mel_frames_first - feats.shape[0], N_MELS), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=0)
        enc_out = self._run_encoder(feats, first=True)
        self._started = True
        self._mel_idx = self.mel_frames_first
        self._greedy_decode_chunk(enc_out, decode_frames or self.enc_frames_per_chunk)
        self._discard_consumed_pcm()

    def _produce_steady_chunk(self, pcm_window: np.ndarray, valid_mel: int | None = None) -> None:
        feats = self.extractor(pcm_window, center=False)
        if valid_mel is not None and valid_mel < feats.shape[0]:
            feats[valid_mel:] = 0.0  # mirror HF's masking of invalid frames
        feats = feats[: self.mel_frames_steady]
        if feats.shape[0] < self.mel_frames_steady:
            pad = np.zeros((self.mel_frames_steady - feats.shape[0], N_MELS), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=0)
        enc_out = self._run_encoder(feats, first=False)
        if valid_mel is None:
            decode_frames = self.enc_frames_per_chunk
        else:
            # VERIFY: encoder frames touched by valid audio ~ ceil(valid_mel / subsampling);
            # boundary frame count may need +0/+1 tuning against HF in phase 2.
            decode_frames = min(self.enc_frames_per_chunk, max(1, -(-valid_mel // SUBSAMPLING)))
        self._mel_idx += self.mel_frames_steady
        self._greedy_decode_chunk(enc_out, decode_frames)
        self._discard_consumed_pcm()

    def _discard_consumed_pcm(self) -> None:
        next_start = self._mel_idx * HOP_LENGTH - N_FFT // 2
        drop = max(0, next_start - self._pcm_start)
        if drop:
            self._pcm = self._pcm[drop:]
            self._pcm_start += drop

    # -- public API ----------------------------------------------------------------

    def accept_waveform(self, pcm: np.ndarray) -> None:
        """Feed float32 mono 16 kHz samples (any length). Runs encoder/decoder
        steps for every complete chunk that can be produced."""
        pcm = np.asarray(pcm, dtype=np.float32).ravel()
        self._flushed = False
        self._pcm = np.concatenate([self._pcm, pcm])
        self.audio_seconds += len(pcm) / SAMPLE_RATE

        pcm_end = self._pcm_start + len(self._pcm)
        if not self._started:
            if len(self._pcm) < self.samples_first:
                return
            self._produce_first_chunk()
        while True:
            start = self._mel_idx * HOP_LENGTH - N_FFT // 2
            if start + self.samples_steady > pcm_end:
                break
            offset = start - self._pcm_start
            self._produce_steady_chunk(self._pcm[offset : offset + self.samples_steady])

    def finish(self) -> str:
        """Flush the audio tail (zero-padded final chunk) and return the text."""
        if self._flushed:
            return self.get_partial()
        self._flushed = True
        pcm_end = self._pcm_start + len(self._pcm)
        if not self._started:
            if pcm_end == 0:
                return ""
            real = len(self._pcm)
            padded = np.pad(self._pcm, (0, max(0, self.samples_first - real)))
            self._pcm = padded[: self.samples_first]
            # valid mel frames under center=True: floor(L / hop)
            valid_mel = min(self.mel_frames_first, real // HOP_LENGTH)
            feats = self.extractor(self._pcm, center=True)
            if valid_mel < feats.shape[0]:
                feats[valid_mel:] = 0.0
            feats = feats[: self.mel_frames_first]
            if feats.shape[0] < self.mel_frames_first:
                feats = np.concatenate(
                    [feats, np.zeros((self.mel_frames_first - feats.shape[0], N_MELS),
                                     dtype=np.float32)], axis=0)
            enc_out = self._run_encoder(feats, first=True)
            self._started = True
            self._mel_idx = self.mel_frames_first
            decode_frames = min(self.enc_frames_per_chunk,
                                max(1, -(-valid_mel // SUBSAMPLING))) if valid_mel else 0
            if decode_frames:
                self._greedy_decode_chunk(enc_out, decode_frames)
            return self.get_partial()
        start = self._mel_idx * HOP_LENGTH - N_FFT // 2
        real = pcm_end - max(start, self._pcm_start) if pcm_end > start else 0
        # a complete mel frame needs at least n_fft real samples (center=False)
        valid_mel = max(0, (real - N_FFT) // HOP_LENGTH + 1)
        if valid_mel > 0:
            offset = max(start - self._pcm_start, 0)
            window = self._pcm[offset : offset + self.samples_steady]
            if len(window) < self.samples_steady:
                window = np.pad(window, (0, self.samples_steady - len(window)))
            self._produce_steady_chunk(window, valid_mel=valid_mel)
        return self.get_partial()

    def get_partial(self) -> str:
        """Current hypothesis (decode so far, without flushing the tail)."""
        return self.piece_decoder.decode(self._tokens, skip_special=True)

    def get_final(self) -> str:
        """Flush and return the final transcript."""
        return self.finish()

    @property
    def detected_language(self) -> str | None:
        """Language tag emitted in ``auto`` mode, e.g. ``<es-ES>`` (or None)."""
        return self._lang_tags[-1] if self._lang_tags else None

    @property
    def rtf(self) -> float:
        """Real-time factor of encoder inference so far (lower is faster)."""
        return self.inference_seconds / self.audio_seconds if self.audio_seconds else 0.0

    # -- convenience ----------------------------------------------------------------

    def transcribe_file(self, path: str | Path, block_seconds: float = 2.0) -> str:
        """Transcribe a WAV/FLAC/... file end-to-end (resets state first)."""
        import soundfile as sf

        pcm, sr = sf.read(str(path), dtype="float32", always_2d=True)
        pcm = pcm.mean(axis=1)  # mono
        if sr != SAMPLE_RATE:
            from scipy.signal import resample_poly

            gcd = np.gcd(sr, SAMPLE_RATE)
            pcm = resample_poly(pcm, SAMPLE_RATE // gcd, sr // gcd).astype(np.float32)
        self.reset()
        block = int(block_seconds * SAMPLE_RATE)
        for i in range(0, len(pcm), block):
            self.accept_waveform(pcm[i : i + block])
        return self.get_final()


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe a file with the ONNX streaming engine")
    parser.add_argument("model_dir")
    parser.add_argument("wav")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--chunk-ms", type=int, default=320)
    parser.add_argument("--precision", default="fp32", choices=["fp32", "int8", "fp16"])
    parser.add_argument("--keep-lang-tags", action="store_true")
    args = parser.parse_args()

    engine = NemotronOnnxStreaming(
        args.model_dir, language=args.language, chunk_ms=args.chunk_ms,
        strip_lang_tags=not args.keep_lang_tags, precision=args.precision,
    )
    text = engine.transcribe_file(args.wav)
    print(f"text: {text}")
    if engine.detected_language:
        print(f"detected language: {engine.detected_language}")
    print(f"RTF: {engine.rtf:.3f}")
