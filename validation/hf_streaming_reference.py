"""HF transformers STREAMING reference at a given lookahead, for cross-checking
the ONNX engine at the matching chunk size. Prints the transcript to stdout."""
import sys

import soundfile as sf
import torch
from transformers import AutoModelForRNNT, AutoProcessor

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"


def main(wav: str, lookahead: int) -> None:
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForRNNT.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    proc.set_num_lookahead_tokens(lookahead)
    print(f"# streaming latency: {proc.streaming_latency_ms} ms", file=sys.stderr)

    audio, sr = sf.read(wav)
    first = proc(
        audio[: proc.num_samples_first_audio_chunk],
        sampling_rate=sr, is_streaming=True, is_first_audio_chunk=True,
        language="auto", return_tensors="pt",
    )

    def gen():
        yield first.input_features[:, : proc.num_mel_frames_first_audio_chunk, :]
        idx = proc.num_mel_frames_first_audio_chunk
        hop = proc.feature_extractor.hop_length
        n_fft = proc.feature_extractor.n_fft
        start = idx * hop - n_fft // 2
        while start < audio.shape[0] and (end := start + proc.num_samples_per_audio_chunk) < audio.shape[0]:
            inp = proc(
                audio[start:end], sampling_rate=sr, is_streaming=True,
                is_first_audio_chunk=False, language="auto", return_tensors="pt",
            )
            yield inp.input_features
            idx += proc.num_mel_frames_per_audio_chunk
            start = idx * hop - n_fft // 2

    kwargs = {**first, "input_features": gen()}
    with torch.no_grad():
        out = model.generate(**kwargs, return_dict_in_generate=True)
    print(proc.decode(out.sequences[0], skip_special_tokens=True))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
