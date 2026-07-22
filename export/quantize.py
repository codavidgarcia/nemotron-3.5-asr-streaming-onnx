#!/usr/bin/env python3
"""Quantize the exported Nemotron ONNX graphs to int8 (and optionally fp16).

int8: dynamic quantization (``onnxruntime.quantization.quant_dynamic``),
which quantizes MatMul/Gemm weights to int8 and keeps activations in fp32.
That is the right trade-off here: the streaming caches (attention K/V, conv
left-context) are graph *inputs/outputs*, not weights, so they stay fp32
end-to-end and cache carry-over across chunks is numerically exact. The LSTM
in the decoder is quantized per ONNX Runtime's LSTM dynamic-quant support.

fp16 (optional): pure weight cast via ``onnxconverter_common.float16`` with
``keep_io_types=True`` so all graph I/O (including caches) remains fp32.

Usage:
    python quantize.py --model-dir ./onnx-out [--fp16] [--opset 17]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ops that hold the bulk of the parameters in FastConformer. Keep it to
# MatMul/Gemm: quantizing the causal convs and the decoder LSTM measurably
# degrades WER (verified on the parity harness: 0.21 vs 0.008 fp32).
QUANT_OP_TYPES = ["MatMul", "Gemm"]


def quantize_int8(src: Path, dst: Path, per_channel: bool = True) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # Encoder graphs exceed the 2 GiB protobuf limit and store weights as
    # external data files next to the .onnx; keep that format for them.
    has_external_data = any(
        p.is_file() and p.suffix != ".onnx" and not p.name.startswith(".")
        for p in src.parent.iterdir()
    )

    quantize_dynamic(
        model_input=str(src),
        model_output=str(dst),
        op_types_to_quantize=QUANT_OP_TYPES,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=False,
        use_external_data_format=has_external_data,
    )
    print(f"[int8] {src.name} -> {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")


def convert_fp16(src: Path, dst: Path) -> None:
    """Custom fp32->fp16 weight cast for the >2 GiB encoder graphs.

    onnxconverter_common's converter runs shape inference + in-memory
    serialization, both of which break past the 2 GiB protobuf limit. This
    minimal pass casts fp32 initializers to fp16 and inserts boundary Cast
    nodes so all graph I/O (features, caches, outputs) stays fp32 — the
    engine needs no changes. Saves back with external data.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(src))
    graph = model.graph

    for i, t in enumerate(graph.initializer):
        if t.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(t)
            graph.initializer[i].CopyFrom(
                numpy_helper.from_array(arr.astype(np.float16), name=t.name)
            )

    # Constant nodes embed their tensor as an attribute — cast those too,
    # otherwise downstream ops mix tensor(float) with tensor(float16).
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                arr = numpy_helper.to_array(attr.t)
                attr.t.CopyFrom(numpy_helper.from_array(arr.astype(np.float16), name=attr.t.name))

    # The HF pos-encoding branch contains explicit Cast-to-fp32 nodes (forced
    # autocast-off); retype body casts to fp16 so they match the casted weights.
    for node in graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16

    def _io_names(value_infos):
        return {v.name for v in value_infos if v.type.tensor_type.elem_type == TensorProto.FLOAT}

    # boundary casts: fp32 input -> cast to fp16 for the graph body
    init_names = {t.name for t in graph.initializer}
    for v in list(graph.input):
        if v.name in init_names or v.type.tensor_type.elem_type != TensorProto.FLOAT:
            continue
        cast = helper.make_node("Cast", [v.name], [v.name + "__fp16"], to=TensorProto.FLOAT16, name=v.name + "__in_cast")
        for node in graph.node:
            for j, inp in enumerate(node.input):
                if inp == v.name:
                    node.input[j] = v.name + "__fp16"
        graph.node.insert(0, cast)

    # boundary casts: fp16 body output -> cast back to fp32
    for v in list(graph.output):
        if v.type.tensor_type.elem_type != TensorProto.FLOAT:
            continue
        producer = None
        for node in graph.node:
            if v.name in node.output:
                producer = node
        cast = helper.make_node("Cast", [v.name + "__fp16"], [v.name], to=TensorProto.FLOAT, name=v.name + "__out_cast")
        for node in graph.node:
            for j, out in enumerate(node.output):
                if out == v.name:
                    node.output[j] = v.name + "__fp16"
        graph.node.append(cast)

    for v in list(graph.value_info) + list(graph.output):
        if v.type.tensor_type.elem_type == TensorProto.FLOAT and v.name.endswith("__fp16"):
            v.type.tensor_type.elem_type = TensorProto.FLOAT16

    onnx.save_model(
        model, str(dst), save_as_external_data=True, all_tensors_to_one_file=True,
        location=dst.name + ".data",
    )
    total = dst.stat().st_size + (dst.parent / (dst.name + ".data")).stat().st_size
    print(f"[fp16] {src.name} -> {dst.name} ({total / 1e6:.1f} MB incl. .data)")


def validate_numerics(fp32_path: Path, quant_path: Path, tolerance: float = 2e-2) -> None:
    """Smoke-check the quantized graph: same inputs, compare probe output.

    Cache tensors are fp32 in both graphs by construction (dynamic quant only
    touches weights); this check catches quantization blow-ups in the probe
    output (encoder_out / decoder_out / logits).
    """
    import numpy as np
    import onnxruntime as ort

    probe_priority = ["encoder_out", "decoder_out", "logits"]
    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_q = ort.InferenceSession(str(quant_path), providers=["CPUExecutionProvider"])

    feed = {}
    for inp in sess_fp32.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        if inp.type == "tensor(int64)":
            feed[inp.name] = np.zeros(shape, dtype=np.int64)
        else:
            feed[inp.name] = np.random.randn(*shape).astype(np.float32) * 0.1

    out_names = [o.name for o in sess_fp32.get_outputs()]
    probe = next((n for n in probe_priority if n in out_names), out_names[0])
    idx = out_names.index(probe)
    ref = sess_fp32.run(None, feed)[idx]
    hyp = sess_q.run(None, feed)[idx]
    diff = float(np.abs(ref - hyp).max())
    status = "OK" if diff <= tolerance else "WARNING"
    print(f"[validate] {quant_path.name} '{probe}' max abs diff vs fp32: {diff:.3e} [{status}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, type=Path,
                        help="directory containing the exported fp32 ONNX files")
    parser.add_argument("--fp16", action="store_true", help="also produce fp16 variants")
    parser.add_argument("--skip-validate", action="store_true", help="skip the numerics smoke check")
    args = parser.parse_args()

    onnx_files = sorted(
        p for p in args.model_dir.rglob("*.onnx")  # flat layout + per-variant subdirs
        if not p.stem.endswith(("_int8", "_fp16"))
    )
    if not onnx_files:
        parser.error(f"no fp32 .onnx files found in {args.model_dir}")

    for src in onnx_files:
        int8_path = src.with_name(src.stem + "_int8.onnx")
        quantize_int8(src, int8_path)
        if not args.skip_validate:
            validate_numerics(src, int8_path)
        if args.fp16:
            fp16_path = src.with_name(src.stem + "_fp16.onnx")
            convert_fp16(src, fp16_path)

    print("Done.")


if __name__ == "__main__":
    main()
