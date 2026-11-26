"""Export the exact deployed ONNX reader graphs to FP16 Core ML ML Programs."""

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import onnx
import onnxruntime as ort
from onnx2torch import convert
import torch


def trained_reader(graph):
    # Torch's ONNX exporter aliases identical constants with Identity nodes.
    # onnx2torch's conv adapter requires its weight and bias as initializers.
    constants = {tensor.name for tensor in graph.graph.initializer}
    aliases = {}
    keep = []
    for node in graph.graph.node:
        for i, name in enumerate(node.input):
            node.input[i] = aliases.get(name, name)
        if node.op_type == "Identity" and node.input[0] in constants:
            aliases[node.output[0]] = node.input[0]
        else:
            keep.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(keep)
    return convert(graph).eval()


def export(path, output):
    session_opts = ort.SessionOptions()
    session_opts.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), session_opts, providers=["CPUExecutionProvider"]
    )
    inputs = session.get_inputs()
    example = tuple(
        torch.rand(tuple(field.shape), dtype=torch.float32) for field in inputs
    )
    reference = session.run(
        None, {field.name: x.numpy() for field, x in zip(inputs, example)}
    )[0]
    model = trained_reader(onnx.load(str(path)))
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
        np.testing.assert_allclose(
            traced(*example).numpy(), reference, atol=2e-4, rtol=2e-4
        )
    package = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        inputs=[
            ct.TensorType(name=field.name, shape=field.shape, dtype=np.float32)
            for field in inputs
        ],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16,
    )
    package.short_description = "See2Hear " + path.stem.replace("_", " ") + " reader"
    destination = output / f"{path.stem}.mlpackage"
    package.save(str(destination))
    spec = ct.models.utils.load_spec(str(destination))
    assert [f.name for f in spec.description.input] == [field.name for field in inputs]
    assert (
        spec.description.output[0].name == "logits"
        and spec.WhichOneof("Type") == "mlProgram"
    )
    print(f"Converted, saved, and inspected {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", type=Path, default=Path(__file__).with_name("models")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.models
    output.mkdir(exist_ok=True, parents=True)
    torch.manual_seed(1773)
    torch.set_num_threads(2)
    models = sorted(args.models.glob("*.onnx"))
    if not models:
        raise FileNotFoundError(f"No ONNX readers in {args.models}")
    for path in models:
        export(path, output)


if __name__ == "__main__":
    main()
