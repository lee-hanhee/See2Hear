"""Export one trained YOLO11n head as a fixed-shape ONNX and Core ML detector."""

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

SIZE = 640


def export_models(checkpoint, output, category):
    output.mkdir(exist_ok=True, parents=True)
    model = YOLO(checkpoint).model.float().eval().fuse()
    head = model.model[-1]
    if not isinstance(head, Detect) or head.nc != 1:
        raise ValueError("Expected a single-category YOLO detection head")
    head.export = True
    head.dynamic = False
    head.format = "onnx"
    torch.set_num_threads(3)
    # Materialize the anchor grids before static tracing. Random input exercises both score and geometry.
    dummy = torch.rand((1, 3, SIZE, SIZE), generator=torch.Generator().manual_seed(42))
    with torch.no_grad():
        reference = model(dummy).numpy()
        traced = torch.jit.trace(model, dummy, strict=False)
        onnx_path = output / f"{category}.onnx"
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            dynamo=False,
            opset_version=17,
            input_names=["image"],
            output_names=["detections"],
        )
    network = onnx.load(onnx_path)
    onnx.helper.set_model_props(
        network,
        {
            "class": category,
            "license": "AGPL-3.0",
            "pretrained_weights": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
            "input": "RGB float32 0..1, 640-square center letterbox, padding 114/255",
            "output": "1 x 5 x 8400; canvas cx, cy, width, height, class probability",
        },
    )
    onnx.save(network, onnx_path)
    onnx.checker.check_model(str(onnx_path))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 3
    session = ort.InferenceSession(
        str(onnx_path), opts, providers=["CPUExecutionProvider"]
    )
    preds = session.run(["detections"], {"image": dummy.numpy()})[0]
    np.testing.assert_allclose(preds, reference, atol=3e-3, rtol=2e-4)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, SIZE, SIZE),
                scale=1 / 255,
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[ct.TensorType(name="detections", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT16,
        skip_model_load=True,
    )
    mlmodel.author = "See2Hear, fine-tuned from Ultralytics YOLO11n."
    mlmodel.license = "AGPL-3.0"
    mlmodel.short_description = (
        f"{category.capitalize()} sign detection, fixed 640-square letterbox."
    )
    mlmodel.input_description["image"] = (
        "Upright RGB, aspect-ratio preserved, centered padding RGB 114."
    )
    mlmodel.output_description["detections"] = (
        "1 x 5 x 8400: center x/y, width, height in canvas pixels, probability."
    )
    package = output / f"{category}.mlpackage"
    mlmodel.save(str(package))
    spec = ct.models.utils.load_spec(str(package))
    assert spec.WhichOneof("Type") == "mlProgram"
    assert tuple(spec.description.output[0].type.multiArrayType.shape) == (1, 5, 8400)
    print(
        f"Verified {category} ONNX/PyTorch agreement and Core ML package specification."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("category", choices=("gate", "restaurant", "washroom"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "models")
    args = parser.parse_args()
    export_models(args.checkpoint, args.output, args.category)
