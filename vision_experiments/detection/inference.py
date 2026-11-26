"""CPU reference for the one-class sign detectors. Coordinates are upright top-left XYXY."""

from pathlib import Path
import math

import numpy as np
import onnxruntime as ort
from PIL import Image

CATEGORIES = ("gate", "restaurant", "washroom")
SIZE = 640
SCORE_THRESHOLD = 0.25
NMS_IOU = 0.45


def letterbox(image):
    rgb = image.convert("RGB")
    scale = min(SIZE / rgb.width, SIZE / rgb.height)
    width = max(1, round(rgb.width * scale))
    height = max(1, round(rgb.height * scale))
    left = (SIZE - width) // 2
    top = (SIZE - height) // 2
    canvas = Image.new("RGB", (SIZE, SIZE), (114, 114, 114))
    canvas.paste(rgb.resize((width, height), Image.Resampling.BILINEAR), (left, top))
    tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(tensor), (
        left,
        top,
        width / rgb.width,
        height / rgb.height,
    )


def decode(output, size, transform, threshold=SCORE_THRESHOLD):
    width, height = size
    left, top, sx, sy = transform
    pred = output[0].T
    pred = pred[pred[:, 4] >= threshold]
    if not len(pred):
        return []
    order = np.argsort(-pred[:, 4], kind="stable")
    pred = pred[order]
    xy = pred[:, :2]
    half_size = pred[:, 2:4] / 2
    boxes = np.concatenate((xy - half_size, xy + half_size), axis=1)
    boxes[:, 0::2] = np.clip((boxes[:, 0::2] - left) / sx, 0, width)
    boxes[:, 1::2] = np.clip((boxes[:, 1::2] - top) / sy, 0, height)
    boxes = np.floor(boxes + 0.5).astype(np.int64)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    remaining = np.flatnonzero(areas > 0)
    keep = []
    while len(remaining):
        i = remaining[0]
        keep.append(i)
        other = remaining[1:]
        overlap = np.maximum(
            0,
            np.minimum(boxes[i, 2:], boxes[other, 2:])
            - np.maximum(boxes[i, :2], boxes[other, :2]),
        ).prod(axis=1)
        union = areas[i] + areas[other] - overlap
        remaining = other[overlap <= NMS_IOU * union]
    if not keep:
        return []
    max_area = max(areas[i] for i in keep)

    def rank(i):
        cx, cy = (boxes[i, :2] + boxes[i, 2:]) / 2
        centrality = 1 - min(
            1.0, math.hypot(cx / width - 0.5, cy / height - 0.5) * math.sqrt(2)
        )
        return (
            pred[i, 4]
            * (1 + 0.30 * math.sqrt(areas[i] / max_area))
            * (1 + 0.15 * centrality)
        )

    keep.sort(key=rank, reverse=True)
    return [(tuple(int(x) for x in boxes[i]), float(pred[i, 4])) for i in keep]


class Detector:
    """Loads each class model at first use; the first candidate is the generic-query selection."""

    def __init__(self, model_dir=None, threads=3):
        self.model_dir = (
            Path(model_dir) if model_dir else Path(__file__).parent / "models"
        )
        self.sessions = {}
        self.options = ort.SessionOptions()
        self.options.intra_op_num_threads = threads
        self.options.inter_op_num_threads = 1

    def predict(self, image, category):
        if category not in CATEGORIES:
            raise ValueError(f"Unknown sign category: {category}")
        if category not in self.sessions:
            self.sessions[category] = ort.InferenceSession(
                str(self.model_dir / f"{category}.onnx"),
                self.options,
                providers=["CPUExecutionProvider"],
            )
        tensor, transform = letterbox(image)
        output = self.sessions[category].run(["detections"], {"image": tensor})[0]
        return decode(output, image.size, transform)
