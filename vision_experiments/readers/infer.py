"""CPU reference for the iPhone-compatible ONNX metadata models."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .preprocess import ARROWS, DISTANCES, GATES, distance_inputs, sign_pixels


class MetadataReaders:
    def __init__(self, model_dir=None):
        self.directory = (
            Path(model_dir) if model_dir else Path(__file__).with_name("models")
        )
        self.sessions = {}
        self.options = ort.SessionOptions()
        self.options.intra_op_num_threads = 2
        self.options.inter_op_num_threads = 1

    def _session(self, category, task):
        key = (category, task)
        if key not in self.sessions:
            self.sessions[key] = ort.InferenceSession(
                str(self.directory / f"{category}_{task}.onnx"),
                self.options,
                providers=["CPUExecutionProvider"],
            )
        return self.sessions[key]

    @staticmethod
    def _float_image(pixels):
        return np.ascontiguousarray(pixels[np.newaxis], dtype=np.float32) / 255.0

    def _read(self, image, category, bbox):
        crop = self._float_image(sign_pixels(image, bbox, category))
        logits = self._session(category, "sign").run(None, {"crop": crop})[0][0]
        probability = np.exp(logits - logits.max())
        probability /= probability.sum()
        cls = int(probability.argmax())
        return cls, float(probability[cls])

    def read_gate(self, image, bbox):
        cls, confidence = self._read(image, "gate", bbox)
        return GATES[cls] if confidence >= 0.70 else None, confidence

    def read_distance(self, image, category, bbox):
        scene, detail, geometry = distance_inputs(image, bbox)
        logits = self._session(category, "distance").run(
            None,
            {
                "scene": self._float_image(scene),
                "detail": self._float_image(detail),
                "geometry": geometry[np.newaxis],
            },
        )[0][0]
        return DISTANCES[int(logits.argmax())]

    def predict(self, image, category, bbox):
        if category not in ("gate", "washroom", "restaurant"):
            raise ValueError(f"Unknown sign category: {category}")
        gate_id, arrow, confidence = None, None, None
        if category == "gate":
            gate_id, confidence = self.read_gate(image, bbox)
        else:
            cls, score = self._read(image, category, bbox)
            if score >= 0.55:
                arrow = ARROWS[cls]
        return {
            "observedGateID": gate_id,
            "gateConfidence": confidence,
            "arrowDirection": arrow,
            "distanceToSign": self.read_distance(image, category, bbox),
        }
