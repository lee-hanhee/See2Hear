"""Reference implementation of the query-specific VisionResult pipeline."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from uuid import UUID, uuid4

from PIL import Image


@dataclass(frozen=True)
class Query:
    label: str
    query_id: UUID
    requested_gate: int | None = None

    def __post_init__(self):
        if self.label not in ("washroom", "restaurant", "gate"):
            raise ValueError("Unknown sign label")
        if self.requested_gate is not None and (
            self.label != "gate"
            or type(self.requested_gate) is not int
            or not 1 <= self.requested_gate <= 6
        ):
            raise ValueError("Only gate queries may specify a gate number from 1 to 6")


@dataclass(frozen=True)
class Match:
    box: tuple[int, int, int, int]


def overlap(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if intersection else 0.0


class VisionModel:
    def __init__(self):
        from detection.inference import Detector
        from readers.infer import MetadataReaders

        self.detector = Detector()
        self.readers = MetadataReaders()

    def frame(self, image, category):
        return AnalyzedFrame(image, category, self.detector, self.readers)


class AnalyzedFrame:
    """Reuse candidate and number readings across queries for the same image."""

    def __init__(self, image, category, detector, readers):
        self.image = image if image.mode == "RGB" else image.convert("RGB")
        self.category = category
        self.readers = readers
        self.candidates = [
            tuple(int(v) for v in box)
            for box, score in detector.predict(self.image, category)
        ]
        self.gate_ids = {}

    def locate(self, query, *, excluded=()):
        """Return the highest-ranked eligible box without reading its number."""
        if query.label != self.category:
            raise ValueError("Frame was analyzed for another sign category")
        return next(
            (Match(box) for box in self.candidates if box not in excluded), None
        )

    def gate_id(self, box):
        if box not in self.gate_ids:
            self.gate_ids[box], _ = self.readers.read_gate(self.image, box)
        return self.gate_ids[box]

    def result(self, query, match, include_details=False):
        result = dict(
            queryID=str(query.query_id),
            boundingBox=None,
            observedGateID=None,
            arrowDirection=None,
            distanceToSign=None,
        )
        if match is None:
            return result
        if include_details and self.category == "gate":
            observed = self.gate_id(match.box)
            if query.requested_gate is not None and observed != str(
                query.requested_gate
            ):
                return result
            result["observedGateID"] = observed
            result["distanceToSign"] = self.readers.read_distance(
                self.image, self.category, match.box
            )
        elif include_details:
            details = self.readers.predict(self.image, self.category, match.box)
            result["arrowDirection"] = details["arrowDirection"]
            result["distanceToSign"] = details["distanceToSign"]
        x1, y1, x2, y2 = match.box
        w, h = self.image.size
        result["boundingBox"] = dict(
            x=x1, y=y1, width=x2 - x1, height=y2 - y1, frameWidth=w, frameHeight=h
        )
        return result


def track_rejections(previous_boxes, candidates):
    """Match visible rejected screens to current detections one-to-one by IoU."""
    pairs = sorted(
        ((overlap(old, new), old, new) for old in previous_boxes for new in candidates),
        key=lambda pair: pair[0],
        reverse=True,
    )
    used = set()
    rejected = set()
    for score, old, new in pairs:
        if score < 0.4:
            break
        if old not in used and new not in rejected:
            used.add(old)
            rejected.add(new)
    return rejected


class VisionSession:
    """Expose the current phase alongside a Swift-shaped result for each frame.

    guiding is a provisional box. verified means the centered slow path succeeded.
    searching, rejected and unreadable have empty results. details_refreshed marks
    a slow-path attempt on this frame, including attempts that returned a miss.
    """

    def __init__(self, model):
        self.model = model
        self.previous_query = None
        self.previous_box = None
        self.previous_time = None
        self.centered_frames = 0
        self.detail_time = None
        self.cached_details = {}
        self.rejected_boxes = set()
        self.phase = "searching"
        self.details_refreshed = False

    def analyze(self, image, query, timestamp_seconds):
        return self.advance(
            self.model.frame(image, query.label), query, timestamp_seconds
        )

    def advance(self, frame, query, timestamp_seconds):
        self.details_refreshed = False
        continuous = (
            self.previous_query == query
            and self.previous_time is not None
            and 0 < timestamp_seconds - self.previous_time <= 0.5
        )
        self.rejected_boxes = (
            track_rejections(self.rejected_boxes, frame.candidates)
            if continuous and self.rejected_boxes
            else set()
        )
        match = frame.locate(query, excluded=self.rejected_boxes)
        box = match.box if match else None
        same_track = (
            continuous
            and self.phase != "rejected"
            and self.previous_box is not None
            and box is not None
            and overlap(self.previous_box, box) >= 0.4
        )
        self.previous_query, self.previous_box, self.previous_time = (
            query,
            box,
            timestamp_seconds,
        )
        if not same_track:
            self.centered_frames = 0
            self.detail_time = None
            self.cached_details = {}
        if match is None:
            self.phase = "rejected" if self.rejected_boxes else "searching"
            return frame.result(query, None)

        cx = (box[0] + box[2]) / (2 * frame.image.width)
        cy = (box[1] + box[3]) / (2 * frame.image.height)
        centered = abs(cx - 0.5) <= 0.12 and abs(cy - 0.5) <= 0.12
        self.centered_frames = self.centered_frames + 1 if centered else 0
        if self.centered_frames < 3:
            self.phase = "guiding"
            return frame.result(query, match)

        if self.detail_time is None or timestamp_seconds - self.detail_time >= 0.5:
            self.detail_time = timestamp_seconds
            self.details_refreshed = True
            result = frame.result(query, match, include_details=True)
            if result["boundingBox"] is None:
                self.cached_details = {}
                if frame.gate_id(box) is None:
                    self.phase = "unreadable"
                else:
                    self.phase = "rejected"
                    self.rejected_boxes.add(box)
            else:
                self.phase = "verified"
                self.cached_details = {
                    k: result[k]
                    for k in ("observedGateID", "arrowDirection", "distanceToSign")
                }
            return result
        if not self.cached_details:
            self.phase = "unreadable"
            return frame.result(query, None)
        self.phase = "verified"
        return {**frame.result(query, match), **self.cached_details}


def main():
    parser = argparse.ArgumentParser(description="Analyze one upright RGB frame")
    parser.add_argument("image", type=Path)
    parser.add_argument("label", choices=("washroom", "restaurant", "gate"))
    parser.add_argument("--gate", type=int)
    parser.add_argument(
        "--details", action="store_true", help="Run the slow reader on the selected box"
    )
    args = parser.parse_args()
    query = Query(args.label, uuid4(), args.gate)
    with Image.open(args.image) as image:
        frame = VisionModel().frame(image, query.label)
        print(
            json.dumps(frame.result(query, frame.locate(query), args.details), indent=2)
        )


if __name__ == "__main__":
    main()
