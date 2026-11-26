"""Evaluate the shipped artifacts on all eligible IMG_8458 records."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from time import perf_counter
from pathlib import Path
from uuid import UUID

from PIL import Image

from replay import gate_box_relation
from vision import Query, VisionModel, overlap

HERE = Path(__file__).resolve().parent
TEST_VIDEO = "IMG_8458.MOV"
CATEGORIES = ("gate", "restaurant", "washroom")
EVAL_ID = UUID("00000000-0000-4000-a000-000000000001")


def ratio(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


class DetectionScores:
    def __init__(self):
        self.counts = Counter()
        self.center_errors = []

    def add(self, match, boxes, size):
        found = match is not None
        positive = bool(boxes)
        valid = found and any(overlap(match.box, b) >= 0.5 for b in boxes)
        self.counts["examples"] += 1
        self.counts["positives"] += positive
        self.counts["negatives"] += not positive
        self.counts["true_positive"] += valid
        self.counts["false_positive"] += found and not valid
        self.counts["false_negative"] += positive and not valid
        self.counts["true_negative"] += not positive and not found
        self.counts["false_alarm_on_negative"] += found and not positive
        if valid:
            actual = max(boxes, key=lambda b: overlap(b, match.box))
            dx = (match.box[0] + match.box[2] - actual[0] - actual[2]) / 2
            dy = (match.box[1] + match.box[3] - actual[1] - actual[3]) / 2
            self.center_errors.append(math.hypot(dx, dy) / math.hypot(*size))
        return valid

    def summary(self):
        c = self.counts
        return dict(
            c,
            precision=ratio(
                c["true_positive"], c["true_positive"] + c["false_positive"]
            ),
            recall=ratio(c["true_positive"], c["positives"]),
            false_alarm_rate=ratio(c["false_alarm_on_negative"], c["negatives"]),
            mean_center_error_fraction_of_diagonal=ratio(
                sum(self.center_errors), len(self.center_errors)
            ),
        )


class FieldScores:
    def __init__(self):
        self.confusion = defaultdict(Counter)

    def add(self, expected, predicted):
        self.confusion[expected or "nil"][predicted or "nil"] += 1

    def summary(self):
        total = sum(sum(row.values()) for row in self.confusion.values())
        exact = sum(row[expected] for expected, row in self.confusion.items())
        answered = sum(
            n
            for row in self.confusion.values()
            for pred, n in row.items()
            if pred != "nil"
        )
        answerable = sum(
            sum(row.values()) for label, row in self.confusion.items() if label != "nil"
        )
        correct_answer = exact - self.confusion.get("nil", {}).get("nil", 0)
        return dict(
            n=total,
            correct=exact,
            accuracy=ratio(exact, total),
            coverage=ratio(answered, total),
            accuracy_on_answers=ratio(correct_answer, answered),
            recall_of_known_values=ratio(correct_answer, answerable),
            confusion={
                k: dict(sorted(row.items()))
                for k, row in sorted(self.confusion.items())
            },
        )


def expected_metadata(r):
    values = {
        "distanceToSign": {"near": "near", "mid": "medium", "far": "far"}[r["distance"]]
    }
    if r["class"] == "gate":
        values["observedGateID"] = str(r["gate_number"])
    else:
        direction = r["arrow_direction"]
        values["arrowDirection"] = "forward" if direction == "front" else direction
    return values


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(data, model, timings=None):
    detections = {c: DetectionScores() for c in CATEGORIES}
    metadata = defaultdict(lambda: defaultdict(FieldScores))
    oracle = defaultdict(lambda: defaultdict(FieldScores))
    joint_correct = Counter()
    by_distance = defaultdict(lambda: defaultdict(Counter))
    numbered = {i: DetectionScores() for i in range(1, 7)}
    false_confirmation_on_other_gate = Counter()
    selected_requested_gate = Counter()
    gate_id_on_known_boxes = FieldScores()
    gate_id_on_ambiguous_boxes = FieldScores()
    unscored = Counter()
    numbered_all = DetectionScores()
    labels_path = data / "labels.json"
    dataset = json.loads(labels_path.read_text())
    if dataset["video"] != TEST_VIDEO or len(dataset["frames"]) != 100:
        raise ValueError(f"Expected 100 annotated frames from {TEST_VIDEO}")
    per_frame = []
    for key, entry in sorted(dataset["frames"].items(), key=lambda item: int(item[0])):
        frame_id = int(key)
        records = entry["labels"]
        image_path = data / entry["image"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        for category in CATEGORIES:
            r = records.get(category)
            if r is None and category != "gate":
                continue
            begin = perf_counter()
            frame = model.frame(image, category)
            if timings is not None:
                timings["detector_" + category].append((perf_counter() - begin) * 1000)
            if r:
                query = Query(category, EVAL_ID)
                match = frame.locate(query)
                begin = perf_counter()
                prediction = frame.result(query, match, include_details=True)
                if timings is not None and match is not None:
                    timings["metadata_" + category].append(
                        (perf_counter() - begin) * 1000
                    )
                positive = r["object_present"]
                valid = detections[category].add(
                    match, [r["bbox"]] if positive else [], image.size
                )
                if positive:
                    group = by_distance[category][r["distance"]]
                    group["positives"] += 1
                    group["localized"] += int(valid)
                    expected = expected_metadata(r)
                    # Isolate reader performance from detection errors using the annotated box.
                    reference = model.readers.predict(image, category, tuple(r["bbox"]))
                    for field, value in expected.items():
                        oracle[category][field].add(value, reference[field])
                        if valid:
                            metadata[category][field].add(value, prediction[field])
                    joint_correct[category] += valid and all(
                        prediction[field] == v for field, v in expected.items()
                    )
                per_frame.append(
                    dict(frame=frame_id, category=category, result=prediction)
                )
            if category == "gate":
                annotation = entry["gates"]
                # Measure the number reader independently of detection and generic ranking.
                for instance in annotation["instances"]:
                    number, _ = model.readers.read_gate(image, tuple(instance["bbox"]))
                    gate_id_on_known_boxes.add(instance["gate_id"], number)
                for box in annotation["ignore_boxes"]:
                    number, _ = model.readers.read_gate(image, tuple(box))
                    gate_id_on_ambiguous_boxes.add(None, number)

                for gate_id, scores in numbered.items():
                    query = Query("gate", EVAL_ID, gate_id)
                    # Force centering only for this static benchmark. The runtime chooses
                    # its single candidate and performs the same slow check as a live call.
                    selected = frame.locate(query)
                    result = frame.result(query, selected, include_details=True)
                    confirmed = selected if result["boundingBox"] is not None else None
                    if confirmed and result["observedGateID"] != str(gate_id):
                        raise ValueError(
                            "Numbered slow result exposed an unverified box"
                        )
                    truth = [
                        g["bbox"]
                        for g in annotation["instances"]
                        if g["gate_id"] == str(gate_id)
                    ]
                    selected_relation = gate_box_relation(
                        selected.box if selected else None, annotation, gate_id
                    )
                    selected_requested_gate[gate_id] += selected_relation == "target"
                    # A positive remains a localization miss if a different box was
                    # confirmed. For absent IDs, ignore unknown screens only.
                    if (
                        not truth
                        and annotation["ignore_boxes"]
                        and (confirmed is None or selected_relation == "ambiguous")
                    ):
                        unscored[gate_id] += 1
                        continue
                    scores.add(confirmed, truth, image.size)
                    numbered_all.add(confirmed, truth, image.size)
                    if confirmed and selected_relation == "other_gate":
                        false_confirmation_on_other_gate[gate_id] += 1
    generic = {}
    for category in CATEGORIES:
        score = detections[category].summary()
        generic[category] = {
            "detection": score,
            "localization_by_distance": dict(sorted(by_distance[category].items())),
            "all_fields_and_box_correct_on_positives": joint_correct[category],
            "all_fields_and_box_recall": ratio(
                joint_correct[category], score["positives"]
            ),
            "metadata_on_correct_detections": {
                f: s.summary() for f, s in metadata[category].items()
            },
            "metadata_on_ground_truth_boxes": {
                f: s.summary() for f, s in oracle[category].items()
            },
        }
    artifacts = sorted(p for p in HERE.rglob("*.onnx") if "models" in p.parts)
    summary = dict(
        test_video=TEST_VIDEO,
        annotated_source_frames=len(dataset["frames"]),
        generic=generic,
        numbered_gate=dict(
            all_queries=numbered_all.summary(),
            requested_gate_selected_before_reading=sum(
                selected_requested_gate.values()
            ),
            false_confirmations=numbered_all.counts["false_positive"],
            confirmed_wrong_known_gate=sum(false_confirmation_on_other_gate.values()),
            oracle_reader_on_readable_boxes=gate_id_on_known_boxes.summary(),
            oracle_reader_on_unreadable_boxes=gate_id_on_ambiguous_boxes.summary(),
            unscored_ambiguous_queries=sum(unscored.values()),
            by_requested_id={
                str(i): dict(
                    s.summary(),
                    requested_gate_selected_before_reading=selected_requested_gate[i],
                    false_confirmations=s.counts["false_positive"],
                    confirmed_wrong_known_gate=false_confirmation_on_other_gate[i],
                    unscored=unscored[i],
                )
                for i, s in numbered.items()
            },
        ),
        model_sha256={str(p.relative_to(HERE)): file_hash(p) for p in artifacts},
        test_labels_sha256=file_hash(labels_path),
    )
    return summary, per_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=HERE.parent / "test_data" / "vision",
        help="Held-out vision fixture directory",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Also run the temporal centering policy on the held-out MOV at 5 Hz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optionally write the machine-readable metrics to JSON",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Optionally write each generic VisionResult to JSON",
    )
    parser.add_argument(
        "--timings",
        type=Path,
        help="Write host CPU timings separately from accuracy metrics",
    )
    args = parser.parse_args()
    timings = defaultdict(list) if args.timings else None
    model = VisionModel()
    summary, predictions = evaluate(args.data, model, timings)
    if timings is not None:
        import numpy as np

        measured = {}
        for stage, values in timings.items():
            warm = values[1:]
            if warm:
                measured[stage] = dict(
                    n=len(warm),
                    p50_ms=round(float(np.median(warm)), 2),
                    p95_ms=round(float(np.percentile(warm, 95)), 2),
                )
        args.timings.write_text(json.dumps(measured, indent=2) + "\n")
    if args.replay:
        from replay import replay

        summary["video_replay"] = replay(args.data, model)
    rendered = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.write_text(rendered)
    if args.predictions:
        args.predictions.write_text(json.dumps(predictions, indent=2) + "\n")


if __name__ == "__main__":
    main()
