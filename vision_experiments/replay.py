"""Replay the held-out MOV through the actual centering state machine at 5 Hz."""

from collections import Counter
import json
from uuid import UUID

from PIL import Image

from vision import Query, VisionSession, overlap


def gate_box_relation(box, annotation, requested_gate):
    """Label one proposed box by its overlap with reviewed gate regions."""
    if box is None:
        return "none"
    if any(
        g["gate_id"] == str(requested_gate) and overlap(box, g["bbox"]) >= 0.5
        for g in annotation["instances"]
    ):
        return "target"
    if any(overlap(box, g["bbox"]) >= 0.5 for g in annotation["instances"]):
        return "other_gate"
    if any(overlap(box, b) >= 0.5 for b in annotation["ignore_boxes"]):
        return "ambiguous"
    return "background_or_mislocalized"


def numbered_counts():
    return Counter(
        dict.fromkeys(
            (
                "provisional_guides",
                "confirmed_frames",
                "fresh_confirmations",
                "cached_confirmations",
                "suppressed_wrong_id_frames",
                "suppressed_unreadable_frames",
                "slow_wrong_id_reads",
                "slow_unreadable_reads",
                "annotated_frames_processed",
                "annotated_correct_confirmations",
                "annotated_wrong_gate_confirmations",
                "annotated_background_or_mislocalized_confirmations",
                "annotated_ambiguous_confirmations",
                "annotated_correct_target_suppressed_on_read",
            ),
            0,
        )
    )


def audit_numbered_frame(counts, session, query, annotation):
    phase = session.phase
    refreshed = session.details_refreshed
    if phase == "guiding":
        counts["provisional_guides"] += 1
    elif phase == "verified":
        counts["confirmed_frames"] += 1
        counts["fresh_confirmations" if refreshed else "cached_confirmations"] += 1
    elif phase == "rejected":
        counts["suppressed_wrong_id_frames"] += 1
        counts["slow_wrong_id_reads"] += refreshed
    elif phase == "unreadable":
        counts["suppressed_unreadable_frames"] += 1
        counts["slow_unreadable_reads"] += refreshed

    if annotation is None:
        return
    counts["annotated_frames_processed"] += 1
    if phase == "verified":
        relation = gate_box_relation(
            session.previous_box, annotation, query.requested_gate
        )
        key = {
            "target": "annotated_correct_confirmations",
            "other_gate": "annotated_wrong_gate_confirmations",
            "background_or_mislocalized": "annotated_background_or_mislocalized_confirmations",
            "ambiguous": "annotated_ambiguous_confirmations",
        }[relation]
        counts[key] += 1
    elif refreshed and phase in ("rejected", "unreadable"):
        relation = gate_box_relation(
            session.previous_box, annotation, query.requested_gate
        )
        counts["annotated_correct_target_suppressed_on_read"] += relation == "target"


def replay(data, model):
    import cv2
    import numpy as np

    dataset = json.loads((data / "labels.json").read_text())
    video = dataset["video"]
    frames = {int(idx): entry for idx, entry in dataset["frames"].items()}
    capture = cv2.VideoCapture(str(data / video))
    if not capture.isOpened():
        raise OSError(f"Cannot open {video}")
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    fps = capture.get(cv2.CAP_PROP_FPS)
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # A regular 5 Hz schedule keeps trigger timing independent of the annotated indices.
    stride = max(1, round(fps / 5))
    cases = [
        ("washroom", None),
        ("restaurant", None),
        ("gate", None),
        ("gate", 5),
        ("gate", 6),
    ]
    streams = [
        (
            Query(c, UUID(int=i + 1), g),
            VisionSession(model),
            numbered_counts() if g else Counter(),
        )
        for i, (c, g) in enumerate(cases)
    ]
    checked = set()
    try:
        for idx in range(length):
            if not capture.grab():
                raise OSError(f"Failed to decode frame {idx}")
            process = idx % stride == 0
            if not process and idx not in frames:
                continue
            ok, bgr = capture.retrieve()
            if not ok:
                raise OSError(f"Failed to retrieve frame {idx}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if idx in frames:
                with Image.open(data / frames[idx]["image"]) as reference:
                    if not np.array_equal(rgb, np.asarray(reference)):
                        raise ValueError(
                            f"Decoded frame {idx} differs from the original annotation pixels"
                        )
                checked.add(idx)
            if not process:
                continue
            image = Image.fromarray(rgb)
            for category in ("washroom", "restaurant", "gate"):
                frame = model.frame(image, category)
                for query, session, counts in streams:
                    if query.label != category:
                        continue
                    result = session.advance(frame, query, idx / fps)
                    counts["frames_processed"] += 1
                    counts["frames_with_box"] += result["boundingBox"] is not None
                    counts["frames_with_detail_result"] += (
                        result["distanceToSign"] is not None
                    )
                    counts["slow_reader_calls"] += session.details_refreshed
                    if query.requested_gate is not None:
                        audit_numbered_frame(
                            counts,
                            session,
                            query,
                            frames[idx]["gates"] if idx in frames else None,
                        )
    finally:
        capture.release()
    if checked != set(frames):
        raise ValueError("Incomplete verification of replay frames")
    return dict(
        fps=round(fps, 6),
        source_frames=length,
        stride=stride,
        verified_pngs=len(checked),
        streams={f"{q.label}{q.requested_gate or ''}": dict(c) for q, _, c in streams},
    )
