"""Fine-tune one COCO-initialized YOLO11n per sign class on clean airport frames."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import strip_optimizer
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CATEGORIES = ("gate", "restaurant", "washroom")
TEST_VIDEO = "IMG_8458.MOV"
INITIAL_WEIGHTS_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
)
INITIAL_WEIGHTS_SHA256 = (
    "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
)
IMAGE_SIZE = 640


def read(path):
    return json.loads(path.read_text())


def xywh_normalized(box, width, height):
    x1, y1, x2, y2 = box
    return [
        (x1 + x2) / (2 * width),
        (y1 + y2) / (2 * height),
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]


def prepare(data, output, category):
    archive = data / ".labeling_work" / "bulk_v1"
    mapping = read(data / "annotated" / category / "ground_truth_bulk_v1.json")
    rows = [r for r in mapping.values() if r["source_video"] != TEST_VIDEO]
    gates = (
        read(data / "annotations/gate_instances.json")["videos"]
        if category == "gate"
        else {}
    )
    original = defaultdict(list)
    if category != "gate":
        for video in sorted({r["source_video"] for r in rows}):
            published = read(archive / "published" / (Path(video).stem + ".json"))
            for r in published["records"][category].values():
                if r["object_present"]:
                    original[video, r["frame_index"]].append(r["bbox"])

    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    (output / "labels").mkdir(exist_ok=True)

    def write_example(r):
        name = Path(r["file_name"]).stem
        video = r["source_video"]
        frame = r["frame_index"]
        width, height = r["width"], r["height"]
        image_path = data / "unannotated" / category / r["file_name"]
        if category == "gate":
            annotations = gates[video][str(frame)]
            positive = [i["bbox"] for i in annotations["instances"]]
            ignored = annotations["ignore_boxes"]
        else:
            positive = [r["bbox"]] if r["object_present"] else []
            ignored = [b for b in original[video, frame] if b not in positive]
        with Image.open(image_path) as im:
            if im.size != (width, height):
                raise ValueError(f"Source image dimensions disagree for {image_path}")
            im.thumbnail((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
            im.convert("RGB").save(
                output / "images" / (name + ".jpg"), quality=97, subsampling=0
            )
        boxes = [xywh_normalized(b, width, height) for b in positive]
        text = "".join("0 " + " ".join(f"{v:.9f}" for v in b) + "\n" for b in boxes)
        (output / "labels" / (name + ".txt")).write_text(text)
        return name, [xywh_normalized(b, width, height) for b in ignored]

    with ThreadPoolExecutor(max_workers=6) as pool:
        ignored = {
            name: boxes for name, boxes in pool.map(write_example, rows) if boxes
        }
    (output / "ignores.json").write_text(json.dumps(ignored))
    # Ultralytics requires both paths even when validation is disabled. Only training data is exposed.
    config = {
        "path": str(output),
        "train": "images",
        "val": "images",
        "names": {0: category},
    }
    (output / "dataset.yaml").write_text(yaml.safe_dump(config))


class IgnoreRegionBCE(torch.nn.BCEWithLogitsLoss):
    """Suppress background gradients at anchor centers inside discarded same-class boxes."""

    mask = None

    def forward(self, predictions, targets):
        loss = super().forward(predictions, targets)
        if self.mask is not None:
            loss = loss.masked_fill(self.mask & (targets == 0), 0)
        return loss


class IgnoreRegionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.bce = IgnoreRegionBCE(reduction="none")

    def __call__(self, preds, batch):
        category = batch["cls"].flatten()
        ignored = category < 0
        features = preds[1] if isinstance(preds, tuple) else preds
        self.bce.mask = None
        if ignored.any():
            device = features[0].device
            anchors, strides = make_anchors(features, self.stride, 0.5)
            centers = anchors * strides
            scale = torch.tensor(batch["img"].shape[-2:][::-1] * 2, device=device)
            ignore_targets = torch.cat(
                (batch["batch_idx"].view(-1, 1), batch["cls"], batch["bboxes"]), 1
            )[ignored]
            boxes = self.preprocess(ignore_targets, features[0].shape[0], scale)[
                :, :, 1:5
            ]
            point = centers[None, :, None, :]
            mask = (
                ((point >= boxes[:, None, :, :2]) & (point < boxes[:, None, :, 2:]))
                .all(-1)
                .any(-1)
            )
            self.bce.mask = mask.unsqueeze(-1)
            batch = {
                **batch,
                **{k: batch[k][~ignored] for k in ("batch_idx", "cls", "bboxes")},
            }
        return super().__call__(preds, batch)


class AirportTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        if mode == "train":
            ignores = read(Path(img_path).parent / "ignores.json")
            for label in dataset.labels:
                rows = ignores.get(Path(label["im_file"]).stem)
                if rows:
                    # Add after Ultralytics' one-class input validation; these rows follow
                    # ordinary augmentation geometry and are removed before assignment.
                    extra = np.asarray(rows, dtype=np.float32)
                    label["bboxes"] = np.concatenate((label["bboxes"], extra))
                    label["cls"] = np.concatenate(
                        (label["cls"], -np.ones((len(extra), 1), dtype=np.float32))
                    )
        return dataset

    def validate(self):
        return {}, 0.0

    def final_eval(self):
        strip_optimizer(self.last)


def initialize_loss(trainer):
    trainer.model.criterion = IgnoreRegionLoss(trainer.model)


def backbone_warmup(trainer):
    if trainer.epoch in (0, 2):
        for block in trainer.model.model[:11]:
            block.requires_grad_(trainer.epoch >= 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data")
    parser.add_argument(
        "--work", type=Path, default=REPO / "data/.vision_training/detection"
    )
    parser.add_argument("--category", required=True, choices=CATEGORIES)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    args.work = args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    weights = args.weights or args.work / "yolo11n.pt"
    if not weights.exists():
        urllib.request.urlretrieve(INITIAL_WEIGHTS_URL, weights)
    if hashlib.sha256(weights.read_bytes()).hexdigest() != INITIAL_WEIGHTS_SHA256:
        raise ValueError("Unexpected SHA-256 for the pretrained COCO weights")
    # Use a new cache when curated labels or the supplemental audit change.
    # The per-video checksums in the active manifest identify the original audit
    # records supplying ignored boxes; its mapping checksums cover each class.
    snapshot = hashlib.sha256((args.data / "annotated" / "bulk_v1_manifest.json").read_bytes())
    snapshot.update((args.data / "annotated" / args.category / "ground_truth_bulk_v1.json").read_bytes())
    if args.category == "gate":
        snapshot.update((args.data / "annotations/gate_instances.json").read_bytes())
    dataset = args.work / "datasets" / f"{args.category}-{snapshot.hexdigest()[:12]}"
    if not (dataset / "dataset.yaml").exists():
        prepare(args.data.resolve(), dataset, args.category)
    # Cached training files can move with the repository.
    config = {"path": str(dataset), "train": "images", "val": "images", "names": {0: args.category}}
    (dataset / "dataset.yaml").write_text(yaml.safe_dump(config))
    model = YOLO(weights)
    model.add_callback("on_train_start", initialize_loss)
    model.add_callback("on_train_epoch_start", backbone_warmup)
    model.train(
        trainer=AirportTrainer,
        data=str(dataset / "dataset.yaml"),
        device=args.device,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=IMAGE_SIZE,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.08,
        cos_lr=True,
        warmup_epochs=1.5,
        warmup_bias_lr=0.001,
        momentum=0.9,
        weight_decay=0.0005,
        seed=42,
        deterministic=True,
        degrees=3.0,
        translate=0.08,
        scale=0.28,
        shear=1.5,
        fliplr=0,
        flipud=0,
        mosaic=0.25,
        close_mosaic=4,
        hsv_h=0.01,
        hsv_s=0.35,
        hsv_v=0.35,
        val=False,
        plots=False,
        workers=3,
        cache="ram",
        amp=False,
        project=str(args.work / "runs"),
        name=args.category,
        exist_ok=True,
        save=True,
    )
    from export import export_models

    export_models(
        args.work / "runs" / args.category / "weights/last.pt",
        HERE / "models",
        args.category,
    )


if __name__ == "__main__":
    main()
