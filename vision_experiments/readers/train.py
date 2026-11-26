"""Train and export the gated metadata readers. Run as python -m readers.train."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .networks import CropReader, DistanceReader
from .preprocess import (
    ARROWS,
    GATES,
    clipped_box,
    distance_inputs,
    pixels,
    sign_pixels,
    CROP_SIZE,
)

HERE = Path(__file__).resolve().parent
HELD_OUT = "IMG_8458.MOV"
SEED = 1773


def jitter(image, bbox, rng, amount):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return clipped_box(
        image,
        [
            coord + rng.uniform(-amount, amount) * size
            for coord, size in zip(bbox, (w, h, w, h))
        ],
    )


def synthetic_gate(rng, gate):
    """Vary header digit, layout and optics independently to reduce scene shortcuts."""
    w, h = 265, 165
    ink = rng.randint(20, 58)
    base = Image.new("RGB", (w, h), tuple(rng.randint(3, 32) for _ in range(3)))
    draw = ImageDraw.Draw(base)
    header = rng.randint(60, 104)
    screen = (rng.randint(6, 55), rng.randint(24, 78), rng.randint(78, 180))
    draw.rectangle((3, 3, w - 4, h - 4), fill=screen)
    lum = rng.randint(155, 248)
    panel = (lum - 10, lum - 8, min(255, lum + 7))
    left = rng.randint(-8, 12)
    draw.rectangle((left, 6, rng.randint(143, 180), header), fill=panel)
    regular = "DejaVuSans.ttf"
    font = ImageFont.truetype(regular, rng.randint(23, 35))
    digit_font = ImageFont.truetype(regular, rng.randint(44, 68))
    draw.text(
        (left + 4, header * 0.5), "Gate", font=font, fill=(ink, ink, ink), anchor="lm"
    )
    draw.text(
        (rng.randint(93, 122), header * 0.5),
        gate,
        font=digit_font,
        fill=(ink, ink, ink),
        anchor="lm",
    )
    draw.text(
        (rng.randint(30, 55), header + 2),
        f"12:{rng.randrange(60):02} PM",
        font=font,
        fill=panel,
    )
    draw.text((rng.randint(177, 195), header * 0.4), "AIR", font=font, fill=panel)
    # Direct affine shear/rotation mimics perspective of a monitor seen at an angle.
    base = base.rotate(
        rng.uniform(-10, 10), Image.Resampling.BICUBIC, fillcolor=(72, 73, 78)
    )
    shear = rng.uniform(-0.15, 0.15)
    base = base.transform(
        (w, h),
        Image.Transform.AFFINE,
        (1, shear, -h * shear / 2, 0, 1, 0),
        Image.Resampling.BILINEAR,
    )
    base = base.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.5)))
    return pixels(base, CROP_SIZE)


def make_samples(data, category, gate_file):
    mapping = json.loads(
        (data / "annotated" / category / "ground_truth_bulk_v1.json").read_text()
    )
    originals = [r for r in mapping.values() if r["source_video"] != HELD_OUT]
    rng = random.Random(SEED)
    sign_x, sign_y, distance_x, distance_y = [], [], [], []
    gate_frames = {}
    if category == "gate":
        all_gates = json.loads(gate_file.read_text())["videos"]
        # Includes reviewed secondary screens and unreadable screen candidates.
        gate_frames = {
            (v, int(idx)): r
            for v, frames in all_gates.items()
            if v != HELD_OUT
            for idx, r in frames.items()
        }
    by_frame = {(r["source_video"], r["frame_index"]): r for r in originals}
    keys = sorted(set(by_frame) | set(gate_frames))
    for key in keys:
        row = by_frame.get(key)
        box_entries = []
        if category == "gate":
            annotation = gate_frames[key]
            box_entries.extend(
                (ent["bbox"], GATES.index(ent["gate_id"]))
                for ent in annotation["instances"]
            )
            box_entries.extend((box, 0) for box in annotation["ignore_boxes"])
        elif row and row["object_present"]:
            direction = row["arrow_direction"]
            cls = ARROWS.index("forward" if direction == "front" else direction)
            box_entries.append((row["bbox"], cls))
        negative = bool(
            category == "gate"
            and row
            and not row["object_present"]
            and not box_entries
            and rng.random() < 0.15
        )
        positive = bool(row and row["object_present"])
        if not box_entries and not negative and not positive:
            continue
        video, frame = key
        source = (
            data / "unannotated" / category / row["file_name"]
            if row
            else data
            / ".labeling_work/bulk_v1"
            / Path(video).stem
            / "frames"
            / f"frame_{frame:06}.png"
        )
        with Image.open(source) as f:
            image = f.convert("RGB")
        if negative:
            w, h = image.size
            bw = rng.randint(35, w // 3)
            bh = rng.randint(30, h // 3)
            x, y = rng.randrange(w - bw), rng.randrange(h - bh)
            box_entries.append(([x, y, x + bw, y + bh], 0))
        for box, cls in box_entries:
            for amount in (0, 0.035, 0.075, 0.12, 0.18):
                b = jitter(image, box, rng, amount)
                sign_x.append(sign_pixels(image, b, category))
                sign_y.append(cls)
        if positive:
            label = ("near", "mid", "far").index(row["distance"])
            for amount in (0, 0.05, 0.10, 0.16):
                b = jitter(image, row["bbox"], rng, amount)
                distance_x.append(distance_inputs(image, b))
                distance_y.append(label)
    if category == "gate":
        for cls in range(1, 7):
            for _ in range(75):
                sign_x.append(synthetic_gate(rng, str(cls)))
                sign_y.append(cls)
    print(
        category,
        "augmented sign label counts",
        dict(Counter(sign_y)),
        "augmented distance label counts",
        dict(Counter(distance_y)),
        flush=True,
    )
    sign = [np.stack(sign_x)]
    distance = [np.stack(a) for a in zip(*distance_x)]
    return (sign, np.array(sign_y)), (distance, np.array(distance_y))


def batches(inputs, labels, batch_size):
    data = TensorDataset(
        *(torch.from_numpy(x) for x in inputs), torch.from_numpy(labels).long()
    )
    freq = Counter(labels.tolist())
    weights = [freq[int(y)] ** (-0.65) for y in labels]
    sampler = WeightedRandomSampler(
        weights, num_samples=min(len(labels), 1800), replacement=True
    )
    return DataLoader(
        data, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=False
    )


def fit(category, task, arrays, device, root, epochs):
    features, labels = arrays
    is_sign = task == "sign"
    num_classes = (7 if category == "gate" else 4) if is_sign else 3
    model = CropReader(num_classes) if is_sign else DistanceReader()
    model.to(device)
    if is_sign:
        for p in model.features[:3].parameters():
            p.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.features.parameters(), "lr": 0.00045},
                {"params": model.readout.parameters(), "lr": 0.0015},
            ],
            weight_decay=0.004,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0018, weight_decay=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=0.000035
    )
    loader = batches(features, labels, 40 if is_sign else 80)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.015)
    for epoch in range(epochs):
        model.train()
        if is_sign:
            model.features[:3].eval()
        seen = correct = 0
        for *raw, y in loader:
            xs = [
                (
                    x.to(device, dtype=torch.float32) / 255.0
                    if x.dtype == torch.uint8
                    else x.to(device)
                )
                for x in raw
            ]
            y = y.to(device)
            # Apply exposure jitter consistently across a sample's RGB views.
            gain = torch.rand(len(y), 1, 1, 1, device=device) * 0.42 + 0.79
            bias = torch.rand(len(y), 1, 1, 1, device=device) * 0.12 - 0.06
            for i, x in enumerate(xs):
                if x.ndim == 4:
                    xs[i] = (x * gain + bias).clamp(0, 1)
            if is_sign and category != "gate":
                flip = torch.rand(len(y), device=device) < 0.45
                xs[0][flip] = xs[0][flip].flip(-1)
                y = torch.where(flip & ((y == 1) | (y == 2)), 3 - y, y)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*xs)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            seen += y.numel()
            correct += (logits.argmax(1) == y).sum().item()
        scheduler.step()
        if epoch in (0, epochs // 2, epochs - 1):
            print(
                f"{category} {task} epoch {epoch + 1}/{epochs} sampled training accuracy {correct / seen:.3f}",
                flush=True,
            )
    model = model.eval().cpu()
    names = ("crop",) if is_sign else ("scene", "detail", "geometry")
    example = tuple(
        torch.from_numpy(x[:1]).float() / (255 if x.dtype == np.uint8 else 1)
        for x in features
    )
    export(model, example, names, root / f"{category}_{task}")


def export(model, example, names, path):
    import onnxruntime as ort

    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            example,
            str(path.with_suffix(".onnx")),
            opset_version=17,
            do_constant_folding=True,
            input_names=list(names),
            output_names=["logits"],
            dynamo=False,
        )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        session = ort.InferenceSession(
            str(path.with_suffix(".onnx")), opts, providers=["CPUExecutionProvider"]
        )
        result = session.run(None, {n: x.numpy() for n, x in zip(names, example)})[0]
        np.testing.assert_allclose(
            result, model(*example).numpy(), atol=1e-4, rtol=1e-4
        )
        print(f"ONNX parity passed: {path.name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=HERE.parents[1] / "data")
    parser.add_argument("--output", type=Path, default=HERE / "models")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["washroom", "restaurant", "gate"],
        choices=["washroom", "restaurant", "gate"],
    )
    parser.add_argument(
        "--gate-annotations",
        type=Path,
        help="Defaults to annotations/gate_instances.json inside --data",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    torch.set_num_threads(3)
    for category in args.categories:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        sign, distance = make_samples(
            args.data,
            category,
            args.gate_annotations or args.data / "annotations/gate_instances.json",
        )
        fit(category, "sign", sign, args.device, args.output, 22)
        fit(category, "distance", distance, args.device, args.output, 48)


if __name__ == "__main__":
    main()
