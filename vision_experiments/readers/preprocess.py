"""Pixel inputs shared by training and deployment. All boxes are top-left xyxy."""

import math

import numpy as np
from PIL import Image

DISTANCES = ("near", "medium", "far")
ARROWS = (None, "left", "right", "forward")
GATES = (None, "1", "2", "3", "4", "5", "6")
CROP_SIZE = (192, 96)
SCENE_SIZE = (48, 32)
DETAIL_SIZE = (48, 24)


def clipped_box(image, box):
    width, height = image.size
    x1, y1, x2, y2 = box
    box = (
        max(0, math.floor(x1)),
        max(0, math.floor(y1)),
        min(width, math.ceil(x2)),
        min(height, math.ceil(y2)),
    )
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError(f"Box has no visible area: {box}")
    return box


def pixels(image, size):
    rgb = image.convert("RGB").resize(size, Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(rgb).transpose(2, 0, 1))


def sign_pixels(image, box, category):
    left, top, right, bottom = clipped_box(image, box)
    if category == "gate":
        # The gate number is in the display header. Exclude the clock/date below it.
        bottom = min(bottom, top + max(1, math.ceil(0.67 * (bottom - top))))
    return pixels(image.crop((left, top, right, bottom)), CROP_SIZE)


def distance_inputs(image, box):
    width, height = image.size
    x1, y1, x2, y2 = clipped_box(image, box)
    w, h = x2 - x1, y2 - y1
    long_side = max(width, height)
    geometry = np.array(
        [
            np.log(w / long_side),
            np.log(h / long_side),
            np.log(w / h),
            (x1 + x2) / (2 * width),
            (y1 + y2) / (2 * height),
            w / width,
            h / height,
            width / long_side,
            height / long_side,
            x1 <= 2,
            y1 <= 2,
            x2 >= width - 2,
            y2 >= height - 2,
        ],
        dtype=np.float32,
    )
    return (
        pixels(image, SCENE_SIZE),
        pixels(image.crop((x1, y1, x2, y2)), DETAIL_SIZE),
        geometry,
    )
