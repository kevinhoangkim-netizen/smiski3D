"""Independent, local COCO person detection with pinned Apache-2.0 YOLOX weights.

Only model installation uses the network. Inference accepts in-memory PIL images
and runs on CPU. Detector scores are model scores, not calibrated probabilities.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from threading import Lock

import numpy as np

REPOSITORY = 'amd/yolox-s'
REVISION = '7c14fb63e32a65d92d173b2119790442f6b2bfc7'
FILENAME = 'yolox-s-int8.onnx'
SHA256 = 'f779442685eb8e050df39318149aa16afd3fdd05644ea7ec4da4aae672e6977a'
MODEL_PATH = Path(os.environ.get('PEOPLE_MODEL_PATH', str(Path(__file__).resolve().parents[2] / '.local/people' / FILENAME)))
MODEL_NAME = 'YOLOX-S / COCO (local CPU)'
_lock = Lock()
_session = None
_verified = None


def _verify():
    global _verified
    if not MODEL_PATH.is_file():
        return False, 'Group detection model is not installed. Restart the launcher to install it.'
    stat = MODEL_PATH.stat()
    signature = (str(MODEL_PATH), stat.st_size, stat.st_mtime_ns)
    if signature != _verified:
        with MODEL_PATH.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != SHA256:
            return False, 'Group detection model checksum does not match. Reinstall the verified model.'
        _verified = signature
    return True, 'Independent person detection runs locally; photos never leave this computer.'


def available():
    ok, detail = _verify()
    if not ok:
        return False, detail
    try:
        import onnxruntime  # noqa: F401
    except (ImportError, OSError) as exc:
        return False, f'Group detection runtime is unavailable ({type(exc).__name__}). Restart the launcher to repair it.'
    return True, detail


def status():
    ready, detail = available()
    return dict(available=ready, model=MODEL_NAME, detail=detail)


def _runtime():
    global _session
    ok, detail = available()
    if not ok:
        raise RuntimeError(detail)
    if _session is None:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        _session = ort.InferenceSession(str(MODEL_PATH), sess_options=options,
                                       providers=['CPUExecutionProvider'])
    return _session


def _iou(first, second):
    a, b = np.asarray(first, float), np.asarray(second, float)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:] - a[:2]).prod() + np.maximum(0, b[2:] - b[:2]).prod() - intersection
    return float(intersection / union) if union > 0 else 0.


def _duplicate(first, second, threshold=.55):
    if _iou(first, second) > threshold:
        return True
    # Window boundaries sometimes yield a narrow vertical sliver of the same
    # adult. Suppress near-contained boxes of the same height, but preserve a
    # shorter child in front of a taller adult.
    a, b = np.asarray(first, float), np.asarray(second, float)
    dimensions_a, dimensions_b = a[2:] - a[:2], b[2:] - b[:2]
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    smaller = min(dimensions_a.prod(), dimensions_b.prod())
    same_height = min(dimensions_a[1], dimensions_b[1]) / max(dimensions_a[1], dimensions_b[1], .001) > .85
    return bool(smaller > 0 and intersection / smaller > .88 and same_height)


def merge(detections, threshold=.55):
    """NMS deduplicates overlapping windows without merging adjacent people."""
    result = []
    for candidate in sorted(detections, key=lambda entry: entry['score'], reverse=True):
        if not any(_duplicate(candidate['bbox'], kept['bbox'], threshold) for kept in result):
            result.append(candidate)
    return sorted(result, key=lambda entry: (entry['bbox'][0] + entry['bbox'][2]) / 2)


def _decode(outputs, scale, offset, full_size, threshold):
    """AMD export exposes three NHWC raw heads; decode COCO class 0 only."""
    found = []
    width, height = full_size
    for output, stride in zip(outputs, (8, 16, 32)):
        prediction = output[0].reshape(-1, 85).copy()
        rows, columns = output.shape[1:3]
        y_grid, x_grid = np.meshgrid(np.arange(rows), np.arange(columns), indexing='ij')
        prediction[:, 4:] = 1 / (1 + np.exp(-np.clip(prediction[:, 4:], -80, 80)))
        scores = prediction[:, 4] * prediction[:, 5]
        selected = scores >= threshold
        if not selected.any():
            continue
        prediction = prediction[selected]
        grid = np.stack((x_grid.ravel(), y_grid.ravel()), axis=1)[selected]
        center = (prediction[:, :2] + grid) * stride / scale
        dimensions = np.exp(np.clip(prediction[:, 2:4], -20, 20)) * stride / scale
        lower = center - dimensions / 2 + np.asarray(offset)
        upper = center + dimensions / 2 + np.asarray(offset)
        for start, end, score in zip(lower, upper, scores[selected]):
            bbox = np.clip(np.r_[start / (width, height), end / (width, height)], 0, 1).tolist()
            # Discard degenerate boxes and objectness for non-person classes.
            if (bbox[2] - bbox[0]) * width >= 12 and (bbox[3] - bbox[1]) * height >= 24:
                found.append(dict(bbox=bbox, score=float(score), source='person-detector'))
    return found


def _detect_crop(session, image, crop, threshold):
    import cv2
    left, top, right, bottom = crop
    pixels = np.asarray(image.crop(crop).convert('RGB'))[:, :, ::-1]  # YOLOX uses BGR, raw 0..255.
    scale = min(640 / pixels.shape[0], 640 / pixels.shape[1])
    resized = cv2.resize(pixels, (int(pixels.shape[1] * scale), int(pixels.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)
    padded = np.full((640, 640, 3), 114, dtype=np.float32)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    outputs = session.run(None, {session.get_inputs()[0].name: padded[None]})
    return _decode(outputs, scale, (left, top), image.size, threshold)


def detect(image, regions=None):
    """Find people globally and in full-height windows and optional retry areas.

    A requested retry region with no detector hit is explicitly retained as an
    unresolved region by the group pipeline, rather than fabricated as a person.
    """
    with _lock:
        session = _runtime()
        boxes = _detect_crop(session, image, (0, 0, image.width, image.height), .38)
        if image.width >= 900 and image.width / image.height >= 1.15:
            window = int(image.width * .68)
            for left in (0, image.width - window):
                window_boxes = _detect_crop(session, image, (left, 0, left + window, image.height), .48)
                # A cropped torso or arm at an internal window boundary is not
                # an additional person. The global pass still covers that area.
                boxes.extend(box for box in window_boxes
                             if not (left > 0 and box['bbox'][0] * image.width < left + 3)
                             and not (left + window < image.width and box['bbox'][2] * image.width > left + window - 3))
        for region in regions or []:
            crop = tuple(int(value * (image.width if index % 2 == 0 else image.height)) for index, value in enumerate(region))
            if crop[2] - crop[0] >= 12 and crop[3] - crop[1] >= 24:
                boxes.extend(_detect_crop(session, image, crop, .32))
    return merge(boxes)
