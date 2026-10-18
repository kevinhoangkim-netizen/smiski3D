"""Local clothing regions and measurements from SCHP ATR-18 segmentation.

Eighteen pixel classes identify broad garment regions. Visible limb coverage
is measured from those regions; the independent CLIP recognizer adds fine cues
such as suit jackets and bouquets. This does not recover fine garment patterns.
Inference has no networking and model installation is an explicit setup step.
"""
from __future__ import annotations

import atexit
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image

MODEL_DIR = Path(os.environ.get('GARMENT_MODEL_DIR', str(Path(__file__).resolve().parents[2] / '.local/garment')))
MODEL_FILE = 'schp-atr-18-int8-static.onnx'
MODEL_SHA256 = '4420d8db8c1f266967c89485786b01209f6d405f320fc0f87e8ced49392cefb5'
MODEL_NAME = 'SCHP ATR-18 · local garment regions and measurements'
_runtime = None
_lock = Lock()


def available():
    if not (MODEL_DIR / MODEL_FILE).is_file() or not (MODEL_DIR / 'manifest.json').is_file():
        return False, 'Photo clothing measurements are not installed. Run scripts/download_garment_model.py.'
    if importlib.util.find_spec('onnxruntime') is None:
        return False, 'Photo clothing measurements need the app Python dependencies.'
    try:
        manifest = json.loads((MODEL_DIR / 'manifest.json').read_text())
        if manifest['files'].get(MODEL_FILE) != MODEL_SHA256:
            return False, 'The clothing measurement manifest is invalid. Rerun scripts/download_garment_model.py.'
    except (OSError, ValueError, KeyError, TypeError):
        return False, 'The clothing measurement manifest is invalid. Rerun scripts/download_garment_model.py.'
    return True, 'Local clothes/skin segmentation measures visible garment coverage. Hidden details remain estimates.'


def status():
    ready, detail = available()
    return dict(available=ready, model=MODEL_NAME, detail=detail)


def default_garment(*, inferred=False):
    return dict(upper='unknown', lower='unknown', sleeve_length=.5,
                hem_length=.25, skirt_flare=.35, trouser_length=1.,
                headwear='none', inferred=inferred)


def person_box(image, subject):
    """A modest per-person crop; line samples further exclude nearby people."""
    left, top, right, bottom = subject['bbox']
    width, height = max(right - left, .04), max(bottom - top, .08)
    left, right = left - width * .16, right + width * .16
    top, bottom = top - height * .13, bottom + height * .06
    return (max(0, int(left * image.width)), max(0, int(top * image.height)),
            min(image.width, max(1, int(right * image.width))),
            min(image.height, max(1, int(bottom * image.height))))


def _disc_ratio(mask, center, radius, labels):
    """Local evidence, never a fraction of somebody else's whole bounding box."""
    x, y = center
    left, right = max(0, int(x - radius)), min(mask.shape[1], int(x + radius + 1))
    top, bottom = max(0, int(y - radius)), min(mask.shape[0], int(y + radius + 1))
    if left >= right or top >= bottom:
        return 0.
    yy, xx = np.ogrid[top:bottom, left:right]
    inside = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    if not inside.any():
        return 0.
    return float(np.isin(mask[top:bottom, left:right][inside], labels).mean())


def _coverage(mask, points, indexes, radius, label=4):
    # Continuous shoulder→elbow→wrist or hip→knee→ankle parameterization.
    coords = np.array([points[k][:2] for k in indexes])
    lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    if min(points[k][2] for k in indexes) < .5 or lengths.sum() < 12:
        return None
    distances = np.r_[0., np.cumsum(lengths)]
    samples, support, skin = [], [], []
    for fraction in np.linspace(.04, .97, 25):
        distance = fraction * distances[-1]
        segment = min(int(np.searchsorted(distances[1:], distance)), len(lengths) - 1)
        alpha = (distance - distances[segment]) / max(lengths[segment], 1e-6)
        location = coords[segment] * (1 - alpha) + coords[segment + 1] * alpha
        samples.append(_disc_ratio(mask, location, radius, [label]))
        support.append(_disc_ratio(mask, location, radius, [2, 4]))
        skin.append(_disc_ratio(mask, location, radius, [2]))
    # Background is an abstention, not bare skin. This distinction matters for
    # full-body/distant photos outside a selfie model's strongest use case.
    if np.mean(support) < .25 or max(support) < .45:
        return None
    # A few isolated colored pixels (jewelry, overlap, classifier noise) do not
    # extend a sleeve. Require a short run and start near the limb's root.
    clothing = np.array(samples) >= .4
    clothing = np.convolve(clothing.astype(int), np.ones(3, dtype=int), mode='same') >= 2
    if clothing[:5].sum() < 2:
        # Pose roots can lie on a hand, bag or nearby person. A missing cuff at
        # the root does not establish a bare limb unless skin is actually seen.
        return 0. if np.mean(skin) > .6 else None
    end = 0
    gaps = 0
    for index, present in enumerate(clothing):
        gaps = 0 if present else gaps + 1
        if gaps > 3:
            break
        if present:
            end = index
    coverage = float(np.clip((end + 1) / len(clothing), 0., 1.))
    if coverage < .85 and np.mean(skin[min(end + 2, len(skin) - 1):]) < .3:
        # A limb disappearing into background/occlusion does not establish a
        # clothing hem. Visible bare skin beyond it is independent evidence.
        return None
    return coverage


def measurements(mask, subject, image_size, crop_box):
    """Pure geometry/mask function, tested independently of neural inference."""
    mask = np.asarray(mask).squeeze()
    if mask.ndim != 2:
        raise ValueError('Expected a two-dimensional clothing mask')
    width, height = image_size
    left, top, right, bottom = crop_box
    points = [(p['x'] * width - left, p['y'] * height - top, p['visibility'])
              for p in subject['landmarks']]
    shoulder_width = np.linalg.norm(np.subtract(points[11][:2], points[12][:2]))
    radius = float(np.clip(shoulder_width * .1, 2, 15))
    torso = np.mean(np.array([points[k][:2] for k in (11, 12, 23, 24)]), axis=0)
    torso_clothes = _disc_ratio(mask, torso, max(radius * 2, 4), [4])
    torso_skin = _disc_ratio(mask, torso, max(radius * 2, 4), [2])
    sleeve = [_coverage(mask, points, indexes, radius) for indexes in ((11, 13, 15), (12, 14, 16))]
    trouser = [_coverage(mask, points, indexes, radius * 1.2) for indexes in ((23, 25, 27), (24, 26, 28))]
    visible_sleeves = [value for value in sleeve if value is not None]
    visible_legs = [value for value in trouser if value is not None]
    # Sample the central hip→ankle axis for a skirt/dress hem, avoiding the head,
    # torso, distant bystanders and most coats belonging to neighboring people.
    hip = np.mean([points[k][:2] for k in (23, 24)], axis=0)
    ankle = np.mean([points[k][:2] for k in (27, 28)], axis=0)
    knee = np.mean([points[k][:2] for k in (25, 26)], axis=0)
    virtual = list(points) + [(*hip, min(points[23][2], points[24][2])),
                              (*knee, min(points[25][2], points[26][2])),
                              (*ankle, min(points[27][2], points[28][2]))]
    hem = _coverage(mask, virtual, (33, 34, 35), radius * 1.5)
    # Width around the visible hem relative to hip width; bounded and only used
    # after an independent skirt/dress classification.
    flare = None
    if hem is not None and hem > .15 and np.linalg.norm(ankle - hip) > 15:
        location = hip * (1 - hem * .8) + ankle * (hem * .8)
        row = int(np.clip(location[1], 0, mask.shape[0] - 1))
        x = int(np.clip(location[0], 0, mask.shape[1] - 1))
        reach = max(5, int(shoulder_width * 1.3))
        lo, hi = max(0, x - reach), min(mask.shape[1], x + reach)
        xs = np.flatnonzero(mask[row, lo:hi] == 4) + lo
        if len(xs) >= 5:
            cloth_width = np.percentile(xs, 95) - np.percentile(xs, 5)
            hip_width = max(np.linalg.norm(np.subtract(points[23][:2], points[24][:2])), shoulder_width * .65, 8)
            flare = float(np.clip((cloth_width / hip_width - 1) / 2, 0, 1))
    warnings = []
    if not visible_sleeves:
        warnings.append('Arms are hidden or too small to measure sleeve length.')
    elif len(visible_sleeves) == 2 and max(visible_sleeves) - min(visible_sleeves) > .3:
        warnings.append('Sleeve coverage differs across arms; the shared sleeve length is an estimate.')
    if not visible_legs:
        warnings.append('Legs are hidden or too small to measure garment length.')
    return dict(torso_clothes=torso_clothes, torso_skin=torso_skin,
                sleeve_length=float(max(visible_sleeves)) if visible_sleeves else None,
                trouser_length=float(max(visible_legs)) if visible_legs else None,
                hem_length=hem, skirt_flare=flare, warnings=warnings,
                visible_clothes=bool(torso_clothes > .35 or any(value > .18 for value in visible_legs)))


def semantic_evidence(labels, subject, image_size, box):
    """Broad categories belong to the person's anatomical region, not the group."""
    width, height = image_size
    left, top, right, bottom = box
    points = np.array([[p['x'] * width - left, p['y'] * height - top]
                       for p in subject['landmarks']])
    shoulder = (points[11] + points[12]) / 2
    hip = (points[23] + points[24]) / 2
    ankle = (points[27] + points[28]) / 2
    scale = max(np.linalg.norm(points[11] - points[12]), 12)
    yy, xx = np.indices(labels.shape)
    coords = np.stack([xx, yy], axis=-1)
    def corridor(start, end, radius):
        segment = end - start
        alpha = np.clip(np.sum((coords - start) * segment, axis=-1) / max(float(segment @ segment), 1), 0, 1)
        projected = start + alpha[:, :, None] * segment
        return np.sum((coords - projected) ** 2, axis=-1) < radius ** 2
    upper_region = corridor(shoulder, hip, scale * .48)
    lower_region = corridor(hip, ankle, scale * .7)
    def fractions(region, names):
        total = max(int(region.sum()), 1)
        return {name: float(np.logical_and(region, labels == label).sum() / total) for name, label in names.items()}
    upper = fractions(upper_region, {'shirt':4, 'dress':7})
    lower = fractions(lower_region, {'pants':6, 'skirt':5, 'dress':7})
    upper_label = max(upper, key=upper.get)
    lower_label = max(lower, key=lower.get)
    # Segmentation predicts pixel classes, not confidence probabilities. Require
    # a material anatomical area, and keep unknown if most of it is obscured.
    if upper[upper_label] < .18:
        upper_label = 'unknown'
    if lower[lower_label] < .12:
        lower_label = 'unknown'
    head = points[0]
    head_region = (xx - head[0]) ** 2 + (yy - head[1]) ** 2 < (scale * .75) ** 2
    neck_region = corridor(head * .35 + shoulder * .65, shoulder, scale * .45)
    hat = int(np.logical_and(head_region, labels == 1).sum()) > max(15, head_region.sum() * .035)
    glasses = int(np.logical_and(head_region, labels == 3).sum()) > max(8, head_region.sum() * .012)
    scarf = int(np.logical_and(neck_region, labels == 17).sum()) > max(20, neck_region.sum() * .08)
    bag_pixels = np.argwhere(labels == 16)
    bag = len(bag_pixels) > max(35, labels.size * .004)
    bag_hand = None
    if bag:
        center = bag_pixels[:, ::-1].mean(axis=0)
        bag_hand = 'left' if np.linalg.norm(center - points[15]) < np.linalg.norm(center - points[16]) else 'right'
    return dict(upper=upper_label, lower=lower_label, upper_fractions=upper,
                lower_fractions=lower, hat=bool(hat), glasses=bool(glasses), scarf=bool(scarf),
                bag=bool(bag), bag_hand=bag_hand)


class GarmentRuntime:
    def __init__(self):
        import onnxruntime as ort
        path = MODEL_DIR / MODEL_FILE
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != MODEL_SHA256:
                raise ValueError('Clothing parser checksum mismatch; rerun scripts/download_garment_model.py')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])

    def parse(self, image, subject):
        box = person_box(image, subject)
        crop = image.crop(box).convert('RGB')
        logits = self.session.run(['logits'], {'pixel_values': prepare_pixels(crop)})[0][0]
        labels = np.argmax(logits, axis=0).astype(np.uint8)
        side = max(crop.size)
        offset = ((side - crop.width) // 2, (side - crop.height) // 2)
        labels = np.asarray(Image.fromarray(labels).resize((side, side), Image.Resampling.NEAREST))
        labels = labels[offset[1]:offset[1] + crop.height, offset[0]:offset[0] + crop.width]
        # Keep the generic measurement function independent of model taxonomy.
        mask = np.zeros_like(labels)
        mask[np.isin(labels, [4, 5, 6, 7])] = 4
        mask[np.isin(labels, [12, 13, 14, 15])] = 2
        result = measurements(mask, subject, image.size, box)
        result['semantics'] = semantic_evidence(labels, subject, image.size, box)
        return result

    def close(self):
        self.session = None


def prepare_pixels(image):
    # Original SCHP inference uses cv2.imread (BGR). The conversion's helper
    # leaves RGB unreversed despite publishing BGR mean/std; real color-photo
    # comparisons show that reversals matter (blue clothes otherwise become
    # skin and bare legs become pants). Follow the trained model's convention.
    image = image.convert('RGB')
    side = max(image.size)
    canvas = Image.new('RGB', (side, side), (0, 0, 0))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    pixels = np.asarray(canvas.resize((512, 512), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    pixels = pixels[:, :, ::-1].copy()
    pixels = (pixels - np.array([.406, .456, .485], dtype=np.float32)) / np.array([.225, .224, .229], dtype=np.float32)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])


def analyze_person(image, subject):
    global _runtime
    ready, detail = available()
    if not ready:
        return None, dict(status='unavailable', model=MODEL_NAME, warnings=[detail])
    try:
        with _lock:
            if _runtime is None:
                _runtime = GarmentRuntime()
            result = _runtime.parse(image, subject)
        return result, dict(status='ready', model=MODEL_NAME, warnings=result['warnings'])
    except Exception:
        # The independent clothing recognizer may still supply broad categories,
        # but the caller must not report lengths as measured in this case.
        return None, dict(status='unavailable', model=MODEL_NAME,
                          warnings=['Local clothing measurements failed; visible garment lengths could not be measured.'])


def close():
    global _runtime
    with _lock:
        if _runtime is not None:
            runtime, _runtime = _runtime, None
            runtime.close()


atexit.register(close)
