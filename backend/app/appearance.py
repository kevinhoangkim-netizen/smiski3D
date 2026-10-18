"""Optional local, zero-shot clothing recognition using quantized CLIP.

No photo bytes are sent over the network. Downloading weights is an explicit setup
step; inference never downloads anything. CLIP is a similarity model, not an
object detector, so uncertain crops remain unset and are labelled for review.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image, ImageOps

MODEL_DIR = Path(os.environ.get('APPEARANCE_MODEL_DIR', str(Path(__file__).resolve().parents[2] / '.local/appearance')))
MODEL_NAME = 'CLIP ViT-B/32 · local quantized ONNX'
_lock = Lock()
_runtime = None
logger = logging.getLogger(__name__)

# Descriptions include negative alternatives. A winner must pass both an absolute
# cosine threshold and a margin over the runner-up; no softmax is called a probability.
LABELS = {
    'outfit': {
        'none': ['a person with a bare chest', 'a shirtless person'],
        'casual': ['a person wearing casual everyday clothes', 'a person wearing athletic sportswear and shorts'],
        'suit': ['a person wearing a formal suit jacket and trousers', 'a person dressed in a tuxedo'],
        'wedding-dress': ['a bride wearing a white wedding dress', 'a person wearing a bridal wedding gown'],
        'other': ['a person wearing a regular dress', 'a person wearing a coat or costume'],
    },
    'hat': {
        'none': ['a person with uncovered hair', 'a bareheaded person'],
        'cap': ['a person wearing a baseball cap', 'a head wearing a peaked sports cap'],
        'top-hat': ['a person wearing a tall formal top hat', 'a head wearing a black top hat'],
        'wide-brim': ['a person wearing a wide brim sun hat', 'a person wearing a broad brimmed hat'],
        'other': ['a person wearing a hood', 'a person wearing a helmet or a head scarf'],
    },
    'bouquet': {
        'no': ['a person with empty hands', 'a person holding a bag or a phone'],
        'yes': ['a person holding a bouquet of flowers in their hands', 'a bride holding a bunch of flowers'],
    },
    'veil': {
        'no': ['a person with uncovered hair', 'a person wearing a regular hat'],
        'yes': ['a bride wearing a wedding veil over her head', 'a bridal veil falling from a person\'s head'],
    },
    'bowtie': {
        'no': ['a person wearing an ordinary long necktie', 'a person with an open shirt collar'],
        'yes': ['a person wearing a formal bow tie at the shirt collar', 'a tuxedo with a bow tie'],
    },
}
THRESHOLDS = {'outfit': (.23, .015), 'hat': (.23, .03), 'bouquet': (.24, .025), 'veil': (.28, .05), 'bowtie': (.24, .035)}


def default_appearance():
    return dict(outfit='none', accessories=[], hat_style='top-hat', bouquet_hand='both')


def available():
    if not (MODEL_DIR / 'manifest.json').is_file():
        return False, 'Automatic outfits are not installed. Run scripts/download_appearance_model.py; manual outfit controls remain available.'
    if not all(importlib.util.find_spec(name) is not None for name in ('onnxruntime', 'tokenizers')):
        return False, 'Automatic outfits need backend/requirements-appearance.txt. Manual outfit controls remain available.'
    try:
        manifest = json.loads((MODEL_DIR / 'manifest.json').read_text())
        names = ('vision_model_quantized.onnx', 'text_model_quantized.onnx', 'tokenizer.json')
        if not all((MODEL_DIR / name).is_file() and name in manifest['files'] for name in names):
            return False, 'Automatic outfit model files are incomplete. Rerun scripts/download_appearance_model.py.'
    except (OSError, ValueError, KeyError, TypeError):
        return False, 'Automatic outfit model manifest is invalid. Rerun scripts/download_appearance_model.py.'
    return True, 'Local CLIP outfit suggestions; photos stay on this computer. Small or hidden details may need correction.'


def choose_label(scores, field):
    """Pure decision rule, intentionally conservative about unsupported garments."""
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    winner, score = ranked[0]
    margin = float(score - ranked[1][1]) if len(ranked) > 1 else 0.0
    minimum, required_margin = THRESHOLDS[field]
    if field == 'outfit' and winner == 'none':
        # A skin-colored sleeve or an overlapping bare arm is not sufficient
        # evidence to remove somebody's clothing.
        minimum, required_margin = .25, .04
    return winner, bool(np.isfinite(score) and score >= minimum and margin >= required_margin and winner != 'other'), float(score), margin


def status():
    ready, detail = available()
    return dict(available=ready, model=MODEL_NAME, detail=detail)


def _pixel_box(image, box):
    left, top, right, bottom = box
    left = max(0, min(image.width - 1, int(left * image.width)))
    top = max(0, min(image.height - 1, int(top * image.height)))
    right = max(left + 1, min(image.width, int(right * image.width)))
    bottom = max(top + 1, min(image.height, int(bottom * image.height)))
    return left, top, right, bottom


def subject_crops(image, subject):
    """Use the person's landmarks, never the entire group, for clothing crops."""
    points = subject['landmarks']
    left, top, right, bottom = subject['bbox']
    width, height = max(right - left, .02), max(bottom - top, .04)
    full_box = (left - width * .13, top - height * .12, right + width * .13, bottom + height * .06)
    full = image.crop(_pixel_box(image, full_box))
    torso_points = [points[k] for k in (11, 12, 23, 24) if points[k]['visibility'] >= .5]
    outfit = full
    if len(torso_points) == 4:
        torso_height = max(p['y'] for p in torso_points) - min(p['y'] for p in torso_points)
        torso_width = max(p['x'] for p in torso_points) - min(p['x'] for p in torso_points)
        # Avoid dressing a nearby person in their neighbor's gown. A torso crop
        # omits most overlapping arms and head/background context.
        torso_box = (min(p['x'] for p in torso_points) - torso_width * .2,
                     min(p['y'] for p in torso_points) - torso_height * .1,
                     max(p['x'] for p in torso_points) + torso_width * .2,
                     max(p['y'] for p in torso_points) + torso_height * .4)
        outfit = image.crop(_pixel_box(image, torso_box))
    shoulder_width = abs(points[11]['x'] - points[12]['x'])
    radius_x = max(shoulder_width * .9, width * .23)
    # Image coordinates have different X/Y scales. Preserve a physical crop radius.
    radius_y = radius_x * image.width / image.height
    nose = points[0]
    face = [points[k] for k in range(11) if points[k]['visibility'] >= .6]
    ear_points = [points[k] for k in (7, 8) if points[k]['visibility'] >= .6]
    head_x = np.mean([p['x'] for p in ear_points]) if ear_points else nose['x']
    head_y = np.mean([p['y'] for p in ear_points]) if ear_points else nose['y']
    shoulder = np.mean([[points[k]['x'] * image.width, points[k]['y'] * image.height] for k in (11, 12)], axis=0)
    neck_length = np.linalg.norm(np.array([nose['x'] * image.width, nose['y'] * image.height]) - shoulder)
    face_span = max((max(p['x'] for p in face) - min(p['x'] for p in face)) * image.width,
                    (max(p['y'] for p in face) - min(p['y'] for p in face)) * image.height) if face else 0
    head_radius = max(face_span * .9, neck_length * .9, 18)
    head_box = (head_x - head_radius / image.width, head_y - head_radius * 1.5 / image.height,
                head_x + head_radius / image.width, head_y + head_radius * 1.1 / image.height)
    head = image.crop(_pixel_box(image, head_box))
    upper_box = (left - width * .1, top - height * .12, right + width * .1, top + height * .62)
    upper = image.crop(_pixel_box(image, upper_box))
    wrists = [p for p in (points[15], points[16]) if p['visibility'] >= .5]
    if wrists:
        hand_box = (min(p['x'] for p in wrists) - radius_x * .9,
                    min(p['y'] for p in wrists) - radius_y * 1.2,
                    max(p['x'] for p in wrists) + radius_x * .9,
                    max(p['y'] for p in wrists) + radius_y * 1.2)
        hands = image.crop(_pixel_box(image, hand_box))
    else:
        hands = full
    crops = {'outfit': outfit, 'hat': head, 'bouquet': hands, 'veil': head, 'bowtie': upper}
    for side, index in (('left', 15), ('right', 16)):
        wrist = points[index]
        if wrist['visibility'] >= .5:
            box = (wrist['x'] - radius_x * .8, wrist['y'] - radius_y,
                   wrist['x'] + radius_x * .8, wrist['y'] + radius_y)
            crops['bouquet-' + side] = image.crop(_pixel_box(image, box))
    return crops


class ClipRuntime:
    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        manifest = json.loads((MODEL_DIR / 'manifest.json').read_text())
        for name, expected in manifest['files'].items():
            path = MODEL_DIR / name
            if Path(name).name != name or not path.is_file():
                raise ValueError('Invalid local appearance model path')
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != expected:
                raise ValueError('Appearance model checksum mismatch; rerun the downloader')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.vision = ort.InferenceSession(str(MODEL_DIR / 'vision_model_quantized.onnx'), options, providers=['CPUExecutionProvider'])
        self.text = ort.InferenceSession(str(MODEL_DIR / 'text_model_quantized.onnx'), options, providers=['CPUExecutionProvider'])
        tokenizer = Tokenizer.from_file(str(MODEL_DIR / 'tokenizer.json'))
        tokenizer.enable_truncation(max_length=77)
        tokenizer.enable_padding(length=77, pad_id=49407, pad_token='<|endoftext|>')
        prompts, references = [], []
        for field, labels in LABELS.items():
            for label, descriptions in labels.items():
                for description in descriptions:
                    prompts.append('a photo of ' + description)
                    references.append((field, label))
        encoded = tokenizer.encode_batch(prompts)
        feed = {'input_ids': np.array([e.ids for e in encoded], dtype=np.int64),
                'attention_mask': np.array([e.attention_mask for e in encoded], dtype=np.int64)}
        inputs = {item.name for item in self.text.get_inputs()}
        outputs = [item.name for item in self.text.get_outputs()]
        embeddings = self.text.run(None, {name: value for name, value in feed.items() if name in inputs})[outputs.index('text_embeds')]
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        self.embeddings = {}
        for field, labels in LABELS.items():
            self.embeddings[field] = {}
            for label in labels:
                vector = embeddings[[i for i, ref in enumerate(references) if ref == (field, label)]].mean(axis=0)
                self.embeddings[field][label] = vector / max(np.linalg.norm(vector), 1e-12)

    def score(self, crops):
        arrays = []
        for crop in crops.values():
            resized = ImageOps.fit(crop.convert('RGB'), (224, 224), method=Image.Resampling.BICUBIC)
            pixels = np.asarray(resized, dtype=np.float32) / 255
            pixels = (pixels - np.array([.48145466, .4578275, .40821073], dtype=np.float32)) / np.array([.26862954, .26130258, .27577711], dtype=np.float32)
            arrays.append(pixels.transpose(2, 0, 1))
        outputs = [item.name for item in self.vision.get_outputs()]
        embeddings = self.vision.run(None, {'pixel_values': np.stack(arrays)})[outputs.index('image_embeds')]
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        return {field: {label: float(embeddings[i] @ vector) for label, vector in self.embeddings[field.split('-')[0]].items()}
                for i, field in enumerate(crops)}


def analyze_person(image, subject):
    global _runtime
    appearance = default_appearance()
    ready, detail = available()
    if not ready:
        return appearance, dict(status='unavailable', model=MODEL_NAME, confidence=None, warnings=[detail])
    try:
        with _lock:
            if _runtime is None:
                _runtime = ClipRuntime()
            scores = _runtime.score(subject_crops(image, subject))
    except Exception:
        logger.exception('Local appearance inference failed')
        return appearance, dict(status='unavailable', model=MODEL_NAME, confidence=None,
                                warnings=['Automatic outfit analysis failed. No outfit or accessory was inferred; use the correction controls.'])
    warnings, decisions = [], {}
    for field in LABELS:
        values = scores[field]
        label, accepted, _score, _margin = choose_label(values, field)
        decisions[field] = (label, accepted)
    outfit, accepted = decisions['outfit']
    if accepted:
        appearance['outfit'] = outfit
    else:
        warnings.append('Outfit uncertain; choose the clothing in the correction controls.')
    hat, accepted = decisions['hat']
    if accepted and hat not in ('none', 'other'):
        appearance['accessories'].append('hat')
        appearance['hat_style'] = hat
    elif not accepted:
        warnings.append('Headwear uncertain; check for a missed hat.')
    for field in ('bouquet', 'veil', 'bowtie'):
        label, accepted = decisions[field]
        if accepted and label == 'yes':
            # A bridal veil and bow tie require coherent outfit evidence too.
            coherent = field == 'bouquet' or (field == 'veil' and appearance['outfit'] == 'wedding-dress') or (field == 'bowtie' and appearance['outfit'] == 'suit')
            if coherent:
                appearance['accessories'].append(field)
            else:
                warnings.append(f'Possible {field}; outfit evidence is uncertain, so it was left unset.')
        elif not accepted:
            warnings.append(f'{field.capitalize()} uncertain; small or hidden details may be missed.')
    if 'bouquet' in appearance['accessories']:
        matched_hands = []
        for side in ('left', 'right'):
            values = scores.get('bouquet-' + side)
            if values:
                label, accepted, _, _ = choose_label(values, 'bouquet')
                if accepted and label == 'yes':
                    matched_hands.append(side)
        if len(matched_hands) == 1:
            appearance['bouquet_hand'] = matched_hands[0]
        elif not matched_hands:
            warnings.append('The bouquet hand is uncertain; it was placed between both hands for review.')
    if min(image.width * (subject['bbox'][2] - subject['bbox'][0]), image.height * (subject['bbox'][3] - subject['bbox'][1])) < 96:
        warnings.append('This person is small in the photo; use a closer, sharper image for outfit details.')
    return appearance, dict(status='uncertain' if warnings else 'ready', model=MODEL_NAME, confidence=None,
                            warnings=warnings + ['Automatic suggestions use image similarity, not measured certainty. Review them before generating.'])
