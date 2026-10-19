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

from . import garment_parser
from .garment_estimation import compact_evidence, estimate_appearance

MODEL_DIR = Path(os.environ.get('APPEARANCE_MODEL_DIR', str(Path(__file__).resolve().parents[2] / '.local/appearance')))
MODEL_NAME = 'Local CLIP + SCHP photo clothing analysis'
_lock = Lock()
_runtime = None
logger = logging.getLogger(__name__)

# Descriptions include negative alternatives. A winner must pass both an absolute
# cosine threshold and a margin over the runner-up; no softmax is called a probability.
LABELS = {
    'upper': {
        'bare': ['a shirtless person with a bare torso', 'a person with no shirt on'],
        'shirt': ['a person wearing a shirt or t-shirt', 'a person wearing a sweater or blouse'],
        'jacket': ['a person wearing a suit jacket', 'a person wearing a coat or jacket'],
        'dress': ['a person wearing a dress', 'a person wearing a bridal gown'],
        'other': ['a background scene with no person', 'an obscured person whose clothes are hidden'],
    },
    'lower': {
        'bare': ['a person with bare legs wearing a swimsuit', 'a person with unclothed legs'],
        'pants': ['a person wearing full length trousers or pants', 'a person wearing long jeans'],
        'shorts': ['a person wearing shorts', 'a person wearing short trousers'],
        'skirt': ['a person wearing a skirt', 'a skirt worn around a person\'s waist'],
        'dress': ['a person wearing a dress', 'a person wearing a long bridal gown'],
        'other': ['a background with no legs visible', 'a person whose legs and clothes are obscured'],
    },
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
        'beanie': ['a person wearing a knitted beanie hat', 'a person wearing a close fitting winter hat'],
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
    'bag': {
        'no': ['a person with empty hands and no bag', 'a person holding flowers or a phone'],
        'yes': ['a person carrying a handbag or shopping bag', 'a person holding a shoulder bag'],
    },
    'scarf': {
        'no': ['a person with no scarf around their neck', 'a person wearing an ordinary shirt collar'],
        'yes': ['a person wearing a scarf around their neck', 'a person wearing a long winter scarf'],
    },
    'glasses': {
        'no': ['a face without glasses', 'a person with uncovered eyes'],
        'yes': ['a person wearing glasses over their eyes', 'a person wearing sunglasses'],
    },
    'mobility': {
        'none': ['a person standing or sitting without a mobility aid', 'a person on an ordinary chair'],
        'wheelchair': ['a person sitting in a wheelchair with large wheels', 'a wheelchair user with their wheelchair'],
        'cane': ['a person holding a walking cane', 'a person using a walking stick for support'],
        'other': ['a person holding a dog leash', 'a person holding an umbrella or a branch'],
    },
    'object': {
        'none': ['a person with empty hands', 'a person holding flowers or a bag'],
        'phone': ['a person holding a mobile phone in their hand', 'a person looking at a smartphone'],
        'book': ['a person holding a book', 'a person carrying a book in their hand'],
        'camera': ['a person carrying a camera around their neck', 'a person holding a photographic camera'],
    },
}
THRESHOLDS = {'outfit': (.23, .015), 'upper': (.23, .02), 'lower': (.23, .02),
              'hat': (.23, .03), 'bouquet': (.24, .025), 'veil': (.28, .05),
              'bowtie': (.24, .035), 'bag': (.27, .035), 'scarf': (.27, .04), 'glasses': (.25, .035),
              'mobility': (.25, .035), 'object': (.26, .04)}


def default_appearance():
    # A photo with unknown clothes is not evidence that somebody is unclothed.
    return dict(outfit='none', accessories=[], hat_style='top-hat', bouquet_hand='both',
                garment=garment_parser.default_garment(inferred=True))


def available():
    if not (MODEL_DIR / 'manifest.json').is_file():
        return False, 'Photo clothing recognition is not installed. Run scripts/download_appearance_model.py.'
    if not all(importlib.util.find_spec(name) is not None for name in ('onnxruntime', 'tokenizers')):
        return False, 'Photo clothing recognition needs backend/requirements-appearance.txt.'
    try:
        manifest = json.loads((MODEL_DIR / 'manifest.json').read_text())
        names = ('vision_model_quantized.onnx', 'text_model_quantized.onnx', 'tokenizer.json')
        if not all((MODEL_DIR / name).is_file() and name in manifest['files'] for name in names):
            return False, 'Automatic outfit model files are incomplete. Rerun scripts/download_appearance_model.py.'
    except (OSError, ValueError, KeyError, TypeError):
        return False, 'Automatic outfit model manifest is invalid. Rerun scripts/download_appearance_model.py.'
    return True, 'Local clothing recognition; photos stay on this computer. Small, overlapping or hidden details may be missed.'


def choose_label(scores, field):
    """Pure decision rule, intentionally conservative about unsupported garments."""
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    winner, score = ranked[0]
    margin = float(score - ranked[1][1]) if len(ranked) > 1 else 0.0
    minimum, required_margin = THRESHOLDS[field]
    if field == 'mobility' and winner == 'cane':
        minimum, required_margin = .28, .06
    if (field == 'outfit' and winner == 'none') or (field in ('upper', 'lower') and winner == 'bare'):
        # A skin-colored sleeve or an overlapping bare arm is not sufficient
        # evidence to remove somebody's clothing.
        minimum, required_margin = .25, .04
    return winner, bool(np.isfinite(score) and score >= minimum and margin >= required_margin and winner != 'other'), float(score), margin


def status():
    ready, detail = available()
    parser = garment_parser.status()
    return dict(available=ready and parser['available'], recognition_available=ready,
                segmentation=parser, model=MODEL_NAME, detail=detail if ready else detail + ' ' + parser['detail'])


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
    hip_y = np.mean([points[k]['y'] for k in (23, 24)])
    ankle_y = max(points[k]['y'] for k in (27, 28))
    hip_x = np.mean([points[k]['x'] for k in (23, 24)])
    leg_width = max(abs(points[23]['x'] - points[24]['x']) * 1.4, width * .4)
    lower = image.crop(_pixel_box(image, (hip_x - leg_width, hip_y - height * .06,
                                          hip_x + leg_width, max(ankle_y, hip_y + height * .15) + height * .04)))
    crops = {'outfit': outfit, 'upper': outfit, 'lower': lower, 'hat': head,
             'bouquet': hands, 'veil': head, 'bowtie': upper, 'bag': full,
             'scarf': upper, 'glasses': head, 'mobility': full, 'object': hands}
    for side, index in (('left', 15), ('right', 16)):
        wrist = points[index]
        if wrist['visibility'] >= .5:
            # Ownership needs a hand-local crop. A broad sleeve/torso-sized
            # crop includes flowers held by neighbors in tightly posed groups.
            hand_radius_x = max(shoulder_width * .4, width * .1)
            hand_radius_y = hand_radius_x * image.width / image.height
            box = (wrist['x'] - hand_radius_x, wrist['y'] - hand_radius_y,
                   wrist['x'] + hand_radius_x, wrist['y'] + hand_radius_y)
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


def _garment_from_evidence(appearance, decisions, measurements, warnings):
    garment = appearance['garment']
    for field in ('upper', 'lower'):
        label, accepted = decisions.get(field, ('unknown', False))
        if accepted and label != 'other':
            garment[field] = label
    # Formal fine cues refine the broad per-part result, but no wedding gown is
    # assigned solely because another person in the crop is wearing white.
    if appearance['outfit'] == 'suit':
        garment['upper'] = 'jacket'
        garment['sleeve_length'] = 1.
        if garment['lower'] == 'unknown':
            garment['lower'] = 'pants'
    elif appearance['outfit'] == 'wedding-dress':
        garment['upper'] = garment['lower'] = 'dress'
        garment['hem_length'] = 1.
    if garment['upper'] == 'dress' or garment['lower'] == 'dress':
        garment['upper'] = garment['lower'] = 'dress'
        if appearance['outfit'] != 'wedding-dress':
            garment['hem_length'] = .65
    if measurements:
        semantics = measurements.get('semantics', {})
        upper, lower = semantics.get('upper', 'unknown'), semantics.get('lower', 'unknown')
        language_lower, language_accepted = decisions.get('lower', ('unknown', False))
        lower_conflict = (lower != 'unknown'
                          and language_lower not in ('unknown', 'other', 'bare')
                          and ({language_lower, lower} not in ({'shorts', 'pants'},))
                          and language_lower != lower)
        if upper == 'dress' or lower == 'dress':
            garment['upper'] = garment['lower'] = 'dress'
        else:
            if upper != 'unknown':
                garment['upper'] = 'jacket' if garment['upper'] == 'jacket' and upper == 'shirt' else upper
            if lower != 'unknown':
                garment['lower'] = lower
            if upper == 'shirt' and garment['lower'] == 'dress' and lower == 'unknown':
                garment['lower'] = 'unknown'
            if appearance['outfit'] == 'wedding-dress' and upper == 'shirt' and lower == 'pants':
                appearance['outfit'] = 'casual'
                warnings.append('Clothing regions contradicted the bridal label; the visible separate garments were used.')
        if lower_conflict:
            garment['lower'] = 'unknown'
            if garment['upper'] == 'dress' and upper != 'dress':
                garment['upper'] = upper
            warnings.append('The clothing recognizers disagree about the lower garment; its shape will be estimated from their combined evidence.')
        # The pixel mask independently establishes clothing when the vocabulary
        # classifier cannot name a cut. It cannot distinguish shirt from jacket.
        if measurements['torso_clothes'] > .35 and garment['upper'] in ('unknown', 'bare'):
            garment['upper'] = 'shirt'
            warnings.append('A visible upper garment was simplified; its exact cut could not be identified.')
        if measurements['torso_skin'] > .7 and garment['upper'] == 'bare':
            garment['upper'] = 'bare'
        elif garment['upper'] == 'bare':
            garment['upper'] = 'unknown'
            warnings.append('A bare torso could not be confirmed; a simple upper garment will be estimated.')
        for field in ('sleeve_length', 'trouser_length', 'hem_length', 'skirt_flare'):
            value = measurements.get(field)
            if value is not None:
                garment[field] = float(np.clip(value, 0, 1))
        if garment['upper'] == 'jacket' and measurements.get('sleeve_length') is None:
            garment['sleeve_length'] = 1.
        warnings.extend(measurements.get('warnings', []))
        if garment['lower'] in ('pants', 'shorts') and measurements.get('trouser_length') is not None:
            garment['lower'] = 'shorts' if measurements['trouser_length'] < .62 else 'pants'
    elif garment['upper'] == 'bare':
        garment['upper'] = 'unknown'
        warnings.append('A bare torso could not be independently confirmed.')
    if garment['upper'] == 'unknown':
        warnings.append('Upper clothing could not be identified confidently; a simple garment will be estimated from the available photo evidence.')
    if garment['lower'] == 'unknown':
        warnings.append('Lower clothing could not be identified confidently; a simple garment will be estimated from the available photo evidence.')
    hat, accepted = decisions.get('hat', ('none', False))
    if accepted:
        garment['headwear'] = {'cap': 'cap', 'top-hat': 'brim', 'wide-brim': 'brim',
                               'beanie': 'beanie', 'none': 'none'}.get(hat, 'unknown')
    else:
        garment['headwear'] = 'unknown'
    semantics = measurements.get('semantics', {}) if measurements else {}
    if semantics.get('hat') and garment['headwear'] in ('none', 'unknown'):
        garment['headwear'] = 'beanie'
        appearance['hat_style'] = 'beanie'
        appearance['accessories'].append('hat')
        warnings.append('Visible headwear was simplified; its precise brim or crown shape was unclear.')
    for accessory in ('bag', 'scarf', 'glasses'):
        language_label, language_accepted = decisions.get(accessory, ('no', False))
        if semantics.get(accessory) and (accessory == 'glasses' or (language_label == 'yes' and language_accepted)):
            appearance['accessories'].append(accessory)
        elif semantics.get(accessory):
            warnings.append(f'A possible {accessory} region was visible, but its type was unclear and no accessory shape was invented.')
    if semantics.get('bag_hand'):
        appearance['bouquet_hand'] = semantics['bag_hand']
    if appearance['outfit'] == 'none' and any(garment[f] not in ('unknown', 'bare') for f in ('upper', 'lower')):
        appearance['outfit'] = 'casual'


def appearance_summary(appearance, measurements=None):
    garment = appearance['garment']
    estimated = set(garment.get('estimated_fields', []))
    def labelled(value, *fields):
        return value + (' (estimated)' if estimated.intersection(fields) else '')
    parts = []
    if appearance['outfit'] == 'wedding-dress':
        parts.append(labelled('Bridal dress', 'upper', 'lower'))
    elif garment['upper'] == garment['lower'] == 'dress':
        parts.append(labelled('Dress', 'upper', 'lower'))
    else:
        parts.append(labelled({'unknown': 'Upper clothing unclear', 'bare': 'Bare torso',
                      'shirt': 'Simplified upper garment', 'jacket': 'Jacket', 'dress': 'Dress'}[garment['upper']], 'upper'))
        parts.append(labelled({'unknown': 'Lower clothing unclear', 'bare': 'Bare legs',
                      'pants': 'Trousers', 'shorts': 'Shorts', 'skirt': 'Skirt', 'dress': 'Dress'}[garment['lower']], 'lower'))
    if garment['upper'] not in ('unknown', 'bare'):
        length = garment['sleeve_length']
        if measurements and measurements.get('sleeve_length') is not None:
            parts.append('sleeveless' if length < .15 else 'short sleeves' if length < .6 else 'long sleeves')
        else:
            parts.append('sleeve length estimated')
    parts.extend(accessory.replace('bowtie', 'bow tie') for accessory in appearance['accessories'])
    return '; '.join(parts) + '.'


def analyze_person(image, subject):
    global _runtime
    appearance = default_appearance()
    measurements, segmentation = garment_parser.analyze_person(image, subject)
    warnings = ['Clothing shapes and accessories are estimates; compare the whole group with your photo.']
    warnings.extend(segmentation.get('warnings', []))
    ready, detail = available()
    scores = {}
    if ready:
        try:
            with _lock:
                if _runtime is None:
                    _runtime = ClipRuntime()
                scores = _runtime.score(subject_crops(image, subject))
        except Exception:
            logger.exception('Local appearance inference failed')
            ready = False
            warnings.append('Local clothing recognition failed; specific garments and accessories could not be identified.')
    else:
        warnings.append(detail)
    decisions = {}
    for field in LABELS:
        values = scores.get(field)
        if values:
            label, accepted, _score, _margin = choose_label(values, field)
            decisions[field] = (label, accepted)
    outfit, accepted = decisions.get('outfit', ('none', False))
    if accepted:
        appearance['outfit'] = outfit
    _garment_from_evidence(appearance, decisions, measurements, warnings)
    hat, accepted = decisions.get('hat', ('none', False))
    if accepted and hat not in ('none', 'other'):
        appearance['accessories'].append('hat')
        appearance['hat_style'] = hat
    for field in ('bouquet', 'veil', 'bowtie', 'bag', 'scarf', 'glasses'):
        label, accepted = decisions.get(field, ('no', False))
        if accepted and label == 'yes':
            coherent = field not in ('veil', 'bowtie') or (field == 'veil' and appearance['outfit'] == 'wedding-dress') or (field == 'bowtie' and appearance['outfit'] == 'suit')
            if coherent:
                appearance['accessories'].append(field)
            else:
                warnings.append(f'Possible {field}; the outfit evidence was insufficient to include it.')
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
            # Flowers in a neighboring person's crop are not evidence that this
            # person holds them. A positive broad crop needs a visible, local
            # wrist match before we attach a printable bouquet to anyone.
            appearance['accessories'].remove('bouquet')
            warnings.append('Flowers were visible nearby, but no visible hand could be associated with them; no bouquet was assigned to this person.')
    if min(image.width * (subject['bbox'][2] - subject['bbox'][0]), image.height * (subject['bbox'][3] - subject['bbox'][1])) < 96:
        warnings.append('This person is small in the photo; clothing and accessory details may be missed.')
    unsupported = []
    for field in ('mobility', 'object'):
        label, accepted = decisions.get(field, ('none', False))
        if accepted and label != 'none':
            unsupported.append(label)
            warnings.append(f'A possible {label} was detected but cannot be reconstructed by this app. The result would omit it.')
    appearance['accessories'] = list(dict.fromkeys(appearance['accessories']))
    appearance['garment_evidence'] = compact_evidence(scores, measurements)
    appearance = estimate_appearance(appearance)
    estimated_fields = appearance['garment'].get('estimated_fields', [])
    if estimated_fields:
        warnings.append('Inferred clothing: uncertain garments were estimated from the strongest available evidence so the whole group can be generated. Compare these shapes with your photo.')
    if not ready and measurements is None:
        result_status = 'unavailable'
    else:
        result_status = 'uncertain' if warnings else 'ready'
    # Similarity margins and segmentation labels are evidence, not calibrated
    # confidence percentages. Missing objects are never claimed absent.
    return appearance, dict(status=result_status, model=MODEL_NAME, confidence=None,
                            summary=appearance_summary(appearance, measurements), segmentation=segmentation,
                            unsupported_items=unsupported, blocking=any(item in ('wheelchair', 'cane') for item in unsupported),
                            warning_classes=['inferred_clothing'] if estimated_fields else [],
                            warnings=list(dict.fromkeys(warnings)),
                            limitations=['Only visible clothing shapes are approximated. Hidden surfaces, fine details and unfamiliar objects may not match.'])
