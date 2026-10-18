"""Local six-person pose and outfit analysis. Uploaded pixels stay in memory."""
import atexit
import os
from pathlib import Path
from threading import Lock

import numpy as np

from . import appearance

MODEL_PATH = Path(os.environ.get('POSE_MODEL_PATH', str(Path(__file__).resolve().parents[2] / '.local/pose_landmarker_lite.task')))
MAX_PEOPLE = 6
_lock = Lock()
_detector = None


def available():
    if not MODEL_PATH.is_file():
        return False, 'Pose model is not installed. Choose a manual preset, or run scripts/download_model.py.'
    try:
        import mediapipe  # noqa: F401
        return True, 'Local MediaPipe pose estimation for up to six people. No photo upload to third parties.'
    except (ImportError, OSError) as exc:
        return False, f'Pose runtime unavailable ({type(exc).__name__}). Choose a manual preset.'


def photo_warnings(image):
    warnings = []
    if min(image.size) < 400:
        warnings.append('This is a small photo. Use a sharp, higher-resolution image for group and accessory recognition.')
    try:
        import cv2
        gray = np.asarray(image.convert('L'))
        # A modest advisory threshold; this never invents or rejects a pose.
        if min(image.size) >= 200 and cv2.Laplacian(gray, cv2.CV_64F).var() < 18:
            warnings.append('The photo has little sharp detail. A clearer image may improve poses and small accessories.')
    except ImportError:
        pass
    return warnings


def _length(points, image_size):
    """Projected anatomical length, including bent legs; absent limbs stay unknown."""
    width, height = image_size
    def point(i):
        p = points[i]
        return np.array([p['x'] * width, p['y'] * height])
    def visible(indices):
        return all(points[k]['visibility'] >= .55 for k in indices)
    if not visible((0, 11, 12, 23, 24)):
        return None
    legs = [np.linalg.norm(point(hip) - point(knee)) + np.linalg.norm(point(knee) - point(ankle))
            for hip, knee, ankle in ((23, 25, 27), (24, 26, 28)) if visible((hip, knee, ankle))]
    if not legs:
        return None
    shoulders = (point(11) + point(12)) / 2
    pelvis = (point(23) + point(24)) / 2
    value = np.linalg.norm(point(0) - shoulders) * 1.8 + np.linalg.norm(shoulders - pelvis) + np.mean(legs)
    return float(value) if value > 10 else None


def suggest_layout(subjects, image_size):
    """Preserve left-to-right arrangement and approximate relative size.

    One view cannot establish physical height or floor depth. All figures stay on
    the same plane; relative sizes are editable estimates.
    """
    if not subjects:
        return
    lengths = [_length(subject['landmarks'], image_size) for subject in subjects]
    known = [length for length in lengths if length is not None]
    reference = float(np.median(known)) if known else image_size[1] * .7
    centers = [(subject['landmarks'][23]['x'] + subject['landmarks'][24]['x']) / 2 for subject in subjects]
    midpoint = (min(centers) + max(centers)) / 2
    # Larger heads need more room than the human proportions in the photo.
    offsets = [(center - midpoint) * image_size[0] / reference * 90 * 1.35 for center in centers]
    largest = max((abs(value) for value in offsets), default=0)
    if largest > 200:
        offsets = [value * 200 / largest for value in offsets]
    for subject, length, offset in zip(subjects, lengths, offsets):
        ratio = float(np.clip(length / reference, .65, 1.3)) if length is not None else 1.0
        subject['suggested_transform'] = dict(x=round(offset, 2), y=0, scale=1)
        subject['suggested_height_mm'] = round(90 * ratio, 1)
        if len(subjects) > 1:
            subject['warnings'].append('Arrangement and relative heights are image-based estimates; perspective and hidden limbs can change apparent size. Floor depth is not inferred.')


def _subject_warnings(points, bbox, confidence, image_size):
    warnings = ['Single-photo depth is ambiguous; canonical anatomy and conservative depth are used.']
    if confidence < .8:
        warnings.append('Some limbs are hidden or uncertain. Review the pose before generating.')
    if min(points[k]['visibility'] for k in (27, 28)) < .5:
        warnings.append('Feet are hidden or uncertain; lower-leg placement may need correction. A full-body photo works best.')
    if any(points[k]['y'] > .985 or points[k]['y'] < .005 for k in (0, 27, 28)) or bbox[0] < .005 or bbox[2] > .995:
        warnings.append('This person touches the photo edge and may be cropped. Include the whole head, hands, and feet if possible.')
    if (bbox[3] - bbox[1]) * image_size[1] < 180:
        warnings.append('This person occupies few pixels; a closer photo may improve pose and clothing suggestions.')
    return warnings


def _same_person(first, second, image_size):
    """Merge overlapping-window detections by head AND pelvis, not box overlap alone."""
    scale = np.array(image_size, dtype=float)
    def at(points, indices):
        return np.mean([[points[i]['x'], points[i]['y']] for i in indices], axis=0) * scale
    head_distance = np.linalg.norm(at(first, (0,)) - at(second, (0,)))
    pelvis_distance = np.linalg.norm(at(first, (23, 24)) - at(second, (23, 24)))
    torso = max(np.linalg.norm(at(points, (11, 12)) - at(points, (23, 24))) for points in (first, second))
    if all(points[k]['visibility'] >= .6 for points in (first, second) for k in (7, 8)):
        # Occluded hips can jump between people in a hug. Matching ear centers
        # identify the same visible head even when its body estimate changes.
        ear_distance = np.linalg.norm(at(first, (7, 8)) - at(second, (7, 8)))
        ear_span = max(np.linalg.norm(at(points, (7,)) - at(points, (8,))) for points in (first, second))
        if ear_distance < max(8, ear_span * .65) and head_distance < max(12, ear_span * 1.5):
            return True
    return head_distance < max(12, torso * .55) and pelvis_distance < max(12, torso * .55)


def _detect_points(detector, image, mp):
    def detect(crop, offset=(0, 0), factor=(1., 1.)):
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(crop.convert('RGB'))))
        return [[dict(x=float(p.x) * factor[0] + offset[0], y=float(p.y) * factor[1] + offset[1],
                      z=float(p.z) * factor[0], visibility=float(p.visibility or 0)) for p in landmarks]
                for landmarks in result.pose_landmarks]
    candidates = detect(image)
    # Global person detectors resize their input. Overlapping windows recover some
    # small people in groups; detections are transformed back into the original photo.
    if max(image.size) >= 950 and len(candidates) < MAX_PEOPLE:
        columns = 3 if image.width / image.height > 1.8 else 2
        rows = 2
        tile_width = min(image.width, int(image.width / columns * 1.25))
        tile_height = min(image.height, int(image.height / rows * 1.25))
        for top in np.linspace(0, image.height - tile_height, rows).astype(int):
            for left in np.linspace(0, image.width - tile_width, columns).astype(int):
                tile = image.crop((left, top, left + tile_width, top + tile_height))
                new = detect(tile, (left / image.width, top / image.height),
                             (tile_width / image.width, tile_height / image.height))
                candidates.extend(new)
    merged = []
    # Keep the global estimate before cropped alternatives. A confident partial
    # crop can attach the correct head to somebody else's obscured torso.
    for points in candidates:
        if not any(_same_person(points, other, image.size) for other in merged):
            merged.append(points)
        if len(merged) == MAX_PEOPLE:
            break
    return merged


def analyze(image):
    global _detector
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    with _lock:
        if _detector is None:
            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=vision.RunningMode.IMAGE, num_poses=MAX_PEOPLE,
                min_pose_detection_confidence=.45, min_pose_presence_confidence=.45,
                min_tracking_confidence=.45)
            _detector = vision.PoseLandmarker.create_from_options(options)
        detections = _detect_points(_detector, image, mp)
    subjects = []
    general_warnings = photo_warnings(image)
    for points in detections:
        confidence = float(np.mean([points[k]['visibility'] for k in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)]))
        visible = [p for p in points if p['visibility'] > .5]
        bbox = [float(np.clip(min(p['x'] for p in visible), 0, 1)), float(np.clip(min(p['y'] for p in visible), 0, 1)),
                float(np.clip(max(p['x'] for p in visible), 0, 1)), float(np.clip(max(p['y'] for p in visible), 0, 1))] if visible else [0., 0., 1., 1.]
        subjects.append(dict(landmarks=points, confidence=confidence, bbox=bbox,
                             warnings=_subject_warnings(points, bbox, confidence, image.size) + general_warnings))
    subjects.sort(key=lambda subject: (subject['bbox'][0] + subject['bbox'][2]) / 2)
    for i, subject in enumerate(subjects):
        subject.update(id=f'person-{i + 1}', name=f'Person {i + 1}', photo_aspect=image.width / image.height)
        for other in subjects:
            if subject is other:
                continue
            a, b = subject['bbox'], other['bbox']
            overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            if smaller > 0 and overlap / smaller > .3:
                subject['warnings'].append('People overlap in this photo. Check whose arms, clothing, and held accessories were assigned to each figure.')
                break
        subject['appearance'], subject['appearance_detection'] = appearance.analyze_person(image, subject)
    suggest_layout(subjects, image.size)
    if len(subjects) == MAX_PEOPLE:
        for subject in subjects:
            subject['warnings'].append('The six-person detection limit was reached. Check whether everyone in the photo was included.')
    return subjects


def close():
    global _detector
    with _lock:
        if _detector is not None:
            detector, _detector = _detector, None
            detector.close()


atexit.register(close)
