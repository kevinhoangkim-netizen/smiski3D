"""Local six-person pose and outfit analysis. Uploaded pixels stay in memory."""
import atexit
import os
from pathlib import Path
from threading import Lock

import numpy as np

from . import appearance, people

MODEL_PATH = Path(os.environ.get('POSE_MODEL_PATH', str(Path(__file__).resolve().parents[2] / '.local/pose_landmarker_lite.task')))
MAX_PEOPLE = 6
_lock = Lock()
_detector = None


def available():
    if not MODEL_PATH.is_file():
        return False, 'Pose model is not installed. Restart the launcher to install it.'
    try:
        import mediapipe  # noqa: F401
        return True, 'Local MediaPipe pose estimation for up to six people. No photo upload to third parties.'
    except (ImportError, OSError) as exc:
        return False, f'Pose runtime unavailable ({type(exc).__name__}). Restart the launcher to repair it.'


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
    _separate_heads(subjects, image_size)


def _separate_heads(subjects, image_size):
    """Least-movement horizontal spacing for enlarged figurine heads.

    Project the original X offsets onto the ordered separation constraints.
    Isotonic regression shares the necessary movement between neighbors rather
    than pushing the entire group in one direction. No floor depth or new pose
    is introduced, and existing wide gaps remain untouched.
    """
    if len(subjects)<2:
        return
    from .geometry import head_x_envelope
    from .schemas import Figure
    ordered=sorted(subjects,key=lambda s:(s['landmarks'][23]['x']+s['landmarks'][24]['x'])/2)
    envelopes=[]
    for index,subject in enumerate(ordered):
        figure=Figure(id=f'layout-{index}',pose='photo',landmarks=subject['landmarks'],
            photo_aspect=image_size[0]/image_size[1],height_mm=subject['suggested_height_mm'],
            base={'shape':'none'})
        envelopes.append(head_x_envelope(figure))
    original=np.asarray([s['suggested_transform']['x'] for s in ordered],float)
    gaps=np.asarray([max(0.,a[1]-b[0]+1.5) for a,b in zip(envelopes[:-1],envelopes[1:])])
    if np.all(np.diff(original)>=gaps):
        return
    cumulative=np.r_[0.,np.cumsum(gaps)]
    blocks=[]
    for index,value in enumerate(original-cumulative):
        blocks.append([index,index,float(value),1])
        while len(blocks)>1 and blocks[-2][2]>blocks[-1][2]:
            right,left=blocks.pop(),blocks.pop()
            weight=left[3]+right[3]
            blocks.append([left[0],right[1],(left[2]*left[3]+right[2]*right[3])/weight,weight])
    projected=np.empty(len(ordered))
    for start,end,value,_ in blocks:
        projected[start:end+1]=value
    adjusted=projected+cumulative
    adjusted-=(adjusted[0]+adjusted[-1])/2
    for subject,offset in zip(ordered,adjusted):
        subject['suggested_transform']['x']=round(float(offset),2)
        subject['warnings'].append('People were moved slightly apart so their oversized figurine heads remain distinct. Poses and left-to-right order are unchanged; photographed contact may be approximated.')


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



def group_status():
    return people.status()


def _usable_pose(points):
    """A head and a body must be visible; never turn a bare box into a preset."""
    return (len(points) == 33 and points[0]['visibility'] >= .45
            and points[0]['y'] <= (points[23]['y'] + points[24]['y']) / 2 + .08
            and np.mean([points[k]['visibility'] for k in (11, 12, 23, 24)]) >= .5
            and sum(points[k]['visibility'] >= .4 for k in (11, 12, 23, 24)) >= 3)


def _pose_box(points):
    visible = [p for p in points if p['visibility'] > .5]
    if not visible:
        return [0., 0., 1., 1.]
    return [float(np.clip(min(p['x'] for p in visible), 0, 1)),
            float(np.clip(min(p['y'] for p in visible), 0, 1)),
            float(np.clip(max(p['x'] for p in visible), 0, 1)),
            float(np.clip(max(p['y'] for p in visible), 0, 1))]


def _plausible_body(points, bbox, image_size):
    # Visibility is a network score, not anatomical validation: a failed pose
    # can place every confident shoulder/hip inside a person's face. Compare
    # physical body spans with the independently detected full-person extent.
    scale = np.asarray(image_size, float)
    def center(indices):
        return np.mean([[points[k]['x'], points[k]['y']] for k in indices], axis=0) * scale
    body_diagonal = np.linalg.norm((np.asarray(bbox[2:]) - bbox[:2]) * scale)
    torso = np.linalg.norm(center((11, 12)) - center((23, 24)))
    head_to_hip = np.linalg.norm(center((0,)) - center((23, 24)))
    return bool(body_diagonal > 0 and torso / body_diagonal >= .10
                and head_to_hip / body_diagonal >= .14)


def _head_in_box(points, bbox):
    left, top, right, bottom = bbox
    width, height = right - left, bottom - top
    return (left - width * .08 <= points[0]['x'] <= right + width * .08
            and top - height * .10 <= points[0]['y'] <= bottom)


def _association(points, bbox, image_size=(1, 1)):
    """Match an independently found body to its head, not nearest box overlap."""
    left, top, right, bottom = bbox
    width, height = max(right - left, .01), max(bottom - top, .01)
    nose = points[0]
    if not _head_in_box(points, bbox) or not _plausible_body(points, bbox, image_size):
        return -1.
    core = [points[k] for k in (11, 12, 23, 24) if points[k]['visibility'] >= .4]
    within = sum(left - width * .1 <= p['x'] <= right + width * .1
                 and top - height * .1 <= p['y'] <= bottom + height * .1 for p in core)
    if within < 2:
        return -1.
    center = (left + right) / 2
    head_center_score = max(0., 1 - abs(nose['x'] - center) / width)
    return within / 4 + head_center_score * .45 + people._iou(_pose_box(points), bbox) * .5


def _crop_pose_points(detector, image, mp, bbox):
    # Keep the full person, including feet. A narrow default crop occasionally
    # produces a confident face-sized body, so retry a tight and a wider context
    # only when no anatomically plausible pose belongs to the target person.
    for horizontal_margin in (.18, 0., .5):
        left, top, right, bottom = bbox
        width, height = right - left, bottom - top
        left = max(0., left - width * horizontal_margin)
        right = min(1., right + width * horizontal_margin)
        top, bottom = max(0., top - height * .12), min(1., bottom + height * .08)
        crop_pixels = (int(left * image.width), int(top * image.height),
                       int(right * image.width), int(bottom * image.height))
        if crop_pixels[2] - crop_pixels[0] < 12 or crop_pixels[3] - crop_pixels[1] < 24:
            return []
        crop = image.crop(crop_pixels)
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                         data=np.asarray(crop.convert('RGB'))))
        x_offset, y_offset = crop_pixels[0] / image.width, crop_pixels[1] / image.height
        x_factor, y_factor = crop.width / image.width, crop.height / image.height
        mapped = [[dict(x=float(p.x) * x_factor + x_offset, y=float(p.y) * y_factor + y_offset,
                        z=float(p.z) * x_factor, visibility=float(p.visibility or 0)) for p in landmarks]
                  for landmarks in result.pose_landmarks]
        usable = [points for points in mapped if _usable_pose(points)
                  and _association(points, bbox, image.size) > 0]
        if usable:
            return usable
    return []


def _match_group(detections, candidates, image_size, regions=None, baseline_points=None):
    """Associate every detection or preserve it as explicitly unresolved.

    Distinct, usable MediaPipe heads may recover a second person inside a single
    YOLOX box (for example a hug). Matching never discards an unmatched body box.
    """
    from scipy.optimize import linear_sum_assignment
    unique = []
    for points in candidates:
        if not _usable_pose(points):
            continue
        covering = [entry['bbox'] for entry in detections if _head_in_box(points, entry['bbox'])]
        if covering and not any(_plausible_body(points, box, image_size) for box in covering):
            continue
        if not any(_same_person(points, other, image_size) for other in unique):
            unique.append(points)
    assigned = {}
    if detections and unique:
        scores = np.asarray([[_association(points, box['bbox'], image_size) for points in unique] for box in detections])
        rows, columns = linear_sum_assignment(-scores)
        for row, column in zip(rows, columns):
            if scores[row, column] > 0:
                assigned[int(row)] = int(column)
    resolved = [(detections[row]['bbox'], unique[column]) for row, column in assigned.items()]
    unresolved = [box['bbox'] for row, box in enumerate(detections) if row not in assigned]
    # A real head/body pose is independent evidence even when the object detector
    # groups two overlapping people into one box. The deduplication above checks
    # head/ear centers as well as pelvis so a changed hidden hip is not a person.
    for index, points in enumerate(unique):
        if index not in assigned.values():
            resolved.append((_pose_box(points), points))
    for region in regions or []:
        # A retry request is not proof of another person. Preserve failure for
        # review, without inventing a standing figure or silently dropping it.
        existing = unique if baseline_points is None else baseline_points
        recovered = any(_association(points, region, image_size) > 0
                        and not any(_same_person(points, old, image_size) for old in existing)
                        for _, points in resolved)
        if not recovered:
            if not any(people._iou(region, box) > .3 for box in unresolved):
                unresolved.append(list(region))
    resolved.sort(key=lambda pair: (pair[0][0] + pair[0][2]) / 2)
    return resolved, unresolved


def analyze_group(image, regions=None):
    """Photo-first group analysis: independent people count plus per-person pose.

    subjects contains only usable image-derived poses. detected_people also counts
    unresolved/overflow people, so the UI can block incomplete group generation.
    The legacy analyze() entry point is retained for older integrations.
    """
    global _detector
    ready, detail = available()
    if not ready:
        raise RuntimeError(detail)
    baseline_detections = people.detect(image)
    detections = people.detect(image, regions=regions) if regions else baseline_detections
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
        global_candidates = _detect_points(_detector, image, mp)
        candidates = []
        # Each independent body gets its own full-height pass. Even when a pose
        # cannot be recovered, its detection remains in unresolved_regions.
        for detection in baseline_detections[:100]:
            candidates.extend(_crop_pose_points(_detector, image, mp, detection['bbox']))
        # Full-person crops distinguish close heads. A global estimate can fuse
        # the heads/hips of an embracing couple, so it is a fallback candidate.
        candidates.extend(global_candidates)
        baseline_resolved, _ = _match_group(baseline_detections, candidates, image.size)
        for detection in detections[:100]:
            if not any(people._duplicate(detection['bbox'], old['bbox']) for old in baseline_detections):
                candidates.extend(_crop_pose_points(_detector, image, mp, detection['bbox']))
        for region in regions or []:
            candidates.extend(_crop_pose_points(_detector, image, mp, region))
    resolved, unresolved = _match_group(detections, candidates, image.size, regions,
                                       baseline_points=[points for _, points in baseline_resolved])
    count = len(resolved) + len(unresolved)
    overflow = count > MAX_PEOPLE
    if len(resolved) > MAX_PEOPLE:
        unresolved.extend(box for box, _ in resolved[MAX_PEOPLE:])
        resolved = resolved[:MAX_PEOPLE]
    warnings = photo_warnings(image)
    if not count:
        warnings.append('No people could be identified. Try a clearer photo showing everyone, including their heads and bodies.')
    if unresolved:
        warnings.append(f'{len(unresolved)} detected or indicated person region(s) could not be reconstructed reliably. Everyone must be resolved before generating the group.')
    if overflow:
        warnings.append('This photo exceeds the six-person limit. Use a photo with six or fewer people; no partial group will be generated.')
    subjects = []
    for index, (bbox, points) in enumerate(resolved):
        confidence = float(np.mean([points[k]['visibility'] for k in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)]))
        subject = dict(id=f'person-{index + 1}', name=f'Person {index + 1}',
                       photo_aspect=image.width / image.height, landmarks=points,
                       confidence=confidence, bbox=bbox,
                       warnings=_subject_warnings(points, bbox, confidence, image.size))
        for other_bbox, _ in resolved:
            if other_bbox is not bbox and people._iou(bbox, other_bbox) > .1:
                subject['warnings'].append('People overlap. Hidden limbs and which person holds an object may be uncertain.')
                break
        subject['appearance'], subject['appearance_detection'] = appearance.analyze_person(image, subject)
        subjects.append(subject)
    suggest_layout(subjects, image.size)
    return dict(subjects=subjects, detected_people=min(count, 100),
                unresolved_regions=unresolved[:100], warnings=warnings,
                detection=group_status(), overflow=overflow)

def close():
    global _detector
    with _lock:
        if _detector is not None:
            detector, _detector = _detector, None
            detector.close()


atexit.register(close)
