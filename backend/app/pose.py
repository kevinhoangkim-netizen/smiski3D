"""Local, multi-person MediaPipe inference; image bytes never leave this process."""
import os
from pathlib import Path
from threading import Lock
import numpy as np

MODEL_PATH = Path(os.environ.get('POSE_MODEL_PATH', str(Path(__file__).resolve().parents[2]/'.local/pose_landmarker_lite.task')))
_lock=Lock()
_detector=None

def available():
    if not MODEL_PATH.is_file():
        return False, 'Pose model is not installed. Choose a manual preset, or run scripts/download_model.py.'
    try:
        import mediapipe
        return True, 'Local MediaPipe multi-person pose estimation. No photo upload to third parties.'
    except (ImportError, OSError) as exc:
        return False, f'Pose runtime unavailable ({type(exc).__name__}). Choose a manual preset.'

def analyze(image):
    global _detector
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    with _lock:
        if _detector is None:
            options=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=vision.RunningMode.IMAGE,num_poses=3,min_pose_detection_confidence=.45,
                min_pose_presence_confidence=.45,min_tracking_confidence=.45)
            _detector=vision.PoseLandmarker.create_from_options(options)
        result=_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.asarray(image.convert('RGB'))))
    subjects=[]
    for i,landmarks in enumerate(result.pose_landmarks):
        points=[dict(x=float(p.x),y=float(p.y),z=float(p.z),visibility=float(p.visibility or 0)) for p in landmarks]
        conf=float(np.mean([points[k]['visibility'] for k in [11,12,13,14,15,16,23,24,25,26,27,28]]))
        visible=[p for p in points if p['visibility']>.5]
        bbox=[min(p['x'] for p in visible),min(p['y'] for p in visible),max(p['x'] for p in visible),max(p['y'] for p in visible)] if visible else [0,0,1,1]
        subjects.append(dict(id=f'person-{i+1}',name=f'Person {i+1}',landmarks=points,confidence=conf,bbox=bbox,
            warnings=['Single-photo depth is ambiguous; canonical anatomy and conservative depth are used.']+(['Occluded or uncertain limbs use a neutral fallback.'] if conf<.8 else [])))
    return subjects


def close():
    global _detector
    with _lock:
        if _detector is not None:
            detector,_detector=_detector,None
            detector.close()

import atexit
atexit.register(close)
