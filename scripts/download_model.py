"""Download the official MediaPipe model over verified HTTPS. No photo data is sent."""
from pathlib import Path
import urllib.request
import hashlib
root=Path(__file__).resolve().parents[1]
target=root/'.local/pose_landmarker_lite.task'
url='https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
target.parent.mkdir(exist_ok=True)
if not target.exists():
    temporary=target.with_suffix('.download')
    try:
        urllib.request.urlretrieve(url,temporary)
        if temporary.stat().st_size<1_000_000:
            raise RuntimeError('Unexpectedly small model download; refusing to use it.')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
print(f'Model: {target.name}; SHA256: {hashlib.sha256(target.read_bytes()).hexdigest()}')
