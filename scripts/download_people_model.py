"""Install pinned Apache-2.0 YOLOX-S weights over verified HTTPS.

No photos are sent. The exact SHA-256 is pinned in the application and checked
before installation and before local inference. CPU ONNX Runtime is sufficient.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.people import FILENAME, MODEL_PATH, REPOSITORY, REVISION, SHA256  # noqa: E402


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def install(target=MODEL_PATH):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and digest(target) == SHA256:
        print('Existing local group detection model checksum verified; no download needed.')
        return
    temporary = target.with_suffix('.onnx.download')
    try:
        url = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{FILENAME}'
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if digest(temporary) != SHA256:
            raise RuntimeError('Group detector SHA-256 mismatch; refusing model installation.')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Local group detection installed: {target.stat().st_size / 1024 / 1024:.1f} MiB; SHA256 {SHA256}')
    print('Model: YOLOX-S, Apache-2.0. Photos are never sent to the model host.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, default=MODEL_PATH)
    install(parser.parse_args().target)
