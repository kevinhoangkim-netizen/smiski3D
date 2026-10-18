"""Install the pinned SCHP ATR-18 clothing segmentation model.

This explicit setup command downloads public weights only. Inference reads local
files, and photos are never sent to a server. Both fresh and reused model files
must match the pinned size and SHA-256; the local manifest cannot override them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILENAME = 'schp-atr-18-int8-static.onnx'
REVISION = '61f4ff3c508b899904e495cefbbaff9cacadeb45'
MODEL_URL = ('https://huggingface.co/pirocheto/schp-atr-18/resolve/'
             + REVISION + '/onnx/' + MODEL_FILENAME)
EXPECTED_SHA256 = '4420d8db8c1f266967c89485786b01209f6d405f320fc0f87e8ced49392cefb5'
EXPECTED_SIZE = 69_141_996
CHUNK_SIZE = 1024 * 1024


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verified(path):
    return (path.is_file() and path.stat().st_size == EXPECTED_SIZE
            and digest(path) == EXPECTED_SHA256)


def install(target=ROOT / '.local/garment'):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    final_path = target / MODEL_FILENAME
    temporary = final_path.with_suffix(final_path.suffix + '.download')
    # A previous interrupted setup must not leave a plausible partial model.
    temporary.unlink(missing_ok=True)
    if verified(final_path):
        print('Existing local clothing/skin model checksum verified; no download needed.')
    else:
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=120) as response, temporary.open('wb') as output:
                while chunk := response.read(CHUNK_SIZE):
                    output.write(chunk)
            if digest(temporary) != EXPECTED_SHA256:
                raise RuntimeError('SHA-256 mismatch for clothing/skin model; refusing model')
            if temporary.stat().st_size != EXPECTED_SIZE:
                raise RuntimeError('Unexpected clothing/skin model size; refusing model')
            temporary.replace(final_path)
            print(f'Verified {MODEL_FILENAME}: {EXPECTED_SIZE / 1024 / 1024:.1f} MiB')
        finally:
            temporary.unlink(missing_ok=True)
    manifest = {
        'model': 'SCHP ATR-18 quantized ONNX',
        'revision': REVISION,
        'files': {MODEL_FILENAME: EXPECTED_SHA256},
        'source': MODEL_URL,
        'license': 'MIT',
        'provenance': 'PeikeLi/Self-Correction-Human-Parsing; pirocheto/schp-atr-18 ONNX conversion',
        'license_source': 'https://github.com/PeikeLi/Self-Correction-Human-Parsing/blob/master/LICENSE',
        'license_notice': 'See scripts/model_licenses/SCHP-MIT.txt in the app source',
    }
    manifest_path = target / 'manifest.json'
    temporary_manifest = manifest_path.with_suffix('.json.download')
    try:
        temporary_manifest.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        temporary_manifest.replace(manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    print('Local clothing/skin parsing installed. Photos are never sent to Hugging Face.')
    return final_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, default=ROOT / '.local/garment')
    arguments = parser.parse_args()
    install(arguments.target)
