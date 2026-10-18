"""Install public quantized CLIP weights with HTTPS and upstream SHA-256 checks.

This setup command only downloads model files. Photos never leave the app.
The installed manifest pins the exact upstream revision and verifies local files
again before inference. Existing valid files are reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPOSITORY = 'Xenova/clip-vit-base-patch32'
DEFAULT_REVISION = 'd15189d7028b43f1d3e65039190477f6af591c2a'
FILES = {'onnx/vision_model_quantized.onnx': 'vision_model_quantized.onnx',
         'onnx/text_model_quantized.onnx': 'text_model_quantized.onnx',
         'tokenizer.json': 'tokenizer.json'}
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def install(target, revision=None):
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / 'manifest.json'
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if existing and revision is None and set(existing.get('files', {})) == set(FILES.values()):
        if all((target / name).is_file() and digest(target / name) == expected
               for name, expected in existing['files'].items()):
            print('Existing local outfit model checksums verified; no download needed.')
            return
    if existing and revision is None:
        revision = existing['revision']
    if revision is None:
        revision = DEFAULT_REVISION
    endpoint = f'https://huggingface.co/api/models/{REPOSITORY}'
    if revision:
        endpoint += '/revision/' + revision
    with urllib.request.urlopen(endpoint + '?blobs=true', timeout=60) as response:
        metadata = json.load(response)
    revision = metadata['sha']
    if len(revision) != 40 or not all(c in '0123456789abcdef' for c in revision):
        raise RuntimeError('Invalid upstream model revision')
    entries = {item['rfilename']: item for item in metadata['siblings']}
    checksums = {}
    for remote, local in FILES.items():
        entry = entries[remote]
        # LFS models expose SHA-256 directly. Small Git blobs are verified against
        # their Git SHA-1 object ID and stored with SHA-256 in our local manifest.
        expected = entry.get('lfs', {}).get('sha256')
        final_path = target / local
        if expected and final_path.exists() and digest(final_path) == expected:
            checksums[local] = expected
            print(f'Reusing verified {local}')
            continue
        temporary = final_path.with_suffix(final_path.suffix + '.download')
        try:
            url = f'https://huggingface.co/{REPOSITORY}/resolve/{revision}/{remote}'
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            actual = digest(temporary)
            if expected:
                if actual != expected:
                    raise RuntimeError(f'SHA-256 mismatch for {local}; refusing model')
            else:
                contents = temporary.read_bytes()
                git_digest = hashlib.sha1(f'blob {len(contents)}\0'.encode() + contents).hexdigest()
                if git_digest != entry['blobId']:
                    raise RuntimeError(f'Git object checksum mismatch for {local}; refusing model')
            temporary.replace(final_path)
            checksums[local] = actual
            print(f'Verified {local}: {final_path.stat().st_size / 1024 / 1024:.1f} MiB')
        finally:
            temporary.unlink(missing_ok=True)
    manifest = {'repository': REPOSITORY, 'revision': revision, 'files': checksums,
                'source': f'https://huggingface.co/{REPOSITORY}/tree/{revision}',
                'license': 'MIT (OpenAI CLIP weights; see upstream model card)'}
    temporary_manifest = manifest_path.with_suffix('.json.download')
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary_manifest.replace(manifest_path)
    print('Local outfit recognition installed. Photos are never sent to Hugging Face.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, default=ROOT / '.local/appearance')
    parser.add_argument('--revision', help='Optional upstream Git commit to pin')
    arguments = parser.parse_args()
    install(arguments.target, arguments.revision)
