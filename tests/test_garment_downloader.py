"""Pinned model reuse, tampering and interrupted-download boundaries."""
import hashlib
from io import BytesIO
import json

import pytest

from scripts import download_garment_model as downloader


@pytest.fixture
def model(monkeypatch):
    contents = b'verified local segmentation model'
    monkeypatch.setattr(downloader, 'EXPECTED_SHA256', hashlib.sha256(contents).hexdigest())
    monkeypatch.setattr(downloader, 'EXPECTED_SIZE', len(contents))
    return contents


def test_reuse_verifies_pinned_file_and_repairs_missing_manifest(monkeypatch, tmp_path, model):
    path = tmp_path / downloader.MODEL_FILENAME
    path.write_bytes(model)
    stale_partial = path.with_suffix(path.suffix + '.download')
    stale_partial.write_bytes(b'interrupted download')

    def no_network(*args, **kwargs):
        pytest.fail('Verified local model must not use the network')

    monkeypatch.setattr(downloader.urllib.request, 'urlopen', no_network)
    assert downloader.install(tmp_path) == path
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['files'] == {downloader.MODEL_FILENAME: downloader.EXPECTED_SHA256}
    assert manifest['source'] == downloader.MODEL_URL
    assert manifest['revision'] == downloader.REVISION and manifest['license'] == 'MIT'
    assert 'PeikeLi/Self-Correction-Human-Parsing' in manifest['provenance']
    assert not stale_partial.exists()


def test_corrupt_model_is_replaced_even_if_manifest_claims_it_is_valid(monkeypatch, tmp_path, model):
    path = tmp_path / downloader.MODEL_FILENAME
    corrupt = b'x' * len(model)
    path.write_bytes(corrupt)
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'files': {downloader.MODEL_FILENAME: hashlib.sha256(corrupt).hexdigest()},
    }))
    calls = []

    def download(url, timeout):
        calls.append((url, timeout))
        return BytesIO(model)

    monkeypatch.setattr(downloader.urllib.request, 'urlopen', download)
    downloader.install(tmp_path)
    assert path.read_bytes() == model
    assert calls == [(downloader.MODEL_URL, 120)]
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['files'][downloader.MODEL_FILENAME] == downloader.EXPECTED_SHA256


def test_bad_checksum_is_never_installed_and_partial_is_removed(monkeypatch, tmp_path, model):
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', lambda *args, **kwargs: BytesIO(b'bad model'))
    with pytest.raises(RuntimeError, match='SHA-256 mismatch'):
        downloader.install(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_failed_redownload_preserves_existing_file_without_trusting_it(monkeypatch, tmp_path, model):
    path = tmp_path / downloader.MODEL_FILENAME
    path.write_bytes(b'corrupted existing model')
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', lambda *args, **kwargs: BytesIO(b'bad replacement'))
    with pytest.raises(RuntimeError, match='SHA-256 mismatch'):
        downloader.install(tmp_path)
    assert path.read_bytes() == b'corrupted existing model'
    assert not downloader.verified(path)
    assert list(tmp_path.iterdir()) == [path]


def test_network_failure_cleans_partial_download(monkeypatch, tmp_path, model):
    class Interrupted(BytesIO):
        reads = 0

        def read(self, size):
            assert size == 1024 * 1024
            self.reads += 1
            if self.reads > 1:
                raise OSError('connection interrupted')
            return b'partial model'

    monkeypatch.setattr(downloader.urllib.request, 'urlopen', lambda *args, **kwargs: Interrupted())
    with pytest.raises(OSError, match='connection interrupted'):
        downloader.install(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_unexpected_size_is_rejected_even_with_matching_digest(monkeypatch, tmp_path, model):
    monkeypatch.setattr(downloader, 'EXPECTED_SIZE', len(model) + 1)
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', lambda *args, **kwargs: BytesIO(model))
    with pytest.raises(RuntimeError, match='Unexpected clothing/skin model size'):
        downloader.install(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_pinned_model_source_is_versioned_and_has_fixed_digest():
    assert downloader.REVISION in downloader.MODEL_URL
    assert downloader.MODEL_URL.startswith('https://huggingface.co/pirocheto/schp-atr-18/resolve/')
    assert downloader.EXPECTED_SIZE == 69_141_996
    assert downloader.EXPECTED_SHA256 == '4420d8db8c1f266967c89485786b01209f6d405f320fc0f87e8ced49392cefb5'
