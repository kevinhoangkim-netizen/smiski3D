"""Meaningful update safety tests using small, fully verified local releases."""
import hashlib
import json
from pathlib import Path
import stat
import zipfile

import pytest

from scripts import update_local as updater


def sha(content):
    return hashlib.sha256(content).hexdigest()


@pytest.fixture
def installed(tmp_path):
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts/launch_local.py').write_text('old launcher')
    (tmp_path / 'Start Little Figures.bat').write_text('old batch')
    (tmp_path / 'frontend/dist').mkdir(parents=True)
    (tmp_path / 'frontend/dist/index.html').write_text('old website')
    (tmp_path / '.local/appearance').mkdir(parents=True)
    (tmp_path / '.local/appearance/model.onnx').write_bytes(b'my large downloaded model')
    (tmp_path / '.venv').mkdir()
    (tmp_path / '.venv/installed.txt').write_text('private environment')
    (tmp_path / 'my-project.json').write_text('my saved project')
    return tmp_path


def release(tmp_path, files=None, name='release.zip'):
    files = files or {'scripts/launch_local.py': b'new launcher', 'frontend/dist/index.html': b'new website'}
    archive = tmp_path / name
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for path, content in files.items():
            output.writestr(path, content)
    manifest = {'schema': 1, 'version': '3.1.0',
                'archive_url': updater.REPOSITORY_PREFIX + 'Little-Figures-Update-3.1.0.zip',
                'archive_sha256': updater.digest(archive), 'archive_size': archive.stat().st_size,
                'files': [{'path': path, 'size': len(content), 'sha256': sha(content)} for path, content in files.items()]}
    return manifest, archive


def stage(tmp_path, manifest, archive):
    updater.validate_manifest(manifest)
    target = tmp_path / 'staged'
    updater.stage_release(archive, manifest, target)
    return target


def test_changed_files_only_preserve_models_projects_and_environment(installed):
    root = installed
    unchanged = root / 'scripts/unchanged.py'
    unchanged.write_bytes(b'same')
    before = unchanged.stat().st_mtime_ns
    manifest, archive = release(root, {'scripts/launch_local.py': b'new launcher',
                                      'frontend/dist/index.html': b'new website',
                                      'scripts/unchanged.py': b'same'})
    staged = stage(root, manifest, archive)
    operations = updater.plan_release(root, manifest)
    assert len(operations) == 2
    backup = updater.apply_release(root, staged, manifest, operations)
    assert unchanged.stat().st_mtime_ns == before
    assert (root / '.local/appearance/model.onnx').read_bytes() == b'my large downloaded model'
    assert (root / '.venv/installed.txt').read_text() == 'private environment'
    assert (root / 'my-project.json').read_text() == 'my saved project'
    assert (backup / 'files/scripts/launch_local.py').read_text() == 'old launcher'
    assert (root / 'scripts/launch_local.py').read_text() == 'new launcher'
    assert updater.plan_release(root, manifest) == []


def test_only_previously_managed_unchanged_obsolete_files_removed(installed):
    root = installed
    managed = root / 'frontend/dist/old.js'
    managed.write_bytes(b'old asset')
    modified = root / 'frontend/dist/modified.js'
    modified.write_bytes(b'user edited')
    untracked = root / 'frontend/dist/user.js'
    untracked.write_bytes(b'untracked')
    updater.write_json(root / '.local/update-state.json', {'files': [
        {'path': 'frontend/dist/old.js', 'sha256': sha(b'old asset')},
        {'path': 'frontend/dist/modified.js', 'sha256': sha(b'original')},
    ]})
    manifest, archive = release(root)
    staged = stage(root, manifest, archive)
    backup = updater.apply_release(root, staged, manifest)
    assert not managed.exists()
    assert modified.read_bytes() == b'user edited'
    assert untracked.read_bytes() == b'untracked'
    updater.restore_backup(root, backup)
    assert managed.read_bytes() == b'old asset'
    assert (root / 'scripts/launch_local.py').read_text() == 'old launcher'


def test_mid_apply_failure_rolls_back_all_changed_and_new_files(installed, monkeypatch):
    root = installed
    manifest, archive = release(root, {'scripts/launch_local.py': b'new launcher',
                                      'scripts/new.py': b'new file',
                                      'frontend/dist/index.html': b'new website'})
    staged = stage(root, manifest, archive)
    real_write = updater.atomic_bytes
    failed = False

    def fail_one(path, content):
        nonlocal failed
        if path == root / 'frontend/dist/index.html' and not failed:
            failed = True
            raise OSError('simulated locked Windows app file')
        return real_write(path, content)

    monkeypatch.setattr(updater, 'atomic_bytes', fail_one)
    with pytest.raises(updater.UpdateError, match='restored'):
        updater.apply_release(root, staged, manifest)
    assert (root / 'scripts/launch_local.py').read_text() == 'old launcher'
    assert (root / 'frontend/dist/index.html').read_text() == 'old website'
    assert not (root / 'scripts/new.py').exists()
    assert not (root / '.local/update-state.json').exists()


@pytest.mark.parametrize('path', ['../scripts/evil.py', '/scripts/evil.py', 'C:/scripts/evil.py',
                                 'scripts/../evil.py', 'scripts\\evil.py', '.local/model.onnx',
                                 '.venv/python.exe', 'my-project.json', 'scripts/CON.txt',
                                 'scripts/a.', 'scripts/a ', 'scripts/a:stream', 'scripts/NUL',
                                 'scripts//bad.py', 'scripts/a\x00.py'])
def test_rejects_unsafe_or_private_release_paths(path):
    with pytest.raises(updater.UpdateError):
        updater.safe_path(path)


def test_manifest_rejects_case_collisions(installed):
    manifest, _ = release(installed, {'scripts/launch_local.py': b'a', 'frontend/dist/index.html': b'b',
                                     'scripts/THING.py': b'c', 'scripts/thing.py': b'd'})
    with pytest.raises(updater.UpdateError, match='duplicate Windows'):
        updater.validate_manifest(manifest)


def test_manifest_rejects_file_folder_conflict(installed):
    manifest, _ = release(installed, {'scripts/launch_local.py': b'a', 'frontend/dist/index.html': b'b',
                                     'scripts/thing': b'c', 'scripts/thing/child.py': b'd'})
    with pytest.raises(updater.UpdateError, match='file and a folder'):
        updater.validate_manifest(manifest)


@pytest.mark.parametrize('url', ['http://raw.githubusercontent.com/kevinhoangkim-netizen/smiski3D/main/downloads/releases/a.zip',
                               updater.REPOSITORY_PREFIX + '../a.zip',
                               updater.REPOSITORY_PREFIX + 'a.zip?token=example',
                               'https://evil.example/a.zip'])
def test_manifest_rejects_untrusted_archive_url(installed, url):
    manifest, _ = release(installed)
    manifest['archive_url'] = url
    with pytest.raises(updater.UpdateError, match='official HTTPS'):
        updater.validate_manifest(manifest)


def test_archive_extra_file_rejected_before_app_changes(installed):
    manifest, archive = release(installed)
    with zipfile.ZipFile(archive, 'a') as output:
        output.writestr('scripts/unexpected.py', b'bad')
    manifest['archive_size'] = archive.stat().st_size
    manifest['archive_sha256'] = updater.digest(archive)
    with pytest.raises(updater.UpdateError, match='file list'):
        stage(installed, manifest, archive)
    assert (installed / 'scripts/launch_local.py').read_text() == 'old launcher'


def test_archive_hash_mismatch_rejected(installed):
    manifest, archive = release(installed)
    manifest['archive_sha256'] = '0' * 64
    with pytest.raises(updater.UpdateError, match='checksum'):
        stage(installed, manifest, archive)
    assert (installed / 'scripts/launch_local.py').read_text() == 'old launcher'


def test_per_file_hash_mismatch_rejected(installed):
    manifest, archive = release(installed)
    manifest['files'][0]['sha256'] = '0' * 64
    with pytest.raises(updater.UpdateError, match='checksum'):
        stage(installed, manifest, archive)


def test_zip_symlink_rejected(installed):
    manifest, archive = release(installed)
    with zipfile.ZipFile(archive, 'w') as output:
        for entry in manifest['files']:
            info = zipfile.ZipInfo(entry['path'])
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            output.writestr(info, b'new launcher' if entry['path'].startswith('scripts') else b'new website')
    manifest['archive_size'] = archive.stat().st_size
    manifest['archive_sha256'] = updater.digest(archive)
    with pytest.raises(updater.UpdateError, match='Links'):
        stage(installed, manifest, archive)


def test_existing_symlink_rejected(installed):
    target = installed / 'elsewhere'
    target.mkdir()
    (installed / 'scripts/launch_local.py').unlink()
    (installed / 'scripts/launch_local.py').symlink_to(target / 'private.py')
    manifest, _ = release(installed)
    with pytest.raises(updater.UpdateError, match='symbolic link'):
        updater.plan_release(installed, manifest)


def test_network_failure_leaves_existing_app_untouched(installed):
    def unavailable(*args):
        raise updater.UpdateError('network unavailable')
    with pytest.raises(updater.UpdateError, match='network unavailable'):
        updater.run_update(installed, no_launch=True, fetch=unavailable)
    assert (installed / 'scripts/launch_local.py').read_text() == 'old launcher'
    assert (installed / 'frontend/dist/index.html').read_text() == 'old website'
    assert not (installed / '.local/update-state.json').exists()


def test_check_downloads_manifest_only_and_does_not_stop_or_change_app(installed, monkeypatch):
    manifest, _ = release(installed)
    requested = []
    def fetch(url, target, limit):
        requested.append(url)
        target.write_text(json.dumps(manifest))
    monkeypatch.setattr(updater, 'stop_running_app', lambda *_: pytest.fail('check must not stop app'))
    result = updater.run_update(installed, check=True, fetch=fetch)
    assert result['changes'] == 2
    assert requested == [updater.MANIFEST_URL]
    assert (installed / 'scripts/launch_local.py').read_text() == 'old launcher'


def test_whole_verified_update_can_run_without_launch(installed, monkeypatch):
    manifest, archive = release(installed)
    def fetch(url, target, limit):
        target.write_bytes(json.dumps(manifest).encode() if url == updater.MANIFEST_URL else archive.read_bytes())
    monkeypatch.setattr(updater, 'stop_running_app', lambda *_: 8765)
    monkeypatch.setattr(updater, 'launch_app', lambda *_: pytest.fail('no-launch must not launch'))
    result = updater.run_update(installed, no_launch=True, fetch=fetch)
    assert result['changes'] == 2
    assert (installed / 'scripts/launch_local.py').read_text() == 'new launcher'


def test_legacy_running_app_aborts_without_killing_any_process(installed, monkeypatch):
    monkeypatch.setattr(updater, 'app_health', lambda port: {'app': 'little-figures-studio', 'version': '3.0.0'} if port == 8765 else None)
    with pytest.raises(updater.UpdateError, match='close the existing'):
        updater.stop_running_app(installed)


def test_authenticated_stop_uses_only_matching_install_token(installed, monkeypatch):
    updater.write_json(installed / '.local/runtime.json', {'app': 'little-figures-studio', 'version': '3.1.0',
                       'port': 8780, 'root': str(installed), 'update_token': 'private-token-' * 4})
    running = True
    requests = []
    def health(port):
        return {'app': 'little-figures-studio', 'version': '3.1.0'} if running and port == 8780 else None
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
    def open_request(request, timeout):
        nonlocal running
        requests.append(request)
        running = False
        return Response()
    monkeypatch.setattr(updater, 'app_health', health)
    monkeypatch.setattr(updater.HEALTH, 'open', open_request)
    assert updater.stop_running_app(installed) == 8780
    assert requests[0].full_url == 'http://127.0.0.1:8780/api/local-update/stop'
    assert requests[0].get_header('X-local-update-token') == 'private-token-' * 4


def test_update_lock_rejects_concurrent_update(installed):
    with updater.updater_lock(installed):
        with pytest.raises(updater.UpdateError, match='already running'):
            with updater.updater_lock(installed):
                pass


def test_standalone_batch_payload_is_exact_python_source(tmp_path):
    source = Path(updater.__file__)
    batch = tmp_path / 'Update Little Figures.bat'
    updater.build_batch(source, batch)
    payload = batch.read_text(encoding='utf-8').split('\n' + updater.BATCH_MARKER + '\n', 1)[1]
    assert payload == source.read_text(encoding='utf-8')
    compile(payload, str(batch), 'exec')
    assert 'exit /b 0' in batch.read_text()
    committed = source.parents[1] / 'Update Little Figures.bat'
    assert committed.read_bytes() == batch.read_bytes(), 'Regenerate the standalone updater after editing Python.'


def test_authenticated_stop_waits_for_recorded_process_after_health_disappears(installed, monkeypatch):
    updater.write_json(installed / '.local/runtime.json', {'app': 'little-figures-studio', 'version': '3.1.0',
                       'port': 8780, 'pid': 123456, 'root': str(installed), 'update_token': 'private-token-' * 4})
    running = True
    process_checks = []
    sleeps = []
    def health(port):
        return {'app': 'little-figures-studio', 'version': '3.1.0'} if running and port == 8780 else None
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
    def open_request(*args, **kwargs):
        nonlocal running
        running = False
        return Response()
    def process_running(pid):
        process_checks.append(pid)
        return len(process_checks) < 3
    monkeypatch.setattr(updater, 'app_health', health)
    monkeypatch.setattr(updater.HEALTH, 'open', open_request)
    monkeypatch.setattr(updater, 'process_running', process_running)
    monkeypatch.setattr(updater.time, 'sleep', sleeps.append)
    updater.stop_running_app(installed)
    assert process_checks == [123456] * 3
    assert sleeps == [.25, .25]


def test_keyboard_interrupt_during_apply_restores_files(installed, monkeypatch):
    manifest, archive = release(installed)
    staged = stage(installed, manifest, archive)
    real_write = updater.atomic_bytes
    interrupted = False
    def interrupt_one(path, content):
        nonlocal interrupted
        if path == installed / 'frontend/dist/index.html' and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        return real_write(path, content)
    monkeypatch.setattr(updater, 'atomic_bytes', interrupt_one)
    with pytest.raises(updater.UpdateError, match='restored'):
        updater.apply_release(installed, staged, manifest)
    assert (installed / 'scripts/launch_local.py').read_text() == 'old launcher'
    assert (installed / 'frontend/dist/index.html').read_text() == 'old website'


def test_interrupted_prepared_update_restores_before_next_network_request(installed, monkeypatch):
    manifest, archive = release(installed)
    staged = stage(installed, manifest, archive)
    backup = updater.apply_release(installed, staged, manifest)
    record = json.loads((backup / 'backup.json').read_text())
    record['status'] = 'prepared'  # Represents a process stopped before its completion marker.
    updater.write_json(backup / 'backup.json', record)
    monkeypatch.setattr(updater, 'stop_running_app', lambda *_: 8765)
    def offline(*args):
        assert (installed / 'scripts/launch_local.py').read_text() == 'old launcher'
        raise updater.UpdateError('offline after recovery')
    with pytest.raises(updater.UpdateError, match='offline after recovery'):
        updater.run_update(installed, no_launch=True, fetch=offline)
    assert json.loads((backup / 'backup.json').read_text())['status'] == 'rolled-back'


def test_prepared_backup_can_be_rolled_back_explicitly(installed, monkeypatch):
    manifest, archive = release(installed)
    staged = stage(installed, manifest, archive)
    backup = updater.apply_release(installed, staged, manifest)
    record = json.loads((backup / 'backup.json').read_text())
    record['status'] = 'prepared'
    updater.write_json(backup / 'backup.json', record)
    monkeypatch.setattr(updater, 'stop_running_app', lambda *_: 8765)
    updater.rollback_latest(installed, no_launch=True)
    assert (installed / 'scripts/launch_local.py').read_text() == 'old launcher'


def test_case_only_rename_is_never_scheduled_for_delete(installed):
    (installed / 'scripts/Old.py').write_bytes(b'same')
    updater.write_json(installed / '.local/update-state.json', {'files': [
        {'path': 'scripts/Old.py', 'sha256': sha(b'same')},
    ]})
    manifest, _ = release(installed, {'scripts/launch_local.py': b'new',
                                     'frontend/dist/index.html': b'new', 'scripts/old.py': b'same'})
    assert all(operation['action'] != 'delete' for operation in updater.plan_release(installed, manifest))


def test_live_recorded_process_without_health_never_allows_update(installed, monkeypatch):
    updater.write_json(installed / '.local/runtime.json', {'app': 'little-figures-studio', 'version': '3.1.0',
                       'port': 8780, 'pid': 123456, 'root': str(installed), 'update_token': 'private-token-' * 4})
    monkeypatch.setattr(updater, 'app_health', lambda *_: None)
    monkeypatch.setattr(updater, 'process_running', lambda pid: pid == 123456)
    with pytest.raises(updater.UpdateError, match='still starting or closing'):
        updater.stop_running_app(installed)
