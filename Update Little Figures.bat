@echo off
setlocal
cd /d "%~dp0"
title Update Little Figures Studio
set "LITTLE_FIGURES_UPDATE_FILE=%~f0"
set "LITTLE_FIGURES_UPDATE_ROOT=%~dp0"
py -3.12 --version >nul 2>&1
if not errorlevel 1 (
  py -3.12 -c "import os,pathlib; p=pathlib.Path(os.environ['LITTLE_FIGURES_UPDATE_FILE']); exec(compile(p.read_text(encoding='utf-8').split(chr(35)+' LITTLE_FIGURES_UPDATER_PYTHON_PAYLOAD',1)[1],str(p),'exec'))" --root "%LITTLE_FIGURES_UPDATE_ROOT%." %*
  if errorlevel 1 (
    echo.
    echo Update did not finish. Read the message above, then try again.
    pause
    exit /b 1
  )
  echo.
  echo You can close this updater window.
  pause
  exit /b 0
)
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
  "%LocalAppData%\Programs\Python\Python312\python.exe" -c "import os,pathlib; p=pathlib.Path(os.environ['LITTLE_FIGURES_UPDATE_FILE']); exec(compile(p.read_text(encoding='utf-8').split(chr(35)+' LITTLE_FIGURES_UPDATER_PYTHON_PAYLOAD',1)[1],str(p),'exec'))" --root "%LITTLE_FIGURES_UPDATE_ROOT%." %*
  if errorlevel 1 (
    echo.
    echo Update did not finish. Read the message above, then try again.
    pause
    exit /b 1
  )
  echo.
  echo You can close this updater window.
  pause
  exit /b 0
)
if exist "%ProgramFiles%\Python312\python.exe" (
  "%ProgramFiles%\Python312\python.exe" -c "import os,pathlib; p=pathlib.Path(os.environ['LITTLE_FIGURES_UPDATE_FILE']); exec(compile(p.read_text(encoding='utf-8').split(chr(35)+' LITTLE_FIGURES_UPDATER_PYTHON_PAYLOAD',1)[1],str(p),'exec'))" --root "%LITTLE_FIGURES_UPDATE_ROOT%." %*
  if errorlevel 1 (
    echo.
    echo Update did not finish. Read the message above, then try again.
    pause
    exit /b 1
  )
  echo.
  echo You can close this updater window.
  pause
  exit /b 0
)
echo Python 3.12 is needed. Run Start Little Figures.bat to finish setup first.
pause
exit /b 1

# LITTLE_FIGURES_UPDATER_PYTHON_PAYLOAD
"""Safely update the installed Windows app without replacing private local data.

This uses only Python's standard library so the downloadable batch file can
carry this entire program and run before the app's dependencies are installed.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile

MANIFEST_URL = 'https://raw.githubusercontent.com/kevinhoangkim-netizen/smiski3D/main/downloads/latest.json'
REPOSITORY_PREFIX = 'https://raw.githubusercontent.com/kevinhoangkim-netizen/smiski3D/main/downloads/releases/'
BATCH_MARKER = '# LITTLE_FIGURES_UPDATER_PYTHON_PAYLOAD'
MAX_ARCHIVE = 64 * 1024 * 1024
MAX_CONTENT = 256 * 1024 * 1024
MAX_FILE = 64 * 1024 * 1024
MAX_FILES = 10000
ROOT_FILES = {
    'Start Little Figures.bat', 'Update Little Figures.bat', 'START HERE - Windows.txt',
    'README.md', 'ARCHITECTURE.md', 'LIMITATIONS.md', 'app-version.json',
}
RESERVED = re.compile(r'^(CON|PRN|AUX|NUL|CLOCK\$|CONIN\$|CONOUT\$|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)', re.I)
HEX = re.compile(r'^[a-f0-9]{64}$')
HEALTH = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class UpdateError(RuntimeError):
    """An update could not safely complete."""


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def safe_path(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 240:
        raise UpdateError('The release contains an invalid file path.')
    if '\\' in value or ':' in value or '\x00' in value:
        raise UpdateError('The release contains an unsafe file path.')
    parts = value.split('/')
    if any(not p or p in {'.', '..'} or p[-1] in {' ', '.'} or RESERVED.match(p)
           or any(ord(c) < 32 or c in '<>"|?*' for c in p) for p in parts):
        raise UpdateError('The release contains an unsafe file path.')
    if PurePosixPath(value).is_absolute():
        raise UpdateError('Absolute release paths are not allowed.')
    permitted = (
        value in ROOT_FILES
        or (len(parts) >= 3 and parts[:2] in (['backend', 'app'], ['frontend', 'dist']))
        or (len(parts) >= 2 and parts[0] == 'scripts')
        or (len(parts) == 2 and parts[0] == 'backend'
            and re.fullmatch(r'requirements(?:[-.][A-Za-z0-9_-]+)*\.txt', parts[1]))
    )
    if not permitted or any(p.casefold() in {'.git', '.local', '.venv', '__pycache__', 'node_modules'} for p in parts):
        raise UpdateError('The release tried to change a private or unmanaged file.')
    return value


def checked_target(root: Path, relative: str) -> Path:
    relative = safe_path(relative)
    current = root
    for component in relative.split('/'):
        current = current / component
        if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
            raise UpdateError(f'Cannot update a symbolic link: {relative}')
    return current


def local_directory(root: Path, *components: str) -> Path:
    current = root
    for component in ('.local', *components):
        current = current / component
        if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
            raise UpdateError('The app update-data directory must not be a symbolic link or junction.')
        current.mkdir(exist_ok=True)
    return current


def allowed_url(url: str, *, manifest=False) -> bool:
    if not isinstance(url, str):
        return False
    if manifest:
        return url == MANIFEST_URL
    if not url.startswith(REPOSITORY_PREFIX):
        return False
    tail = url[len(REPOSITORY_PREFIX):]
    return bool(re.fullmatch(r'[A-Za-z0-9._-]+\.zip', tail))


class ReleaseRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        if not (allowed_url(newurl) or allowed_url(newurl, manifest=True)):
            raise UpdateError('The release redirected outside the official download location.')
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def download(url: str, destination: Path, limit: int) -> None:
    if not (allowed_url(url) or allowed_url(url, manifest=True)):
        raise UpdateError('Only official HTTPS release downloads are permitted.')
    opener = urllib.request.build_opener(ReleaseRedirectHandler())
    request = urllib.request.Request(url, headers={'User-Agent': 'Little-Figures-Updater/1'})
    try:
        with opener.open(request, timeout=45) as response, destination.open('wb') as output:
            if not (allowed_url(response.geturl()) or allowed_url(response.geturl(), manifest=True)):
                raise UpdateError('The release redirected outside the official download location.')
            count = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > limit:
                    raise UpdateError('The downloaded release is larger than the permitted limit.')
                output.write(chunk)
    except (OSError, urllib.error.URLError) as error:
        raise UpdateError('Could not download the update. Your existing app was not changed. Check your internet connection and try again.') from error


def validate_manifest(data: object) -> dict:
    if not isinstance(data, dict) or type(data.get('schema')) is not int or data['schema'] != 1:
        raise UpdateError('This updater cannot read the release manifest.')
    version = data.get('version')
    if not isinstance(version, str) or not re.fullmatch(r'\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?', version):
        raise UpdateError('The release version is invalid.')
    if not allowed_url(data.get('archive_url')):
        raise UpdateError('The release archive is not at the official HTTPS download location.')
    if not isinstance(data.get('archive_sha256'), str) or not HEX.fullmatch(data['archive_sha256']):
        raise UpdateError('The release archive checksum is missing or invalid.')
    size = data.get('archive_size')
    if type(size) is not int or not 0 < size <= MAX_ARCHIVE:
        raise UpdateError('The release archive size is invalid.')
    files = data.get('files')
    if not isinstance(files, list) or not 0 < len(files) <= MAX_FILES:
        raise UpdateError('The release file list is invalid.')
    paths = set()
    total = 0
    for entry in files:
        if not isinstance(entry, dict):
            raise UpdateError('The release contains an invalid file entry.')
        path = safe_path(entry.get('path'))
        key = path.casefold()
        if key in paths:
            raise UpdateError('The release contains duplicate Windows file paths.')
        paths.add(key)
        if type(entry.get('size')) is not int or not 0 <= entry['size'] <= MAX_FILE:
            raise UpdateError('A release file size is invalid.')
        if not isinstance(entry.get('sha256'), str) or not HEX.fullmatch(entry['sha256']):
            raise UpdateError('A release file checksum is invalid.')
        total += entry['size']
    if total > MAX_CONTENT:
        raise UpdateError('The unpacked release is too large.')
    for path in paths:
        components = path.split('/')
        if any('/'.join(components[:i]) in paths for i in range(1, len(components))):
            raise UpdateError('A release path is both a file and a folder.')
    if 'scripts/launch_local.py' not in paths or 'frontend/dist/index.html' not in paths:
        raise UpdateError('The release is missing required app files.')
    return data


def stage_release(archive: Path, manifest: dict, stage: Path) -> None:
    """Validate every byte before allowing any installed app file to change."""
    if archive.stat().st_size != manifest['archive_size'] or digest(archive) != manifest['archive_sha256']:
        raise UpdateError('The downloaded archive checksum did not match. Your app was not changed.')
    expected = {f['path']: f for f in manifest['files']}
    stage.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(archive) as source:
            infos = source.infolist()
            if len(infos) != len(expected):
                raise UpdateError('The archive file list does not match its manifest.')
            seen = set()
            for info in infos:
                path = safe_path(info.filename)
                if path not in expected or path.casefold() in seen or info.is_dir():
                    raise UpdateError('The archive contains unexpected or duplicate files.')
                seen.add(path.casefold())
                mode = info.external_attr >> 16
                if (mode and stat.S_IFMT(mode) not in (0, stat.S_IFREG)) or info.flag_bits & 1:
                    raise UpdateError('Links and encrypted archive entries are not permitted.')
                if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise UpdateError('Unsupported archive compression.')
                entry = expected[path]
                if info.file_size != entry['size'] or info.file_size > MAX_FILE:
                    raise UpdateError('An archive entry has the wrong size.')
                if info.file_size > 1024 * 1024 and info.file_size > max(1, info.compress_size) * 250:
                    raise UpdateError('An archive entry expands beyond the permitted limit.')
                target = stage / path
                target.parent.mkdir(parents=True, exist_ok=True)
                checksum = hashlib.sha256()
                written = 0
                with source.open(info) as incoming, target.open('wb') as output:
                    while True:
                        chunk = incoming.read(1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > entry['size']:
                            raise UpdateError('An archive entry expands beyond its declared size.')
                        checksum.update(chunk)
                        output.write(chunk)
                if written != entry['size'] or checksum.hexdigest() != entry['sha256']:
                    raise UpdateError('An extracted file checksum did not match its manifest.')
    except Exception as error:
        if isinstance(error, UpdateError):
            raise
        raise UpdateError('The release archive could not be verified. Your app was not changed.') from error


def write_json(path: Path, value: object) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2) + '\n').encode())


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.update-' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_state(root: Path) -> dict:
    location = root / '.local/update-state.json'
    if not location.exists():
        return {'files': []}
    try:
        state = json.loads(location.read_text(encoding='utf-8'))
        if not isinstance(state, dict) or not isinstance(state.get('files'), list):
            raise ValueError()
        for entry in state['files']:
            if not isinstance(entry, dict) or not HEX.fullmatch(entry.get('sha256', '')):
                raise ValueError()
            safe_path(entry.get('path'))
        return state
    except (OSError, ValueError, TypeError) as error:
        raise UpdateError('The previous update record could not be read. Your app was not changed.') from error


def plan_release(root: Path, manifest: dict) -> list[dict]:
    state = read_state(root)
    operations = []
    latest = {entry['path'].casefold() for entry in manifest['files']}
    for entry in manifest['files']:
        target = checked_target(root, entry['path'])
        if target.exists() and not target.is_file():
            raise UpdateError(f'A folder is in the way of an app file: {entry["path"]}')
        if not target.is_file() or target.stat().st_size != entry['size'] or digest(target) != entry['sha256']:
            operations.append({'path': entry['path'], 'action': 'replace', 'existed': target.exists()})
    for entry in state['files']:
        if entry['path'].casefold() in latest:
            continue
        target = checked_target(root, entry['path'])
        if target.is_file() and digest(target) == entry['sha256']:
            operations.append({'path': entry['path'], 'action': 'delete', 'existed': True})
        elif target.exists():
            print(f'Keeping a locally modified obsolete file: {entry["path"]}', flush=True)
    return operations


def restore_backup(root: Path, backup: Path) -> None:
    record = json.loads((backup / 'backup.json').read_text(encoding='utf-8'))
    errors = []
    for operation in reversed(record['operations']):
        try:
            target = checked_target(root, operation['path'])
            if operation['existed']:
                atomic_bytes(target, (backup / 'files' / operation['path']).read_bytes())
            else:
                target.unlink(missing_ok=True)
        except OSError as error:
            errors.append(str(error))
    state = root / '.local/update-state.json'
    if record['had_state']:
        atomic_bytes(state, (backup / 'update-state.json').read_bytes())
    else:
        state.unlink(missing_ok=True)
    if errors:
        raise UpdateError('Some backup files could not be restored. Close the app and retry rollback. Backup: ' + str(backup))
    record['status'] = 'rolled-back'
    write_json(backup / 'backup.json', record)


def apply_release(root: Path, stage: Path, manifest: dict, operations: list[dict] | None = None) -> Path:
    operations = operations if operations is not None else plan_release(root, manifest)
    identifier = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    backup = local_directory(root, 'updates', 'backups') / identifier
    backup.mkdir()
    state = root / '.local/update-state.json'
    record = {'schema': 1, 'version': manifest['version'], 'status': 'prepared',
              'had_state': state.exists(), 'operations': operations}
    for operation in operations:
        if operation['existed']:
            source = checked_target(root, operation['path'])
            target = backup / 'files' / operation['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    if state.exists():
        shutil.copy2(state, backup / 'update-state.json')
    write_json(backup / 'backup.json', record)
    try:
        for operation in operations:
            target = checked_target(root, operation['path'])
            if operation['action'] == 'delete':
                target.unlink()
            else:
                atomic_bytes(target, (stage / operation['path']).read_bytes())
        write_json(state, {'schema': 1, 'version': manifest['version'], 'files': manifest['files']})
        record['status'] = 'applied'
        write_json(backup / 'backup.json', record)
    except (Exception, KeyboardInterrupt) as error:
        try:
            restore_backup(root, backup)
        except Exception as rollback_error:
            raise UpdateError(f'The update could not finish or fully roll back. Your backup is at {backup}. Close the app, then run the updater with --rollback.') from rollback_error
        raise UpdateError('The update could not finish. Your previous app files were restored; close the app and try again.') from error
    return backup


@contextlib.contextmanager
def updater_lock(root: Path):
    local = local_directory(root)
    stream = (local / 'update.lock').open('a+b')
    try:
        stream.seek(0)
        if not stream.read(1):
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise UpdateError('Another update is already running. Wait for that window to finish.') from error
        yield
    finally:
        stream.close()


def app_health(port: int) -> dict | None:
    try:
        with HEALTH.open(f'http://127.0.0.1:{port}/api/health', timeout=.35) as response:
            data = json.load(response)
        return data if isinstance(data, dict) and data.get('app') == 'little-figures-studio' else None
    except (OSError, ValueError):
        return None


def process_running(pid: object) -> bool:
    """Query process state without ever sending a signal (including on Windows)."""
    if type(pid) is not int or pid <= 0:
        return False
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            # Access denied means that safe exit could not be established.
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
        finally:
            kernel.CloseHandle(handle)
    # Cloud verification runs on Linux. A zombie has exited even while its
    # launcher has not reaped it. Never use os.kill(pid, 0), which is unsafe on Windows.
    try:
        status = Path(f'/proc/{pid}/stat').read_text()
        return status.rsplit(')', 1)[1].strip().split()[0] != 'Z'
    except (OSError, IndexError):
        return False


def stop_running_app(root: Path) -> int:
    """Stop only this install through its authenticated loopback control."""
    runtime_file = root / '.local/runtime.json'
    metadata = None
    if runtime_file.is_file():
        try:
            value = json.loads(runtime_file.read_text(encoding='utf-8'))
            port = value.get('port')
            if (value.get('app') == 'little-figures-studio' and type(port) is int
                    and 1024 <= port <= 65535 and Path(value.get('root', '')).resolve() == root.resolve()):
                metadata = value
        except (OSError, ValueError, TypeError):
            pass
    port = metadata['port'] if metadata else 8765
    health = app_health(port) if metadata else None
    if metadata and not health and process_running(metadata.get('pid')):
        raise UpdateError('Little Figures is still starting or closing. Wait for its launcher to finish, or close that window, then run the updater again. Your app was not changed.')
    if health:
        token = metadata.get('update_token')
        if not isinstance(token, str) or len(token) < 20 or health.get('version') != metadata.get('version'):
            raise UpdateError('Please close the existing Little Figures launcher window, then run this updater again.')
        request = urllib.request.Request(f'http://127.0.0.1:{port}/api/local-update/stop',
                                         data=b'', method='POST', headers={'X-Local-Update-Token': token})
        try:
            with HEALTH.open(request, timeout=8) as response:
                if response.status != 200:
                    raise UpdateError('The running app could not stop safely.')
        except urllib.error.HTTPError as error:
            if error.code == 409:
                raise UpdateError('Finish the current photo analysis or model generation, then run the updater again. Your app was not changed.') from error
            raise UpdateError('Please close the existing Little Figures launcher window, then run this updater again.') from error
        except (OSError, urllib.error.URLError) as error:
            raise UpdateError('Please close the existing Little Figures launcher window, then run this updater again.') from error
        deadline = time.monotonic() + 30
        while app_health(port) or process_running(metadata.get('pid')):
            if time.monotonic() >= deadline:
                raise UpdateError('The app is still closing. Close its launcher window, then run this updater again.')
            time.sleep(.25)
    # Version 3 has no authenticated stop endpoint. Never replace its live files.
    for candidate in range(8765, 8785):
        if app_health(candidate):
            raise UpdateError('Please close the existing Little Figures launcher window first, then run Update Little Figures.bat again. Your current app has not been changed.')
    return port


def launch_app(root: Path, port: int) -> None:
    python = root / ('.venv/Scripts/python.exe' if os.name == 'nt' else '.venv/bin/python')
    if not python.is_file():
        python = Path(sys.executable)
    options = {'cwd': root}
    if os.name == 'nt':
        options['creationflags'] = subprocess.CREATE_NEW_CONSOLE
    subprocess.Popen([str(python), str(root / 'scripts/launch_local.py'), '--port', str(port)], **options)


def run_update(root: Path, *, no_launch=False, check=False, fetch=download) -> dict:
    root = root.resolve()
    if not (root / 'scripts/launch_local.py').is_file() or not (root / 'Start Little Figures.bat').is_file():
        raise UpdateError('Put Update Little Figures.bat in the same extracted folder as Start Little Figures.bat, then open it again. Do not run it from inside a ZIP.')
    with updater_lock(root):
        workspace = local_directory(root, 'updates')
        # A complete prepared backup also survives window closure or power loss.
        # Recover it before installing anything else, rather than leave mixed files.
        for record_path in sorted((workspace / 'backups').glob('*/backup.json'), reverse=True):
            try:
                pending = json.loads(record_path.read_text(encoding='utf-8'))
            except (OSError, ValueError) as error:
                raise UpdateError('An update recovery record is unreadable. Your existing app was not changed.') from error
            if pending.get('status') == 'prepared':
                if check:
                    raise UpdateError('An interrupted update needs recovery. Run the updater normally to restore its backup first.')
                print('Recovering app files from an interrupted update...', flush=True)
                stop_running_app(root)
                restore_backup(root, record_path.parent)
        with tempfile.TemporaryDirectory(prefix='stage-', dir=workspace) as temporary:
            temporary = Path(temporary)
            print('Checking the official Little Figures release...', flush=True)
            fetch(MANIFEST_URL, temporary / 'latest.json', 4 * 1024 * 1024)
            try:
                manifest = validate_manifest(json.loads((temporary / 'latest.json').read_text(encoding='utf-8')))
            except (ValueError, UnicodeError) as error:
                raise UpdateError('The release manifest could not be read. Your app was not changed.') from error
            print(f'Checking version {manifest["version"]}...', flush=True)
            operations = plan_release(root, manifest)
            if check:
                print(f'{len(operations)} app files need updating. Installed models and saved projects will stay in place.', flush=True)
                return {'version': manifest['version'], 'changes': len(operations)}
            if not operations:
                print(f'Your app is already up to date ({manifest["version"]}).', flush=True)
                return {'version': manifest['version'], 'changes': 0}
            print('Downloading and verifying the update before changing any app files...', flush=True)
            archive = temporary / 'release.zip'
            fetch(manifest['archive_url'], archive, MAX_ARCHIVE)
            stage = temporary / 'files'
            stage_release(archive, manifest, stage)
            port = stop_running_app(root)
            # Recheck paths after the app exits, before backing up or writing.
            operations = plan_release(root, manifest)
            backup = apply_release(root, stage, manifest, operations)
            print(f'Updated to {manifest["version"]}. Your models, projects, and browser saves were kept.', flush=True)
            print('A backup of changed app files is kept in .local/updates/backups.', flush=True)
            if not no_launch:
                print('Restarting Little Figures Studio...', flush=True)
                launch_app(root, port)
            return {'version': manifest['version'], 'changes': len(operations), 'backup': str(backup)}


def rollback_latest(root: Path, *, no_launch=False) -> None:
    with updater_lock(root):
        candidates = sorted((root / '.local/updates/backups').glob('*/backup.json'), reverse=True)
        for candidate in candidates:
            if json.loads(candidate.read_text(encoding='utf-8')).get('status') in {'prepared', 'applied'}:
                port = stop_running_app(root)
                restore_backup(root, candidate.parent)
                print('The previous app files have been restored. Your models and projects were kept.', flush=True)
                if not no_launch:
                    launch_app(root, port)
                return
        raise UpdateError('No completed update backup is available to roll back.')


def build_batch(source: Path, output: Path) -> None:
    # Windows CMD exits before the payload. The same Python code is embedded,
    # so the initial standalone shortcut never downloads an executable helper.
    loader = """-c "import os,pathlib; p=pathlib.Path(os.environ['LITTLE_FIGURES_UPDATE_FILE']); exec(compile(p.read_text(encoding='utf-8').split(chr(35)+' LITTLE_FIGURES_UPDATER_PYTHON_PAYLOAD',1)[1],str(p),'exec'))" --root "%LITTLE_FIGURES_UPDATE_ROOT%." %*"""
    header = r'''@echo off
setlocal
cd /d "%~dp0"
title Update Little Figures Studio
set "LITTLE_FIGURES_UPDATE_FILE=%~f0"
set "LITTLE_FIGURES_UPDATE_ROOT=%~dp0"
py -3.12 --version >nul 2>&1
'''
    # CMD parses the entire parenthesized branch before Python starts. Its
    # success/error continuation therefore survives replacement of this .bat.
    continuation = r'''
  if errorlevel 1 (
    echo.
    echo Update did not finish. Read the message above, then try again.
    pause
    exit /b 1
  )
  echo.
  echo You can close this updater window.
  pause
  exit /b 0
)
'''
    for condition, python in [
        ('if not errorlevel 1', 'py -3.12'),
        (r'if exist "%LocalAppData%\Programs\Python\Python312\python.exe"', r'"%LocalAppData%\Programs\Python\Python312\python.exe"'),
        (r'if exist "%ProgramFiles%\Python312\python.exe"', r'"%ProgramFiles%\Python312\python.exe"'),
    ]:
        header += condition + ' (\n  ' + python + ' ' + loader + continuation
    header += ('echo Python 3.12 is needed. Run Start Little Figures.bat to finish setup first.\n'
               'pause\nexit /b 1\n')
    output.write_bytes((header + '\n' + BATCH_MARKER + '\n' + source.read_text(encoding='utf-8')).replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8'))


def main() -> None:
    parser = argparse.ArgumentParser(description='Update Little Figures without replacing models or saved projects.')
    parser.add_argument('--root', type=Path, default=None)
    parser.add_argument('--no-launch', action='store_true')
    parser.add_argument('--check', action='store_true', help='Check the release without downloading or installing app files.')
    parser.add_argument('--rollback', action='store_true', help='Restore files from the most recent successful update.')
    parser.add_argument('--build-batch', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root.resolve() if args.root else Path(__file__).resolve().parents[1]
    if args.build_batch:
        build_batch(Path(__file__).resolve(), root / 'Update Little Figures.bat')
        return
    try:
        if args.rollback:
            rollback_latest(root, no_launch=args.no_launch)
        else:
            run_update(root, no_launch=args.no_launch, check=args.check)
    except (UpdateError, OSError) as error:
        print(f'\nUpdate could not finish: {error}\n', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
