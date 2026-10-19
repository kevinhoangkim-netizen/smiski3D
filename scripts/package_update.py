"""Build the verified runtime patch used by the standalone Windows updater."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json

from update_local import ROOT_FILES, safe_path, validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def package():
    version = json.loads((ROOT/'app-version.json').read_text())['version']
    if not (ROOT/'frontend/dist/index.html').is_file():
        raise RuntimeError('Build the frontend before packaging.')
    files = [ROOT/name for name in ROOT_FILES]
    files.extend((ROOT/'backend').glob('requirements*.txt'))
    for folder in ('backend/app', 'frontend/dist', 'scripts'):
        files.extend(path for path in (ROOT/folder).rglob('*') if path.is_file()
                     and not any(part in {'__pycache__', 'node_modules', '.pytest_cache'} for part in path.parts)
                     and path.suffix not in ('.pyc', '.tsbuildinfo'))
    files = sorted(set(files))
    entries = []
    for path in files:
        relative = safe_path(path.relative_to(ROOT).as_posix())
        content = path.read_bytes()
        entries.append(dict(path=relative,size=len(content),sha256=hashlib.sha256(content).hexdigest()))
    target = ROOT/f'downloads/releases/Little-Figures-Update-{version}.zip'
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary = target.with_suffix('.zip.tmp')
    with ZipFile(temporary,'w',compression=ZIP_DEFLATED,compresslevel=6) as archive:
        for path,entry in zip(files,entries):
            archive.write(path,entry['path'])
    with ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Update archive CRC verification failed.')
    temporary.replace(target)
    manifest = dict(schema=1,version=version,
                    archive_url=f'https://raw.githubusercontent.com/kevinhoangkim-netizen/smiski3D/main/downloads/releases/{target.name}',
                    archive_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    archive_size=target.stat().st_size,files=entries)
    validate_manifest(manifest)
    (ROOT/'downloads/latest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'{target.name}: {len(entries)} verified runtime files; {target.stat().st_size:,} bytes; SHA256 {manifest["archive_sha256"]}')


if __name__=='__main__':
    package()
