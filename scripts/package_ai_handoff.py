"""Package tracked editable project files and an AI handoff, without local data."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'downloads/Little-Figures-Studio-AI-Handoff.zip'
PREFIX = 'Little Figures Studio - AI Handoff'
EXCLUDED = {'.git','.local','.venv','node_modules','__pycache__','.pytest_cache','.env'}


def package():
    tracked = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode('utf-8').split('\0')
    names = {name for name in tracked if name}
    names.update({'AI_HANDOFF.md','scripts/package_ai_handoff.py'})
    names = sorted(name for name in names
                   if not (set(Path(name).parts)&EXCLUDED)
                   and Path(name).parts[0]!='downloads'
                   and not name.startswith('frontend/dist/'))
    entries = []
    for name in names:
        path = ROOT/name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f'Expected a regular source file: {name}')
        content = path.read_bytes()
        entries.append(dict(path=name,size=len(content),sha256=hashlib.sha256(content).hexdigest()))
    manifest = dict(format=1,app_version=json.loads((ROOT/'app-version.json').read_text())['version'],
                    source_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    note='Working-tree files at packaging time; this manifest excludes itself.',files=entries)
    TARGET.parent.mkdir(exist_ok=True)
    temporary = TARGET.with_suffix('.zip.tmp')
    with ZipFile(temporary,'w',compression=ZIP_DEFLATED,compresslevel=6) as archive:
        for entry in entries:
            archive.write(ROOT/entry['path'],f"{PREFIX}/{entry['path']}")
        archive.writestr(f'{PREFIX}/SOURCE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Handoff archive failed CRC verification.')
        for entry in entries:
            content = archive.read(f"{PREFIX}/{entry['path']}")
            if len(content)!=entry['size'] or hashlib.sha256(content).hexdigest()!=entry['sha256']:
                raise RuntimeError(f"Handoff archive verification failed: {entry['path']}")
    temporary.replace(TARGET)
    print(f'{TARGET.name}: {len(entries)} project files plus manifest; {TARGET.stat().st_size:,} bytes')
    print('SHA256:',hashlib.sha256(TARGET.read_bytes()).hexdigest())


if __name__=='__main__':
    package()
