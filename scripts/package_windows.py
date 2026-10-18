"""Build the distributable with compiled UI, excluding secrets/caches/models."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'downloads/Little-Figures-Studio-Windows.zip'


def package():
    if not (ROOT / 'frontend/dist/index.html').is_file():
        raise RuntimeError('Build frontend before packaging.')
    files = []
    for folder in ('backend', 'frontend', 'scripts', 'tests', 'examples'):
        for path in (ROOT / folder).rglob('*'):
            relative = path.relative_to(ROOT)
            if not path.is_file() or any(part in {'node_modules','__pycache__','.pytest_cache','.vite'} for part in relative.parts):
                continue
            if path.suffix == '.tsbuildinfo':
                continue
            files.append(path)
    for name in ('Start Little Figures.bat','START HERE - Windows.txt','README.md','ARCHITECTURE.md','LIMITATIONS.md','.env.example','.gitignore','.node-version','.python-version','pytest.ini'):
        files.append(ROOT / name)
    TARGET.parent.mkdir(exist_ok=True)
    temporary = TARGET.with_suffix('.zip.tmp')
    with ZipFile(temporary, 'w', compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            archive.write(path, Path('Little Figures Studio') / path.relative_to(ROOT))
    with ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Package CRC verification failed.')
        names = archive.namelist()
        for name in ('Start Little Figures.bat','scripts/launch_local.py','frontend/dist/index.html',
                     'backend/requirements-appearance.txt','scripts/download_appearance_model.py',
                     'scripts/download_people_model.py','scripts/download_garment_model.py',
                     'scripts/model_licenses/SCHP-MIT.txt','backend/app/people.py',
                     'backend/app/garment_parser.py','frontend/src/PhotoReview.tsx'):
            if 'Little Figures Studio/' + name not in names:
                raise RuntimeError(f'Missing package file: {name}')
        if any('/.local/' in name or '/.venv/' in name or '/.git/' in name or '/node_modules/' in name for name in names):
            raise RuntimeError('Private setup/cache directory found in package.')
    temporary.replace(TARGET)
    print(f'{TARGET.name}: {TARGET.stat().st_size/1024/1024:.2f} MiB; SHA256 {hashlib.sha256(TARGET.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    package()
