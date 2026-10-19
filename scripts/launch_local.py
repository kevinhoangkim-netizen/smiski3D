"""Prepare the private local app, start it, and open the user's browser."""
import argparse
import hashlib
import json
import os
import secrets
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT=Path(__file__).resolve().parents[1]
VENV=ROOT/'.venv'
LOCAL=ROOT/'.local'
PYTHON=VENV/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
HEALTH=urllib.request.build_opener(urllib.request.ProxyHandler({}))
APP_VERSION=json.loads((ROOT/'app-version.json').read_text(encoding='utf-8'))['version']

def run(command):
    subprocess.run([str(a) for a in command],cwd=ROOT,check=True)

def is_our_app(port):
    try:
        with HEALTH.open(f'http://127.0.0.1:{port}/api/health',timeout=2) as response:
            body=json.load(response)
            return body.get('app')=='little-figures-studio' and body.get('version')==APP_VERSION
    except Exception:
        return False

def prepare(skip_install=False):
    if sys.version_info[:2]!=(3,12):
        raise RuntimeError('Please install Python 3.12, then open Start Little Figures.bat again.')
    LOCAL.mkdir(exist_ok=True)
    for folder in ['matplotlib','cache']:
        (LOCAL/folder).mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(LOCAL/'matplotlib')
    os.environ['XDG_CACHE_HOME']=str(LOCAL/'cache')
    os.environ['PYTHONUNBUFFERED']='1'
    if not PYTHON.is_file():
        print('Setting up a private Python environment for this app...',flush=True)
        run([sys.executable,'-m','venv',VENV])
    lock=ROOT/'backend/requirements.lock.txt'
    stamp=LOCAL/'local-install.sha256'
    appearance_lock=ROOT/'backend/requirements-appearance.txt'
    expected=hashlib.sha256(lock.read_bytes()+(appearance_lock.read_bytes() if appearance_lock.exists() else b'')).hexdigest()
    if not skip_install and (not stamp.is_file() or stamp.read_text()!=expected):
        print('First-time setup: downloading app dependencies. This can take a few minutes.',flush=True)
        run([PYTHON,'-m','pip','install','--disable-pip-version-check','-r',lock])
        if appearance_lock.exists():
            print('Installing the local automatic outfit recognizer...',flush=True)
            run([PYTHON,'-m','pip','install','--disable-pip-version-check','-r',appearance_lock])
        stamp.write_text(expected)
    if not (ROOT/'frontend/dist/index.html').is_file():
        npm=shutil.which('npm')
        if not npm:
            raise RuntimeError('The built website is missing. Download the prepared Windows ZIP, which includes it. No Node.js is needed for that ZIP.')
        run([npm,'ci','--prefix','frontend','--cache',LOCAL/'npm-cache'])
        run([npm,'run','build','--prefix','frontend'])
    if not (LOCAL/'pose_landmarker_lite.task').is_file():
        print('Downloading the official local photo-pose model...',flush=True)
        try:
            run([PYTHON,'scripts/download_model.py'])
        except subprocess.CalledProcessError:
            print('Pose model download failed. Photo generation needs this model; reopen the launcher when online to retry.',flush=True)

    if not (LOCAL/'appearance/manifest.json').is_file():
        print('Downloading the local outfit/accessory model. This can take a few minutes...',flush=True)
        try:
            run([PYTHON,'scripts/download_appearance_model.py'])
        except subprocess.CalledProcessError:
            print('Outfit model download failed. Photo appearance analysis is incomplete; reopen the launcher when online to retry.',flush=True)


    for script, description in [('download_people_model.py','group person detector'),('download_garment_model.py','clothing coverage model')]:
        print(f'Preparing the local {description}...',flush=True)
        try:
            run([PYTHON,'scripts/'+script])
        except subprocess.CalledProcessError:
            print(f'The {description} could not be installed. Photo generation will report the missing model; reopen the launcher when online to retry.',flush=True)


def write_runtime(process, port, token):
    record=dict(app='little-figures-studio',version=APP_VERSION,port=port,pid=process.pid,
                root=str(ROOT.resolve()),update_token=token)
    temporary=LOCAL/'runtime.json.tmp'
    temporary.write_text(json.dumps(record),encoding='utf-8')
    if os.name!='nt': temporary.chmod(0o600)
    temporary.replace(LOCAL/'runtime.json')


def clear_runtime(pid, token):
    path=LOCAL/'runtime.json'
    try:
        current=json.loads(path.read_text(encoding='utf-8'))
        if current.get('pid')==pid and current.get('update_token')==token:
            path.unlink()
    except (OSError,ValueError):
        pass


def choose_port(requested):
    for candidate in range(requested,min(requested+20,65536)):
        try:
            # Check for a live listener before SO_REUSEADDR, whose Windows
            # semantics can otherwise allow a probe alongside another server.
            with socket.create_connection(('127.0.0.1',candidate),timeout=.2):
                continue
        except OSError:
            pass
        with socket.socket() as probe:
            # Uvicorn also uses reuse-address. Match that behavior so TIME_WAIT
            # after an update does not change the browser origin and hide drafts.
            probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            try:
                probe.bind(('127.0.0.1',candidate))
                return candidate
            except OSError:
                continue
    raise RuntimeError('Could not find an available local port. Close another copy and try again.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--no-browser',action='store_true',help='For headless validation.')
    parser.add_argument('--skip-install',action='store_true',help='Use already installed dependencies for validation.')
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    if not 1024<=args.port<=65535: raise RuntimeError('Choose a local port between 1024 and 65535.')
    print('\nLittle Figures Studio\nYour app runs on this computer. Keep this window open while using it.\n',flush=True)
    if is_our_app(args.port):
        print('The studio is already running. Opening your browser.',flush=True)
        if not args.no_browser: webbrowser.open(f'http://127.0.0.1:{args.port}/welcome/')
        return
    prepare(args.skip_install)
    port=choose_port(args.port)
    print('Starting your studio...',flush=True)
    token=secrets.token_urlsafe(32)
    child_env=os.environ.copy()
    child_env['LITTLE_FIGURES_UPDATE_TOKEN']=token
    process=subprocess.Popen([str(PYTHON),'-m','uvicorn','backend.app.main:app','--host','127.0.0.1','--port',str(port)],cwd=ROOT,env=child_env)
    try:
        write_runtime(process,port,token)
        for attempt in range(180):
            if process.poll() is not None:
                raise RuntimeError('The app could not start. See the message above for the missing dependency.')
            if is_our_app(port): break
            time.sleep(.5)
        else: raise RuntimeError('The app took too long to start. Close this window and try again.')
        print('Ready! Your browser will open. To stop the app, close this window or press Ctrl+C.',flush=True)
        if not args.no_browser: webbrowser.open(f'http://127.0.0.1:{port}/welcome/')
        process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=8)
            except subprocess.TimeoutExpired: process.kill();process.wait()
        clear_runtime(process.pid,token)

if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStudio stopped. You can open the launcher again whenever you like.')
    except (RuntimeError,OSError,subprocess.CalledProcessError) as error:
        print(f'\nSetup could not finish: {error}\n',file=sys.stderr)
        sys.exit(1)
