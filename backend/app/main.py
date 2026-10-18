from collections import OrderedDict
from io import BytesIO
from pathlib import Path
import logging
import math
import threading
import time
import uuid

import numpy as np
import trimesh
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from .schemas import Project, ExportRequest
from . import geometry, pose

MAX_UPLOAD = 10 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000
jobs = OrderedDict()
job_lock = threading.Lock()
work_slots = threading.BoundedSemaphore(2)
analysis_slot = threading.Lock()
logger = logging.getLogger(__name__)
app = FastAPI(title='Little Figures Studio', version='2.0.0',
              description='Local original group figurines. Geometry and STL use millimeters, Z up. Photos are processed temporarily on this app server.')


class RequestSizeLimit:
    """Bound the raw request stream before multipart parsing/spooling."""
    def __init__(self, app): self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        limit = MAX_UPLOAD + 65536
        length = dict(scope['headers']).get(b'content-length')
        if length is not None:
            try:
                too_big = int(length) > limit or int(length) < 0
            except ValueError:
                return await JSONResponse({'detail': 'Invalid Content-Length.'}, status_code=400)(scope, receive, send)
            if too_big:
                return await JSONResponse({'detail': 'Request exceeds the 10 MB upload limit.'}, status_code=413)(scope, receive, send)
        total = 0
        async def bounded_receive():
            nonlocal total
            message = await receive()
            total += len(message.get('body', b''))
            if total > limit:
                raise HTTPException(status_code=413, detail='Request exceeds the 10 MB upload limit.')
            return message
        await self.app(scope, bounded_receive, send)


app.add_middleware(RequestSizeLimit)


def fail(code, message):
    raise HTTPException(status_code=code, detail=message)


def appearance_status():
    try:
        from . import appearance
        return appearance.status()
    except (ImportError, OSError):
        return dict(available=False, model='Local vision classifier',
                    detail='Automatic clothing recognition is not installed. Pose estimation remains available; appearance can be corrected manually.')


@app.get('/api/health')
def health():
    ready, detail = pose.available()
    return dict(app='little-figures-studio', version='2.0.0', status='ok',
                pose_available=ready, pose_detail=detail, appearance=appearance_status(),
                max_people=6, units='mm', up_axis='Z')


@app.get('/api/poses')
def presets():
    return dict(poses=geometry.PRESETS)


def analyze_locked(image):
    # Limit inference concurrency without retaining images in a job/result store.
    with analysis_slot:
        return pose.analyze(image)


@app.post('/api/analyze')
async def analyze_photo(file: UploadFile = File(...)):
    try:
        data = await file.read(MAX_UPLOAD + 1)
    finally:
        await file.close()
    if len(data) > MAX_UPLOAD:
        fail(413, 'Photo exceeds the 10 MB limit.')
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in ('JPEG', 'PNG', 'WEBP'):
                fail(415, 'Choose a JPG, PNG, or WEBP photo.')
            if source.width * source.height > 20_000_000:
                fail(413, 'Photo exceeds 20 megapixels. Resize it first.')
            source.load()
            image = ImageOps.exif_transpose(source).convert('RGB')
            image.thumbnail((1600, 1600))
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        fail(415, 'Could not decode this image safely. Choose a valid JPG, PNG, or WEBP.')
    ready, detail = pose.available()
    if not ready:
        fail(503, detail)
    try:
        subjects = await run_in_threadpool(analyze_locked, image)
    except Exception:
        logger.exception('Pose inference failed')
        fail(503, 'Local pose estimation failed. No image-derived pose was generated. Choose a manual preset or inspect the server log.')
    if not subjects:
        return dict(mode='photo', subjects=[], appearance_detection=appearance_status(),
                    warnings=['No confident people detected. Try a clear full-body photo or explicitly choose a manual preset.'])
    return dict(mode='photo', subjects=subjects, appearance_detection=appearance_status(),
                warnings=['Estimates up to six visible people. Group spacing, relative height, depth, clothing and hidden limbs are approximate; review uncertain suggestions.'])


def scene(project):
    return geometry.project_scene(project)


def scene_to_render(project, entries):
    by_id = {figure.id: figure for figure in project.figures}
    result = []
    for figure_id, name, mesh, report in entries:
        local_mesh = mesh.copy()
        if figure_id in by_id:
            transform = by_id[figure_id].transform
            matrix = trimesh.transformations.rotation_matrix(math.radians(transform.rotation), [0, 0, 1])
            matrix[:3, :3] *= transform.scale
            matrix[:3, 3] = [transform.x, transform.y, 0]
            local_mesh.apply_transform(np.linalg.inv(matrix))
        result.append(dict(id=figure_id, name=name, vertices=local_mesh.vertices.tolist(),
                           faces=local_mesh.faces.tolist(), report=report,
                           face_groups=mesh.metadata.get('face_groups')))
    return result


def process_job(job_id, project):
    with work_slots:
        try:
            jobs[job_id].update(state='generating', stage='Sculpting figures, clothing and accessories')
            entries = scene(project)
            jobs[job_id].update(state='validating', stage='Checking solid geometry and dimensions')
            result = scene_to_render(project, entries)
            jobs[job_id].update(state='ready', stage='Ready', result=dict(figures=result))
        except Exception:
            logger.exception('Mesh generation failed')
            jobs[job_id].update(state='error', stage='Generation failed',
                                error='Could not generate this scene safely. Review the pose/accessories, try a simpler layout, or restore defaults.')


@app.post('/api/jobs', status_code=202)
def generate_job(project: Project, background_tasks: BackgroundTasks):
    with job_lock:
        now = time.time()
        for key in list(jobs):
            if now - jobs[key]['created'] > 1800 and jobs[key]['state'] in ('ready', 'error'):
                del jobs[key]
        if sum(job['state'] not in ('ready', 'error') for job in jobs.values()) >= 4:
            fail(429, 'Generation queue is full. Wait for a current job to finish.')
        while len(jobs) >= 12:
            done = next((key for key, job in jobs.items() if job['state'] in ('ready', 'error')), None)
            if done is None:
                break
            del jobs[done]
        job_id = uuid.uuid4().hex
        jobs[job_id] = dict(id=job_id, state='queued', stage='Queued', created=now)
    background_tasks.add_task(process_job, job_id, project)
    return dict(id=job_id, state='queued')


@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    with job_lock:
        job = jobs.get(job_id)
        if not job:
            fail(404, 'Job not found or expired. Generate again.')
        return job.copy()


@app.post('/api/validate')
def validate_project(project: Project):
    with work_slots:
        try:
            entries = scene(project)
            combined = geometry.union_or_compose([entry[2] for entry in entries])
            return dict(figures=[dict(id=entry[0], report=entry[3]) for entry in entries],
                        dimensions=geometry.dimensions(combined),
                        combined_report=geometry.validate(combined, project.minimum_feature_mm))
        except (ValueError, RuntimeError):
            fail(422, 'This scene could not be validated safely. Review its layout and accessories.')


@app.post('/api/export/stl')
def export_stl(request: ExportRequest):
    with work_slots:
        try:
            if request.figure_id:
                figures = [f for f in request.project.figures if f.included and f.id == request.figure_id]
                if not figures:
                    fail(404, 'Included figure not found.')
                # Separate prints retain this person's configured individual pedestal.
                mesh = geometry.union_or_compose([geometry.generate(figures[0])])
            else:
                mesh = geometry.union_or_compose([entry[2] for entry in scene(request.project)])
            if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
                fail(422, 'Export refused: mesh has open, inward or inconsistently wound surfaces.')
            content = mesh.export(file_type='stl')
            # Verify the actual binary STL precision/welding rather than only the source mesh.
            restored = trimesh.load(BytesIO(content), file_type='stl', force='mesh')
            if not restored.is_watertight or not restored.is_winding_consistent or restored.volume <= 0:
                fail(422, 'Export refused: geometry did not survive STL precision safely. Try a simpler scene.')
        except (ValueError, RuntimeError):
            fail(422, 'This scene could not be exported safely. Review its layout and accessories.')
        return Response(content, media_type='model/stl', headers={
            'Content-Disposition': 'attachment; filename="little-figures-mm.stl"', 'X-Model-Units': 'mm'})


@app.post('/api/project/check', response_model=Project)
def check_project(project: Project):
    return project


# Both app and landing page are served by the same loopback-only local process.
_frontend_dist = Path(__file__).resolve().parents[2] / 'frontend' / 'dist'
if _frontend_dist.is_dir():
    app.mount('/', StaticFiles(directory=_frontend_dist, html=True), name='website')
