from collections import OrderedDict
from io import BytesIO
import logging
import threading
import time
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from fastapi.concurrency import run_in_threadpool
from PIL import Image, ImageOps, UnidentifiedImageError
from .schemas import Project, ExportRequest
from .geometry import project_meshes, figure_report, union_or_compose, dimensions, PRESETS, generate
from . import pose

MAX_UPLOAD=10*1024*1024
Image.MAX_IMAGE_PIXELS=20_000_000
jobs=OrderedDict()
job_lock=threading.Lock()
work_slots=threading.BoundedSemaphore(2)
logger=logging.getLogger(__name__)
app=FastAPI(title='Little Figures Studio',version='1.0.0',description='Local original figurine generation. Geometry and STL coordinates use millimeters, Z up.')

class RequestSizeLimit:
    """Bound the raw request stream before multipart parsing/spooling."""
    def __init__(self, app): self.app=app
    async def __call__(self, scope, receive, send):
        if scope['type']!='http':
            return await self.app(scope,receive,send)
        limit=MAX_UPLOAD+65536
        length=dict(scope['headers']).get(b'content-length')
        if length is not None:
            try: too_big=int(length)>limit or int(length)<0
            except ValueError:
                return await JSONResponse({'detail':'Invalid Content-Length.'},status_code=400)(scope,receive,send)
            if too_big:
                return await JSONResponse({'detail':'Request exceeds the 10 MB upload limit.'},status_code=413)(scope,receive,send)
        total=0
        async def bounded_receive():
            nonlocal total
            message=await receive()
            total+=len(message.get('body',b''))
            if total>limit:
                raise HTTPException(status_code=413,detail='Request exceeds the 10 MB upload limit.')
            return message
        await self.app(scope,bounded_receive,send)

app.add_middleware(RequestSizeLimit)

def fail(code,message):
    raise HTTPException(status_code=code,detail=message)

@app.get('/api/health')
def health():
    ready,detail=pose.available()
    return dict(app='little-figures-studio',status='ok',pose_available=ready,pose_detail=detail,units='mm',up_axis='Z')

@app.get('/api/poses')
def presets():
    return dict(poses=PRESETS)

@app.post('/api/analyze')
async def analyze_photo(file: UploadFile=File(...)):
    data=await file.read(MAX_UPLOAD+1)
    await file.close()
    if len(data)>MAX_UPLOAD:
        fail(413,'Photo exceeds the 10 MB limit.')
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in ('JPEG','PNG','WEBP'):
                fail(415,'Choose a JPG, PNG, or WEBP photo.')
            if source.width*source.height>20_000_000:
                fail(413,'Photo exceeds 20 megapixels. Resize it first.')
            source.load()
            image=ImageOps.exif_transpose(source).convert('RGB')
            image.thumbnail((1600,1600))
    except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError,Image.DecompressionBombWarning):
        fail(415,'Could not decode this image safely. Choose a valid JPG, PNG, or WEBP.')
    ready,detail=pose.available()
    if not ready:
        fail(503,detail)
    try:
        subjects=await run_in_threadpool(pose.analyze,image)
    except Exception:
        logger.exception('Pose inference failed')
        fail(503,'Local pose estimation failed. No image-derived pose was generated. Choose a manual preset or inspect the server log.')
    if not subjects:
        return dict(mode='photo',subjects=[],warnings=['No confident people detected. Try a full-body photo or explicitly choose a manual preset.'])
    return dict(mode='photo',subjects=subjects,warnings=['Estimates up to three visible people. Occlusion, unusual poses and crowded photos reduce reliability.'])

def process_job(job_id,project):
    with work_slots:
        try:
            jobs[job_id].update(state='generating',stage='Generating solid geometry')
            pairs=project_meshes(project)
            jobs[job_id].update(state='validating',stage='Checking surfaces and dimensions')
            result=[]
            for f,mesh in pairs:
                local_mesh=generate(f,False)
                result.append(dict(id=f.id,name=f.name,vertices=local_mesh.vertices.tolist(),faces=local_mesh.faces.tolist(),
                    report=figure_report(f,mesh,project.minimum_feature_mm)))
            jobs[job_id].update(state='ready',stage='Ready',result=dict(figures=result))
        except Exception:
            logger.exception('Mesh generation failed')
            jobs[job_id].update(state='error',stage='Generation failed',error='Could not generate this pose. Try a manual preset or restore defaults.')

@app.post('/api/jobs',status_code=202)
def generate_job(project: Project,background_tasks: BackgroundTasks):
    with job_lock:
        # Bounded in-memory results, expiring after 30 minutes. No uploaded images retained.
        now=time.time()
        for key in list(jobs):
            if now-jobs[key]['created']>1800 and jobs[key]['state'] in ('ready','error'):
                del jobs[key]
        if sum(j['state'] not in ('ready','error') for j in jobs.values())>=4:
            fail(429,'Generation queue is full. Wait for a current job to finish.')
        while len(jobs)>=24:
            done=next((k for k,v in jobs.items() if v['state'] in ('ready','error')),None)
            if done is None: break
            del jobs[done]
        job_id=uuid.uuid4().hex
        jobs[job_id]=dict(id=job_id,state='queued',stage='Queued',created=now)
    background_tasks.add_task(process_job,job_id,project)
    return dict(id=job_id,state='queued')

@app.get('/api/jobs/{job_id}')
def get_job(job_id: str):
    with job_lock:
        job=jobs.get(job_id)
        if not job: fail(404,'Job not found or expired. Generate again.')
        return job.copy()

@app.post('/api/validate')
def validate_project(project: Project):
    with work_slots:
        pairs=project_meshes(project)
        return dict(figures=[dict(id=f.id,report=figure_report(f,m,project.minimum_feature_mm)) for f,m in pairs],
                    dimensions=dimensions(union_or_compose([m for _,m in pairs])))

@app.post('/api/export/stl')
def export_stl(request: ExportRequest):
    with work_slots:
        pairs=project_meshes(request.project)
        if request.figure_id:
            pairs=[(f,m) for f,m in pairs if f.id==request.figure_id]
            if not pairs: fail(404,'Included figure not found.')
        mesh=union_or_compose([m for _,m in pairs])
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            fail(422,'Export refused: mesh has open or inconsistently wound surfaces.')
        content=mesh.export(file_type='stl')
        return Response(content,media_type='model/stl',headers={'Content-Disposition':'attachment; filename="little-figures-mm.stl"','X-Model-Units':'mm'})

@app.post('/api/project/check',response_model=Project)
def check_project(project: Project):
    return project

# Serve the built studio and landing page from the same local Python process.
# Vite remains available independently for development.
from pathlib import Path
from fastapi.staticfiles import StaticFiles
_frontend_dist = Path(__file__).resolve().parents[2] / 'frontend' / 'dist'
if _frontend_dist.is_dir():
    app.mount('/', StaticFiles(directory=_frontend_dist, html=True), name='website')
