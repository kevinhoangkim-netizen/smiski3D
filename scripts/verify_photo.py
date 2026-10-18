"""Run real local photo inference through the API, then reimport the exported STL."""
import argparse
from pathlib import Path
from io import BytesIO
from fastapi.testclient import TestClient
import trimesh
from backend.app.main import app
from backend.app.schemas import Figure, Project
from backend.app.pose import close

parser=argparse.ArgumentParser()
parser.add_argument('image',type=Path)
parser.add_argument('--expected-people',type=int,choices=[1,2,3,4,5,6])
args=parser.parse_args()
with TestClient(app) as client:
    with args.image.open('rb') as image:
        response=client.post('/api/analyze',files={'file':(args.image.name,image,'image/jpeg')})
    assert response.status_code==200,response.text
    subjects=response.json()['subjects']
    assert subjects,'No people detected: use a clear full-body reference.'
    if args.expected_people: assert len(subjects)==args.expected_people,len(subjects)
    figures=[Figure(
        id=s['id'], pose='photo', landmarks=s['landmarks'],
        photo_aspect=s.get('photo_aspect',1), appearance=s.get('appearance',{}),
        appearance_source='automatic' if s.get('appearance_detection',{}).get('status')!='unavailable' else 'unavailable',
        analysis_warnings=s.get('warnings',[]) + s.get('appearance_detection',{}).get('warnings',[]),
        transform={'x':i*100},
    ) for i,s in enumerate(subjects)]
    project=Project(figures=figures).model_dump()
    created=client.post('/api/jobs',json=project)
    assert created.status_code==202,created.text
    job=client.get('/api/jobs/'+created.json()['id']).json()
    assert job['state']=='ready',job
    exported=client.post('/api/export/stl',json={'project':project})
    assert exported.status_code==200,exported.text
    mesh=trimesh.load(BytesIO(exported.content),file_type='stl')
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert abs(mesh.extents[2]-90)<.01
    print(f'Real inference: {len(subjects)} subject(s); confidence '+str([round(s['confidence'],3) for s in subjects]))
    print(f'Pose-driven STL: {len(mesh.faces)} triangles, {len(mesh.split())} closed solid(s), {mesh.extents[2]:.3f} mm tall.')
close()
