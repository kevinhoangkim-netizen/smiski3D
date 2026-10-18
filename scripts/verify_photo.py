"""Exercise complete photo analysis, whole-group generation and actual STL reimport.

Uses the returned relative heights/layout and refuses unresolved/overflow people.
Fixtures must remain local; this helper never uploads to an external service.
"""
import argparse
from pathlib import Path
from io import BytesIO
import numpy as np
from fastapi.testclient import TestClient
import trimesh
from backend.app.main import app
from backend.app.schemas import Figure, Project, PhotoReview
from backend.app.pose import close

parser=argparse.ArgumentParser()
parser.add_argument('image',type=Path)
parser.add_argument('--expected-people',type=int,choices=[1,2,3,4,5,6])
parser.add_argument('--shared-base',action='store_true')
args=parser.parse_args()
with TestClient(app) as client:
    with args.image.open('rb') as image:
        response=client.post('/api/analyze',files={'file':(args.image.name,image,'image/jpeg')})
    assert response.status_code==200,response.text
    analysis=response.json();subjects=analysis['subjects']
    assert subjects,'No usable people detected: choose a clearer full-body reference.'
    assert analysis['detection']['available'],analysis['detection']
    assert not analysis.get('overflow') and not analysis['unresolved_regions'],analysis['warnings']
    assert analysis['detected_people']==len(subjects),'Incomplete group: not exporting.'
    assert not any(s.get('appearance_detection',{}).get('blocking') for s in subjects),'A major object cannot be reconstructed.'
    if args.expected_people: assert len(subjects)==args.expected_people,len(subjects)
    figures=[Figure(
        id=s['id'],name=s['name'],pose='photo',landmarks=s['landmarks'],
        photo_aspect=s.get('photo_aspect',1),appearance=s.get('appearance',{}),
        height_mm=s.get('suggested_height_mm',90),base={'shape':'none'},
        appearance_source='automatic' if s.get('appearance_detection',{}).get('status')!='unavailable' else 'unavailable',
        analysis_warnings=list(dict.fromkeys(s.get('warnings',[])+s.get('appearance_detection',{}).get('warnings',[])))[:20],
        analysis_blockers=s.get('appearance_detection',{}).get('unsupported_items',[]) if s.get('appearance_detection',{}).get('blocking') else [],
        transform=s.get('suggested_transform',{}),
    ) for s in subjects]
    project=Project(figures=figures,photo_review=PhotoReview(detected_people=len(subjects),confirmed=True),shared_base={'shape':'round' if args.shared_base else 'none','inscription':'Together' if args.shared_base else ''}).model_dump()
    created=client.post('/api/jobs',json=project)
    assert created.status_code==202,created.text
    job=client.get('/api/jobs/'+created.json()['id']).json()
    assert job['state']=='ready',job
    assert {f['id'] for f in job['result']['figures'] if f['id']!='__group_base__'}=={f.id for f in figures}
    validation=client.post('/api/validate',json=project)
    assert validation.status_code==200,validation.text
    exported=client.post('/api/export/stl',json={'project':project})
    assert exported.status_code==200,exported.text
    mesh=trimesh.load(BytesIO(exported.content),file_type='stl')
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0
    dims=validation.json()['dimensions']
    np.testing.assert_allclose(mesh.extents,[dims[k] for k in ['width','depth','height']],atol=.01)
    print(f'Real inference: {len(subjects)} people, all present in the generated scene.')
    print('Photo-driven styles: '+str([s['appearance_detection'].get('summary','Unknown') for s in subjects]))
    print(f'STL: {len(mesh.faces)} triangles, {len(mesh.split())} closed solid(s); dimensions {dims} mm.')
close()
