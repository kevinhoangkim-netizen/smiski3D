from io import BytesIO
import pytest
import trimesh
from PIL import Image
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app import pose
from backend.app.schemas import Project, Figure

client=TestClient(app)

def project(): return Project(figures=[Figure(id='demo')]).model_dump()

def test_full_manual_workflow_and_export():
    assert client.get('/api/health').json()['units']=='mm'
    response=client.post('/api/jobs',json=project())
    assert response.status_code==202
    job=client.get('/api/jobs/'+response.json()['id']).json()
    assert job['state']=='ready'
    mesh=job['result']['figures'][0]
    assert mesh['report']['dimensions']['height']==90
    assert len(mesh['vertices'])>100
    checked=client.post('/api/validate',json=project())
    assert checked.status_code==200
    exported=client.post('/api/export/stl',json={'project':project()})
    assert exported.status_code==200
    restored=trimesh.load(BytesIO(exported.content),file_type='stl')
    assert restored.is_watertight
    assert restored.extents[2]==pytest.approx(90,abs=.01)

def test_unknown_job_and_excluded_export():
    assert client.get('/api/jobs/nonexistent').status_code==404
    assert client.post('/api/export/stl',json={'project':project(),'figure_id':'missing'}).status_code==404

@pytest.mark.parametrize('payload',[
 {'version':2,'figures':[{'id':'a'}]},
 {'figures':[{'id':'../bad'}]},
 {'figures':[{'id':'a','height_mm':999999}]},
 {'figures':[{'id':'a','pose':'photo'}]},
 {'figures':[{'id':'a','included':False}]},
 {'figures':[{'id':'a'},{'id':'a'}]},
 {'figures':[{'id':'a','transform':{'x':float('1e200')}}]},
 {'figures':[{'id':'a'}],'code':'print(1)'},
])
def test_invalid_projects_rejected(payload):
    assert client.post('/api/jobs',json=payload).status_code==422

def test_invalid_and_oversized_uploads():
    assert client.post('/api/analyze',files={'file':('evil.png',b'not an image','image/png')}).status_code==415
    assert client.post('/api/analyze',files={'file':('big.jpg',b'x'*(10*1024*1024+1),'image/jpeg')}).status_code==413
    b=BytesIO();Image.new('RGB',(10,10)).save(b,format='BMP')
    assert client.post('/api/analyze',files={'file':('a.bmp',b.getvalue(),'image/bmp')}).status_code==415

def image_bytes():
    b=BytesIO();Image.new('RGB',(32,32),'white').save(b,format='PNG');return b.getvalue()

def test_missing_model_is_explicit_not_fabricated(monkeypatch,tmp_path):
    monkeypatch.setattr(pose,'MODEL_PATH',tmp_path/'missing.task')
    r=client.post('/api/analyze',files={'file':('valid.png',image_bytes(),'image/png')})
    assert r.status_code==503
    assert 'manual' in r.json()['detail'].lower()

def test_no_detection_is_explicit(monkeypatch):
    monkeypatch.setattr(pose,'available',lambda:(True,'test'))
    monkeypatch.setattr(pose,'analyze',lambda image:[])
    r=client.post('/api/analyze',files={'file':('blank.png',image_bytes(),'image/png')})
    assert r.status_code==200 and r.json()['subjects']==[]
    assert 'No confident' in r.json()['warnings'][0]

def test_runtime_failure_is_explicit(monkeypatch):
    monkeypatch.setattr(pose,'available',lambda:(True,'test'))
    def broken(image): raise RuntimeError('failure')
    monkeypatch.setattr(pose,'analyze',broken)
    r=client.post('/api/analyze',files={'file':('blank.png',image_bytes(),'image/png')})
    assert r.status_code==503
    assert 'No image-derived pose' in r.json()['detail']

def test_low_confidence_landmarks_use_labelled_neutral_pose():
    f=Figure(id='photo',pose='photo',landmarks=[{'x':.5,'y':.5,'visibility':0}]*33)
    r=client.post('/api/jobs',json=Project(figures=[f]).model_dump())
    result=client.get('/api/jobs/'+r.json()['id']).json()
    assert result['state']=='ready'
    assert result['result']['figures'][0]['report']['status']!='fail'

def test_project_round_trip_contract():
    p=project()
    assert client.post('/api/project/check',json=p).json()==p

def test_raw_request_limit_applies_before_json_or_multipart_parsing():
    r=client.post('/api/jobs',content=b'x'*(11*1024*1024),headers={'content-type':'application/json'})
    assert r.status_code==413
    def chunks():
        yield b'--sample\r\nContent-Disposition: form-data; name="file"; filename="big.png"\r\nContent-Type: image/png\r\n\r\n'
        for _ in range(11): yield b'x'*(1024*1024)
        yield b'\r\n--sample--\r\n'
    r=client.post('/api/analyze',content=chunks(),headers={'content-type':'multipart/form-data; boundary=sample'})
    assert r.status_code==413
