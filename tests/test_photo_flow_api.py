"""Whole-photo review contracts; detection accuracy uses separate real-photo checks."""
from io import BytesIO
import json
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app import pose

client=TestClient(app)

def photo_project(count=2):
    landmarks=[dict(x=.5,y=.5,z=0,visibility=1) for _ in range(33)]
    return dict(version=1,figures=[dict(id=f'p{i}',pose='photo',landmarks=landmarks,transform={'x':i*90}) for i in range(count)],photo_review=dict(detected_people=count,unresolved_people=0,confirmed=True))

@pytest.mark.parametrize('update',[
    {'confirmed':False}, {'unresolved_people':1}, {'detected_people':3}, {'detected_people':7},
])
def test_incomplete_review_blocks_generation_validation_and_export(update):
    p=photo_project();p['photo_review'].update(update)
    for path,body in [('/api/jobs',p),('/api/validate',p),('/api/export/stl',{'project':p})]:
        response=client.post(path,json=body)
        assert response.status_code==422,(path,response.text)


def test_confirmed_scene_cannot_drop_someone_or_replace_their_photo_pose():
    for field,value in [('included',False),('pose','standing')]:
        p=photo_project();p['figures'][1][field]=value
        assert client.post('/api/jobs',json=p).status_code==422


def test_empty_photo_first_project_can_be_saved_but_not_generated():
    p={'version':1,'figures':[]}
    assert client.post('/api/project/check',json=p).status_code==200
    assert client.post('/api/jobs',json=p).status_code==422
    assert client.post('/api/export/stl',json={'project':p}).status_code==422


def test_every_confirmed_person_is_generated_in_one_job():
    p=photo_project()
    r=client.post('/api/jobs',json=p)
    assert r.status_code==202,r.text
    job=client.get('/api/jobs/'+r.json()['id']).json()
    assert job['state']=='ready',job
    assert {f['id'] for f in job['result']['figures']}=={'p0','p1'}


def upload(regions):
    stream=BytesIO();Image.new('RGB',(50,50),'white').save(stream,format='PNG')
    return client.post('/api/analyze',files={'file':('reference.png',stream.getvalue(),'image/png')},data={'regions':regions})

@pytest.mark.parametrize('regions',[
    'not JSON','{}','[[0,0,1]]','[[0,0,2,1]]','[[0.8,0,0.2,1]]','[[true,0,1,1]]',json.dumps([[0,0,1,1]]*7),
])
def test_invalid_recovery_regions_are_rejected(regions):
    assert upload(regions).status_code==422


def test_retry_keeps_entire_photo_and_reports_unresolved_people(monkeypatch):
    monkeypatch.setattr(pose,'available',lambda:(True,'test'))
    seen=[]
    def analyze(image,regions=None):
        seen.append((image.size,regions))
        return dict(subjects=[],detected_people=1,unresolved_regions=regions,warnings=['The indicated person remains unresolved.'],overflow=False,detection={'available':True})
    monkeypatch.setattr(pose,'analyze_group',analyze)
    result=upload('[[0.1,0.2,0.8,0.9]]')
    assert result.status_code==200,result.text
    assert seen==[((50,50),[[.1,.2,.8,.9]])]
    assert result.json()['detected_people']==1 and result.json()['unresolved_regions']==[[.1,.2,.8,.9]]

@pytest.mark.parametrize('change',[
    {'analysis_blockers':['Wheelchair reconstruction is unsupported.']},
    {'appearance_source':'unavailable'},
    {'appearance':{'garment':{'inferred':True,'upper':'unknown','lower':'unknown'}}},
    {'appearance':{'garment':{'inferred':True,'upper':'shirt','lower':'unknown'}}},
])
def test_major_unknown_appearance_blocks_the_whole_photo_group(change):
    p=photo_project();p['figures'][0].update(change)
    for path,body in [('/api/jobs',p),('/api/export/stl',{'project':p})]:
        r=client.post(path,json=body)
        assert r.status_code==422,(path,r.text)
