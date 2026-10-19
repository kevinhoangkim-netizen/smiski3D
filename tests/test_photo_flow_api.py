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

def test_unsupported_major_object_still_blocks_the_whole_photo_group():
    p=photo_project();p['figures'][0]['analysis_blockers']=['Wheelchair reconstruction is unsupported.']
    for path,body in [('/api/jobs',p),('/api/validate',p),('/api/export/stl',{'project':p})]:
        r=client.post(path,json=body)
        assert r.status_code==422,(path,r.text)


@pytest.mark.parametrize('change',[
    {'appearance_source':'unavailable'},
    {'appearance':{'garment':{'inferred':True,'upper':'unknown','lower':'unknown'}}},
    {'appearance':{'garment':{'inferred':True,'upper':'shirt','lower':'unknown','sleeve_length':1}}},
])
def test_old_saved_photo_clothing_is_estimated_and_all_people_generate(change):
    p=photo_project();p['figures'][0].update(change)
    checked=client.post('/api/project/check',json=p)
    assert checked.status_code==200,checked.text
    canonical=checked.json()
    garment=canonical['figures'][0]['appearance']['garment']
    assert garment['upper']=='shirt' and garment['lower']=='pants'
    assert 'lower' in garment['estimated_fields'] and garment['estimation_notes']
    assert canonical['figures'][0]['landmarks']==p['figures'][0]['landmarks']
    assert canonical['figures'][1]['transform']['x']==90
    response=client.post('/api/jobs',json=p)
    assert response.status_code==202,response.text
    result=client.get('/api/jobs/'+response.json()['id']).json()
    assert result['state']=='ready',result
    assert {figure['id'] for figure in result['result']['figures']}=={'p0','p1'}
    assert client.post('/api/project/check',json=canonical).json()==canonical


def test_estimates_survive_saved_project_and_stl_uses_the_same_clothing():
    import numpy as np
    import trimesh
    p=photo_project(1)
    p['figures'][0]['appearance']={'garment':{'inferred':True,'upper':'shirt','lower':'unknown'},
        'garment_evidence':{'lower_fractions':{'skirt':.65,'pants':.08},'lower_region':'skirt','hem_length':.7}}
    canonical=client.post('/api/project/check',json=p).json()
    garment=canonical['figures'][0]['appearance']['garment']
    assert garment['lower']=='skirt' and garment['hem_length']==.7
    assert garment['estimated_fields'] and garment['inferred']
    original=client.post('/api/export/stl',json={'project':p})
    saved=client.post('/api/export/stl',json={'project':canonical})
    assert original.status_code==saved.status_code==200,(original.text[:100],saved.text[:100])
    a=trimesh.load(BytesIO(original.content),file_type='stl');b=trimesh.load(BytesIO(saved.content),file_type='stl')
    assert a.is_watertight and b.is_watertight
    np.testing.assert_allclose(a.bounds,b.bounds,atol=.0001)
    np.testing.assert_allclose(a.volume,b.volume,atol=.0001)
    assert client.post('/api/validate',json=p).status_code==200
