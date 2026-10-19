from collections import OrderedDict
from io import BytesIO
import time
import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from backend.app import main
from backend.app.schemas import Figure,Project

client=TestClient(main.app)


def test_legacy_project_migrates_without_losing_fields():
    legacy={'version':1,'figures':[{'id':'old','height_mm':100,'pose':'seated','transform':{'x':17}}]}
    r=client.post('/api/project/check',json=legacy)
    assert r.status_code==200
    p=r.json()
    assert p['figures'][0]['height_mm']==100 and p['figures'][0]['transform']['x']==17
    assert p['figures'][0]['appearance']['outfit']=='none'
    assert p['shared_base']['shape']=='none' and p['occasion']=='everyday'


def test_six_person_contract_and_seven_person_rejection():
    p=Project(figures=[Figure(id=f'p{i}',transform={'x':i*70}) for i in range(6)])
    assert client.post('/api/project/check',json=p.model_dump()).status_code==200
    invalid=p.model_dump();invalid['figures'].append({'id':'p7'})
    assert client.post('/api/project/check',json=invalid).status_code==422


@pytest.mark.parametrize('change',[
    {'appearance':{'outfit':'arbitrary-python'}},
    {'appearance':{'accessories':['hat','hat']}},
    {'appearance':{'accessories':['../../etc/passwd']}},
    {'appearance_confidence':1.01},
])
def test_unsafe_or_unsupported_style_inputs_rejected(change):
    p={'figures':[{'id':'safe',**change}]}
    assert client.post('/api/project/check',json=p).status_code==422


def test_inscription_charset_and_length_are_bounded():
    for text in ['<script>alert(1)</script>','x'*41,'💐']:
        p={'figures':[{'id':'a'}],'shared_base':{'shape':'round','inscription':text}}
        assert client.post('/api/project/check',json=p).status_code==422
    p={'figures':[{'id':'a'}],'shared_base':{'shape':'round','inscription':"A & B - 10/18/2026"}}
    assert client.post('/api/project/check',json=p).status_code==200


def test_shared_pedestal_render_export_and_individual_export_are_consistent():
    project=Project(figures=[
        Figure(id='one',height_mm=80,transform={'x':-35,'rotation':20,'scale':1.2},appearance={'outfit':'suit','accessories':['bowtie']}),
        Figure(id='two',height_mm=90,transform={'x':35},appearance={'outfit':'wedding-dress'}),
        Figure(id='excluded',included=False),
    ],shared_base={'shape':'rectangle','inscription':'A & B'})
    response=client.post('/api/jobs',json=project.model_dump())
    assert response.status_code==202
    job=client.get('/api/jobs/'+response.json()['id']).json()
    assert job['state']=='ready',job
    records=job['result']['figures']
    assert {r['id'] for r in records}=={'one','two','__group_base__'}
    meshes=[]
    for record in records:
        m=trimesh.Trimesh(vertices=record['vertices'],faces=record['faces'],process=False)
        if record['id']!='__group_base__':
            f=next(f for f in project.figures if f.id==record['id'])
            m.apply_scale(f.transform.scale)
            m.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(f.transform.rotation),[0,0,1]))
            m.apply_translation([f.transform.x,f.transform.y,0])
            assert len(record['face_groups'])==len(m.faces)
        assert m.is_watertight
        meshes.append(m)
    render_bounds=np.stack([m.bounds for m in meshes])
    render_extent=render_bounds[:,1].max(axis=0)-render_bounds[:,0].min(axis=0)
    exported=client.post('/api/export/stl',json={'project':project.model_dump()})
    assert exported.status_code==200,exported.text
    restored=trimesh.load(BytesIO(exported.content),file_type='stl')
    assert restored.is_watertight and restored.is_winding_consistent
    assert len(restored.split())==1
    np.testing.assert_allclose(restored.extents,render_extent,atol=.005)
    single=client.post('/api/export/stl',json={'project':project.model_dump(),'figure_id':'one'})
    assert single.status_code==200
    individual=trimesh.load(BytesIO(single.content),file_type='stl')
    assert individual.extents[2]==pytest.approx(96,abs=.005)


def test_queue_full_does_not_accept_unbounded_work(monkeypatch):
    monkeypatch.setattr(main,'jobs',OrderedDict((str(i),{'state':'queued','created':time.time()}) for i in range(4)))
    response=client.post('/api/jobs',json={'figures':[{'id':'new'}]})
    assert response.status_code==429
    assert len(main.jobs)==4


def test_health_tells_model_availability_honestly():
    health=client.get('/api/health').json()
    assert health['version']=='3.1.0' and health['max_people']==6
    assert isinstance(health['appearance']['available'],bool)
    assert health['appearance']['model'] and health['appearance']['detail']
    assert isinstance(health['detection']['available'],bool)


def test_reserved_scene_identity_cannot_be_used_as_a_figure():
    r=client.post('/api/project/check',json={'figures':[{'id':'__group_base__'}]})
    assert r.status_code==422
    assert 'reserved' in r.text

def test_photo_aspect_survives_project_round_trip():
    p=Project(figures=[Figure(id='wide',photo_aspect=2.25)])
    r=client.post('/api/project/check',json=p.model_dump())
    assert r.status_code==200 and r.json()['figures'][0]['photo_aspect']==2.25
