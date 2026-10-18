from io import BytesIO
import numpy as np
import pytest
import trimesh
from backend.app.geometry import generate, validate, dimensions, project_meshes, union_or_compose, measure, PRESETS
from backend.app.schemas import Figure, Project

@pytest.mark.parametrize('pose',PRESETS)
def test_original_poses_are_real_closed_connected_solids(pose):
    mesh=generate(Figure(id='figure',pose=pose))
    assert mesh.volume>1000
    assert len(mesh.faces)>1000
    assert min(mesh.extents)>15
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert len(mesh.split())==1
    assert validate(mesh)['status']!='fail'

@pytest.mark.parametrize('base',['none','round','rectangle'])
def test_exact_height_and_binary_stl_round_trip(base):
    f=Figure(id='one',height_mm=100,base={'shape':base})
    m=generate(f)
    assert m.extents[2]==pytest.approx(100,abs=.01)
    binary=m.export(file_type='stl')
    assert len(binary)==84+50*len(m.faces)
    restored=trimesh.load(BytesIO(binary),file_type='stl',force='mesh')
    np.testing.assert_allclose(restored.extents,m.extents,atol=.002)
    assert restored.is_watertight and restored.is_winding_consistent

def test_group_inclusion_and_scale():
    project=Project(figures=[Figure(id='a',height_mm=100),Figure(id='b',height_mm=80,transform={'x':100}),Figure(id='c',included=False)])
    pairs=project_meshes(project)
    assert [f.id for f,_ in pairs]==['a','b']
    assert pairs[0][1].extents[2]==pytest.approx(100)
    assert pairs[1][1].extents[2]==pytest.approx(80)
    combined=union_or_compose([m for _,m in pairs])
    assert len(combined.split())==2
    assert combined.bounds[1,0]>100
    assert combined.is_watertight

def test_touching_group_is_fused_not_overlapping_shells():
    a=generate(Figure(id='a'))
    b=generate(Figure(id='b',transform={'x':20}))
    combined=union_or_compose([a,b])
    assert combined.volume<a.volume+b.volume
    assert combined.is_watertight
    assert len(combined.split())==1
    restored=trimesh.load(BytesIO(combined.export(file_type='stl')),file_type='stl')
    assert restored.is_watertight

def test_numeric_distance_and_bounds():
    assert measure([0,0,0],[3,4,12])==13
    assert dimensions(trimesh.creation.box([10,20,30]))=={'width':10,'depth':20,'height':30}

def test_defects_are_not_falsely_passed():
    m=trimesh.creation.box()
    m.update_faces(np.arange(len(m.faces))!=0)
    report=validate(m)
    assert report['status']=='fail'
    assert next(c for c in report['checks'] if c['name']=='Watertight')['status']=='fail'
    degenerate=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[2,0,0]],faces=[[0,1,2]],process=False)
    r=validate(degenerate)
    assert next(c for c in r['checks'] if c['name']=='Degenerate faces')['status']=='fail'

def test_no_base_warns_about_contact():
    r=validate(generate(Figure(id='a',base={'shape':'none'})))
    assert next(c for c in r['checks'] if c['name']=='Build-plate stability')['status']=='warning'

def test_uniform_scale_and_rotation():
    m=generate(Figure(id='a',height_mm=100,transform={'x':25,'rotation':45,'scale':1.5}))
    assert m.extents[2]==pytest.approx(150)
    reference=generate(Figure(id='a',height_mm=100,transform={'rotation':45,'scale':1.5}))
    np.testing.assert_allclose(m.vertices-reference.vertices,np.tile([25,0,0],(len(m.vertices),1)),atol=1e-6)
