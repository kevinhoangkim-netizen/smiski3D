"""Visible pose relationships and actual print surfaces, independent of recognition."""
from io import BytesIO

import numpy as np
import pytest
import trimesh

from backend.app.figure_style import accessory_meshes
from backend.app.geometry import body_frame, generate, joints, validate
from backend.app.schemas import Figure


def clasped_pose():
    points=[dict(x=.5,y=.15,z=0.,visibility=1.) for _ in range(33)]
    for index,(x,y) in {11:(.40,.30),12:(.60,.30),13:(.37,.45),14:(.64,.45),
                       15:(.5,.48),16:(.5,.48),23:(.43,.56),24:(.57,.56),
                       25:(.40,.72),26:(.61,.71),27:(.46,.89),28:(.64,.88),
                       2:(.48,.13),5:(.52,.13)}.items():
        points[index].update(x=x,y=y)
    return points


def test_visible_pose_reprojects_with_one_scale_and_preserves_touching_hands():
    landmarks=clasped_pose()
    figure=Figure(id='contact',pose='photo',landmarks=landmarks,photo_aspect=1.5)
    skeleton=joints(figure)
    names=['shoulder_l','shoulder_r','elbow_l','elbow_r','wrist_l','wrist_r',
           'hip_l','hip_r','knee_l','knee_r','ankle_l','ankle_r']
    indices=[11,12,13,14,15,16,23,24,25,26,27,28]
    actual=np.array([skeleton[name][[0,2]] for name in names])
    photo=np.array([[landmarks[i]['x']*1.5,-landmarks[i]['y']] for i in indices])
    centered=photo-photo.mean(axis=0)
    scale=np.sum(centered*(actual-actual.mean(axis=0)))/np.sum(centered**2)
    reconstructed=centered*scale+actual.mean(axis=0)
    assert np.max(np.linalg.norm(reconstructed-actual,axis=1))<1e-5
    assert np.linalg.norm(skeleton['wrist_l']-skeleton['wrist_r'])<1e-5
    assert np.linalg.norm(skeleton['elbow_l']-skeleton['wrist_l']) != pytest.approx(12,abs=.1)


@pytest.mark.parametrize('shape',['round','rectangle'])
@pytest.mark.parametrize('height',[40,90,150])
def test_bases_keep_physical_thickness_and_stability_after_resizing(shape,height):
    figure=Figure(id='stable',pose='leaning',height_mm=height,
                  base={'shape':shape,'thickness_mm':3,'margin_mm':5})
    mesh=generate(figure,False)
    restored=trimesh.load(BytesIO(mesh.export(file_type='stl')),file_type='stl',force='mesh')
    assert restored.is_watertight and len(restored.split())==1
    assert restored.extents[2]==pytest.approx(height,abs=.01)
    horizontal=restored.triangles[np.ptp(restored.triangles[:,:,2],axis=1)<1e-5]
    assert np.any(abs(horizontal[:,:,2].mean(axis=1)-3)<.001)
    report=validate(restored)
    stability=next(item for item in report['checks'] if item['name']=='Build-plate stability')
    assert stability['status']=='pass',stability


def test_detailed_garments_and_bouquet_remain_one_exported_solid():
    figure=Figure(id='details',pose='photo',landmarks=clasped_pose(),photo_aspect=1,
        base={'shape':'round'},appearance={'outfit':'wedding-dress','accessories':['bouquet','veil'],
        'garment':{'inferred':True,'upper':'dress','lower':'dress','sleeve_length':.2,'hem_length':1,'skirt_flare':.7}})
    skeleton=joints(figure)
    _,shoulder,up,lateral,forward=body_frame(skeleton)
    pieces=accessory_meshes(figure,skeleton,shoulder+up*18,up,lateral,forward)
    labels={mesh.metadata['style_feature'] for mesh in pieces}
    assert {'sleeve-cuff','garment-hem','dress-hem','bouquet-petals','bouquet-leaves','bouquet-ribbon'}<=labels
    petals=[mesh for mesh in pieces if mesh.metadata['style_feature']=='bouquet-petals']
    assert len(petals)>=30 and all(mesh.is_watertight for mesh in petals)
    mesh=generate(figure,False)
    restored=trimesh.load(BytesIO(mesh.export(file_type='stl')),file_type='stl',force='mesh')
    assert restored.is_watertight and restored.is_winding_consistent and len(restored.split())==1
    assert restored.extents[2]==pytest.approx(90,abs=.01)


def test_photo_hat_keeps_recognized_crown_style_instead_of_becoming_a_sun_hat():
    def crown(style):
        figure=Figure(id='hat',appearance={'accessories':['hat'],'hat_style':style,
                      'garment':{'inferred':True,'headwear':'brim'}})
        skeleton=joints(figure)
        _,shoulder,up,lateral,forward=body_frame(skeleton)
        pieces=accessory_meshes(figure,skeleton,shoulder+up*18,up,lateral,forward)
        return pieces[-1]
    formal,sun=crown('top-hat'),crown('wide-brim')
    assert formal.extents[2]>sun.extents[2]+5
    assert formal.is_watertight and sun.is_watertight
