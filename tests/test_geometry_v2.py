"""Export-facing tests for clothing, photo gestures and complete group keepsakes."""
from io import BytesIO
import numpy as np
import pytest
import trimesh
from backend.app.geometry import generate, joints, project_scene, union_or_compose
from backend.app.schemas import Figure, Project


def assert_printable_round_trip(mesh, components=1):
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.volume>0
    restored=trimesh.load(BytesIO(mesh.export(file_type='stl')),file_type='stl',force='mesh')
    assert restored.is_watertight and restored.is_winding_consistent
    assert len(restored.split())==components
    np.testing.assert_allclose(restored.bounds,mesh.bounds,atol=.002)


@pytest.mark.parametrize('appearance',[
    {'outfit':'casual'},
    {'outfit':'suit','accessories':['bowtie','hat'],'hat_style':'top-hat'},
    {'outfit':'wedding-dress','accessories':['veil','bouquet']},
    {'accessories':['hat'],'hat_style':'cap'},
    {'accessories':['hat'],'hat_style':'wide-brim'},
    {'accessories':['bouquet'],'bouquet_hand':'left'},
    {'accessories':['bouquet'],'bouquet_hand':'right'},
])
def test_outfits_and_accessories_are_actual_attached_exported_solids(appearance):
    mesh=generate(Figure(id='styled',height_mm=100,appearance=appearance))
    assert mesh.extents[2]==pytest.approx(100,abs=.005)
    assert_printable_round_trip(mesh)


@pytest.mark.parametrize('pose',['seated','crouching','arms-raised'])
def test_accessories_follow_nonstanding_poses_without_detached_islands(pose):
    mesh=generate(Figure(id='bride',pose=pose,base={'shape':'none'},appearance={
        'outfit':'wedding-dress','accessories':['veil','bouquet']}))
    assert_printable_round_trip(mesh)


def landmarks():
    points=[dict(x=.5,y=.3,z=0,visibility=1) for _ in range(33)]
    for i,(x,y) in {
        0:(.58,.13),2:(.56,.12),5:(.60,.12),7:(.54,.15),8:(.62,.15),
        11:(.45,.3),12:(.65,.24),13:(.4,.45),14:(.75,.31),15:(.43,.56),16:(.82,.21),
        23:(.4,.56),24:(.54,.53),25:(.3,.70),26:(.65,.70),27:(.42,.82),28:(.73,.87)
    }.items():
        points[i].update(x=x,y=y)
    return points


def test_photo_retains_torso_lean_shoulder_angle_head_tilt_and_bent_knee():
    figure=Figure(id='photo',pose='photo',landmarks=landmarks(),base={'shape':'none'})
    skeleton=joints(figure)
    shoulder=(skeleton['shoulder_l']+skeleton['shoulder_r'])/2
    hip=(skeleton['hip_l']+skeleton['hip_r'])/2
    assert shoulder[0]>hip[0]+5  # torso leans as it does in the source
    assert skeleton['shoulder_r'][2]>skeleton['shoulder_l'][2]
    assert skeleton['knee_l'][0]<skeleton['hip_l'][0]
    assert skeleton['ankle_l'][0]>skeleton['knee_l'][0]  # visible bent knee
    assert skeleton['head_up'][0]>0
    assert_printable_round_trip(generate(figure))


def test_photo_horizontal_forearm_keeps_hand_closed_and_attached():
    points=landmarks()
    points[14].update(x=.72,y=.30,z=0)
    points[16].update(x=.85,y=.30,z=0)
    figure=Figure(id='horizontal',pose='photo',landmarks=points)
    assert_printable_round_trip(generate(figure))


def test_same_pixel_pose_keeps_its_angles_in_portrait_and_landscape_photos():
    physical=landmarks()
    physical[14].update(x=.75,y=.34,z=.05)
    physical[16].update(x=.85,y=.44,z=.08)
    def encoded(width,height):
        # The same 400-pixel person is centered on two differently sized photos.
        points=[dict(p,x=(p['x']*400+(width-400)/2)/width,
                     y=(p['y']*400+(height-400)/2)/height,
                     z=p['z']*400/width) for p in physical]
        return Figure(id='aspect',pose='photo',landmarks=points,photo_aspect=width/height)
    portrait=encoded(800,1200)
    landscape=encoded(1600,900)
    a,b=joints(portrait),joints(landscape)
    assert a.keys()==b.keys()
    for name in a:
        np.testing.assert_allclose(a[name],b[name],atol=1e-8,err_msg=name)
    diagonal=a['elbow_r']-a['shoulder_r']
    assert abs(diagonal[0])==pytest.approx(abs(diagonal[2]),abs=1e-8)
    # Aspect is part of the geometry cache key, so correcting metadata on the
    # same landmarks changes the actual sculpt instead of reusing a stale mesh.
    wrong=landscape.model_copy(update={'photo_aspect':1})
    assert not np.allclose(joints(wrong)['wrist_r'],b['wrist_r'])
    assert_printable_round_trip(generate(landscape))


def test_new_style_has_an_oversized_head_and_no_projecting_nose():
    mesh=generate(Figure(id='style',height_mm=100,base={'shape':'none'}))
    top=mesh.bounds[1,2]
    head_vertices=mesh.vertices[mesh.vertices[:,2]>top-38]
    assert np.ptp(head_vertices[:,0])>38
    # The face's center stays behind the head's curved front silhouette: there
    # is no projecting nose in the small region between eyes and mouth.
    front=head_vertices[(abs(head_vertices[:,0])<1.5)&
                        (head_vertices[:,2]>top-22)&(head_vertices[:,2]<top-17)]
    assert front[:,1].min()>=head_vertices[:,1].min()-.05
    groups=np.asarray(mesh.metadata['face_groups'])
    assert len(groups)==len(mesh.faces)
    assert set(groups)=={0,1}
    dark_centers=mesh.triangles_center[groups==1]
    assert len(dark_centers)>20
    assert len(dark_centers)<len(mesh.faces)*.02
    assert np.min(dark_centers[:,2])>top-28
    assert np.max(dark_centers[:,1])<0


def test_six_person_scene_shared_named_pedestal_is_one_printable_solid():
    project=Project(figures=[Figure(id=f'p{i}',height_mm=65+i*3,
        transform={'x':i*38,'y':(i%2)*8}) for i in range(6)],
        shared_base={'shape':'rectangle','inscription':'OUR FAMILY 2026'})
    scene=project_scene(project)
    assert [item[0] for item in scene][-1]=='__group_base__'
    assert len(scene)==7
    combined=union_or_compose([item[2] for item in scene])
    assert_printable_round_trip(combined)
    assert combined.bounds[0,2]==pytest.approx(0,abs=.001)
    assert combined.bounds[1,0]>200


def test_shared_base_disabled_keeps_separate_figure_export_components():
    project=Project(figures=[Figure(id='a'),Figure(id='b',transform={'x':120})])
    scene=project_scene(project)
    assert len(scene)==2
    assert_printable_round_trip(union_or_compose([item[2] for item in scene]),components=2)
