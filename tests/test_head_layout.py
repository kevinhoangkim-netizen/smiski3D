"""Enlarged heads remain distinct without changing photographed limb directions."""
from copy import deepcopy

import numpy as np
import pytest

from backend.app import geometry, pose
from backend.app.schemas import Figure


def subject(center, offset=0, lean=0, height=90):
    points=[dict(x=center,y=.14,z=0,visibility=1) for _ in range(33)]
    for index,x,y in [(0,center+lean*1.3,.14),(2,center+lean*1.3-.015,.14),
                      (5,center+lean*1.3+.015,.14),(7,center+lean*1.3-.025,.14),
                      (8,center+lean*1.3+.025,.14),
                      (11,center+lean-.045,.32),(12,center+lean+.045,.32),
                      (13,center-.065,.46),(14,center+.065,.46),
                      (15,center-.055,.58),(16,center+.055,.58),
                      (23,center-.035,.57),(24,center+.035,.57),
                      (25,center-.03,.74),(26,center+.03,.74),
                      (27,center-.03,.91),(28,center+.03,.91)]:
        points[index].update(x=x,y=y)
    return dict(landmarks=points,warnings=[],suggested_height_mm=height,
                suggested_transform=dict(x=offset,y=0,scale=1))


def figure(item,index=0,appearance=None):
    return Figure(id=f'head-{index}',pose='photo',landmarks=item['landmarks'],
        photo_aspect=1.5,height_mm=item['suggested_height_mm'],base={'shape':'none'},
        appearance=appearance or {},transform=item['suggested_transform'])


def test_dense_leaning_group_has_separate_heads_in_generated_geometry():
    subjects=[subject(.46,lean=.035),subject(.50,lean=-.025),subject(.55,lean=.01)]
    pose.suggest_layout(subjects,(1200,800))
    bounds=[np.asarray(geometry.generate(figure(item,index)).metadata['head_bounds_mm'])
            for index,item in enumerate(subjects)]
    for first,second in zip(bounds[:-1],bounds[1:]):
        assert second[0,0]>first[1,0]+1.4
    offsets=[item['suggested_transform']['x'] for item in subjects]
    assert offsets==sorted(offsets)
    assert offsets[0]+offsets[-1]==pytest.approx(0,abs=.01)
    assert all(any('heads remain distinct' in warning for warning in item['warnings'])
               for item in subjects)


def test_safe_group_does_not_move_or_receive_spacing_warning():
    subjects=[subject(.2,-80),subject(.5,0),subject(.8,80)]
    original=deepcopy(subjects)
    pose._separate_heads(subjects,(1200,800))
    assert subjects==original


def test_close_pair_moves_only_as_far_as_its_head_clearance_requires():
    subjects=[subject(.49,-1),subject(.51,1)]
    left=geometry.head_x_envelope(figure(subjects[0]))
    right=geometry.head_x_envelope(figure(subjects[1]))
    required=left[1]-right[0]+1.5
    pose._separate_heads(subjects,(1200,800))
    actual=subjects[1]['suggested_transform']['x']-subjects[0]['suggested_transform']['x']
    assert actual==pytest.approx(required,abs=.01)


def test_spacing_is_planar_preserves_heights_and_landmarks():
    subjects=[subject(.49,-1,lean=.02,height=65),subject(.51,1,lean=-.02,height=110)]
    original=deepcopy(subjects)
    pose._separate_heads(subjects,(1200,800))
    for before,after in zip(original,subjects):
        assert before['landmarks']==after['landmarks']
        assert before['suggested_height_mm']==after['suggested_height_mm']
        assert after['suggested_transform']['y']==0
        assert after['suggested_transform']['scale']==1
        first,second=geometry.joints(figure(before)),geometry.joints(figure(after))
        for key in first:
            np.testing.assert_array_equal(first[key],second[key])


def test_spacing_uses_photo_order_even_if_subject_input_is_unsorted():
    subjects=[subject(.53,2),subject(.47,-2),subject(.5,0)]
    pose._separate_heads(subjects,(1200,800))
    ordered=sorted(subjects,key=lambda item:item['landmarks'][23]['x'])
    assert np.all(np.diff([item['suggested_transform']['x'] for item in ordered])>30)


def test_spacing_does_not_generate_any_mesh(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('Analysis must not run full geometry generation')
    monkeypatch.setattr(geometry,'generate',forbidden)
    monkeypatch.setattr(geometry,'body_surface',forbidden)
    pose._separate_heads([subject(.49,-1),subject(.51,1)],(1200,800))


@pytest.mark.parametrize('appearance',[{}, {'accessories':['hat'],'hat_style':'top-hat'}])
def test_analytic_interval_contains_final_scaled_head(appearance):
    item=subject(.5,lean=.05,height=105)
    f=figure(item,appearance=appearance)
    interval=geometry.head_x_envelope(f)
    actual=np.asarray(geometry.generate(f,transformed=False).metadata['head_bounds_mm'])
    assert interval[0]<=actual[0,0]
    assert interval[1]>=actual[1,0]
