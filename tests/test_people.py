"""Group completeness regressions independent of downloaded weights."""
from types import SimpleNamespace

import numpy as np
from PIL import Image

from backend.app import people, pose


def skeleton(center=.5, top=.12, bottom=.9):
    points = [dict(x=center, y=top, z=0., visibility=.95) for _ in range(33)]
    for left, right, y, half in ((11, 12, .32, .04), (23, 24, .57, .03),
                                 (25, 26, .73, .025), (27, 28, bottom, .025),
                                 (7, 8, top, .018)):
        for index, sign in ((left, -1), (right, 1)):
            points[index].update(x=center + half * sign, y=y)
    return points


def detection(center, width=.14):
    return dict(bbox=[center - width / 2, .06, center + width / 2, .96], score=.8)


def test_adjacent_people_survive_detector_nms_but_repeated_window_does_not():
    first = detection(.3)
    repeated = dict(bbox=[.232, .061, .372, .961], score=.7)
    neighbor = detection(.45)
    assert len(people.merge([first, repeated, neighbor])) == 2


def test_independent_count_retains_person_with_no_pose():
    resolved, unresolved = pose._match_group([detection(.25), detection(.75)],
                                             [skeleton(.25)], (1000, 800))
    assert len(resolved) == 1
    assert unresolved == [detection(.75)['bbox']]


def test_hugging_people_can_be_recovered_inside_one_object_box():
    resolved, unresolved = pose._match_group(
        [dict(bbox=[.15, .04, .85, .96], score=.9)],
        [skeleton(.35), skeleton(.65)], (1000, 800))
    assert len(resolved) == 2
    assert not unresolved


def test_duplicate_pose_does_not_inflate_people_count():
    first = skeleton(.3)
    duplicate = [dict(point) for point in first]
    for point in duplicate:
        point['x'] += .001
    resolved, unresolved = pose._match_group([detection(.3)], [first, duplicate], (1000, 800))
    assert len(resolved) == 1
    assert not unresolved


def test_retry_region_does_not_invent_a_person_or_remove_global_group():
    missing = [.7, .05, .95, .95]
    resolved, unresolved = pose._match_group([detection(.25)], [skeleton(.25)],
                                             (1000, 800), regions=[missing])
    assert len(resolved) == 1
    assert unresolved == [missing]


def test_no_visible_body_cannot_become_a_standing_figure():
    points = skeleton(.5)
    for index in (11, 12, 23, 24):
        points[index]['visibility'] = .1
    resolved, unresolved = pose._match_group([detection(.5)], [points], (1000, 800))
    assert not resolved
    assert len(unresolved) == 1


def prepare_group(monkeypatch, count):
    centers = np.linspace(.07, .93, count)
    monkeypatch.setattr(pose, 'available', lambda: (True, 'ready'))
    monkeypatch.setattr(pose, 'group_status', lambda: dict(available=True, model='test', detail='local'))
    monkeypatch.setattr(pose, '_detector', object())
    monkeypatch.setattr(people, 'detect', lambda image, regions=None: [detection(center, .1) for center in centers])
    monkeypatch.setattr(pose, '_detect_points', lambda *args: [skeleton(center) for center in centers])
    monkeypatch.setattr(pose, '_crop_pose_points', lambda *args: [])
    monkeypatch.setattr(pose.appearance, 'analyze_person', lambda *args: ({'outfit': 'none', 'accessories': []}, {'available': False}))
    monkeypatch.setattr(pose, 'photo_warnings', lambda image: [])


def test_group_analysis_returns_every_one_of_six_people(monkeypatch):
    prepare_group(monkeypatch, 6)
    result = pose.analyze_group(Image.new('RGB', (1200, 800)))
    assert result['detected_people'] == 6
    assert len(result['subjects']) == 6
    assert not result['unresolved_regions']
    assert result['overflow'] is False
    assert all(subject['photo_aspect'] == 1.5 for subject in result['subjects'])
    assert len({subject['id'] for subject in result['subjects']}) == 6


def test_overflow_is_explicit_instead_of_a_six_person_success(monkeypatch):
    prepare_group(monkeypatch, 7)
    result = pose.analyze_group(Image.new('RGB', (1200, 800)))
    assert result['detected_people'] == 7
    assert len(result['subjects']) == 6
    assert len(result['unresolved_regions']) == 1
    assert result['overflow'] is True
    assert any('no partial group' in warning for warning in result['warnings'])


def test_empty_photo_is_explained_without_a_placeholder(monkeypatch):
    prepare_group(monkeypatch, 0)
    result = pose.analyze_group(Image.new('RGB', (1200, 800)))
    assert result['detected_people'] == 0
    assert result['subjects'] == []
    assert any('No people' in warning for warning in result['warnings'])


def test_model_with_wrong_checksum_is_not_loaded(monkeypatch, tmp_path):
    path = tmp_path / 'bad.onnx'
    path.write_bytes(b'not a trusted model')
    monkeypatch.setattr(people, 'MODEL_PATH', path)
    monkeypatch.setattr(people, '_verified', None)
    ready, detail = people.available()
    assert ready is False
    assert 'checksum' in detail


def test_person_detector_decode_excludes_low_person_score_other_class():
    outputs = []
    for size in (80, 40, 20):
        output = np.full((1, size, size, 85), -30., dtype=np.float32)
        outputs.append(output)
    # First object has high objectness but is a dog, not a person.
    outputs[0][0, 5, 5, :4] = [0., 0., 2., 2.]
    outputs[0][0, 5, 5, 4] = 10
    outputs[0][0, 5, 5, 21] = 10
    # Second is a person. Decode to original coordinates at 2x input scale.
    outputs[0][0, 10, 10, :4] = [0., 0., 2., 3.]
    outputs[0][0, 10, 10, 4:6] = 10
    found = people._decode(outputs, 2., (10, 20), (400, 400), .38)
    assert len(found) == 1
    assert found[0]['bbox'][0] < .125 < found[0]['bbox'][2]
    assert found[0]['bbox'][1] < .15 < found[0]['bbox'][3]


def test_retry_overlapping_known_person_requires_an_additional_identity():
    known = skeleton(.4)
    region = [.25, .04, .75, .96]
    resolved, unresolved = pose._match_group([detection(.4)], [known], (1000, 800),
                                             regions=[region], baseline_points=[known])
    assert len(resolved) == 1
    assert unresolved == [region]


def test_retry_can_resolve_a_new_person_beside_an_existing_one():
    known, recovered = skeleton(.4), skeleton(.65)
    region = [.25, .04, .75, .96]
    resolved, unresolved = pose._match_group([detection(.4), detection(.65)],
                                             [known, recovered], (1000, 800),
                                             regions=[region], baseline_points=[known])
    assert len(resolved) == 2
    assert not unresolved


def test_window_sliver_is_merged_but_child_in_front_is_preserved():
    adult = dict(bbox=[.2, .1, .5, .95], score=.9)
    sliver = dict(bbox=[.22, .12, .31, .94], score=.7)
    child = dict(bbox=[.3, .5, .45, .95], score=.8)
    assert len(people.merge([adult, sliver, child])) == 2


def test_upside_down_hallucination_from_half_height_crop_is_not_a_person():
    false_pose = skeleton(.5)
    false_pose[0]['y'] = .8
    resolved, unresolved = pose._match_group([detection(.5)], [false_pose], (1000, 800))
    assert not resolved
    assert len(unresolved) == 1


def test_high_visibility_face_sized_body_is_unresolved_not_an_extra_person():
    collapsed = skeleton(.3)
    for point in collapsed:
        point['y'] = .2 + point['y'] * .05
        point['visibility'] = 1.
    resolved, unresolved = pose._match_group([detection(.3)], [collapsed], (1275, 935))
    assert resolved == []
    assert unresolved == [detection(.3)['bbox']]


def test_plausible_alternative_replaces_collapsed_pose_of_the_same_person():
    valid = skeleton(.3)
    collapsed = [dict(point, y=.2 + point['y'] * .05) for point in valid]
    resolved, unresolved = pose._match_group([detection(.3)], [collapsed, valid], (1275, 935))
    assert len(resolved) == 1
    assert resolved[0][1] is valid
    assert not unresolved
