"""Meaningful abstention, privacy, crop and group-layout boundary tests."""
from io import BytesIO
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from backend.app import appearance, pose
from backend.app.schemas import Appearance


@pytest.fixture(autouse=True)
def no_implicit_garment_model(monkeypatch):
    # Unit tests isolate CLIP decisions; real parser inference has its own test.
    monkeypatch.setattr(appearance.garment_parser, 'analyze_person', lambda *args:
                        (None, {'status': 'unavailable', 'warnings': []}))


def person(center=.5, scale=1):
    points = [dict(x=center, y=.2, z=0, visibility=1) for _ in range(33)]
    coords = {0: (0, .12), 11: (-.08, .26), 12: (.08, .26),
              15: (-.12, .55), 16: (.12, .55), 23: (-.06, .52), 24: (.06, .52),
              25: (-.06, .7), 26: (.06, .7), 27: (-.06, .9), 28: (.06, .9)}
    for k, (x, y) in coords.items():
        points[k].update(x=center + x * scale, y=.1 + (y - .1) * scale)
    return dict(landmarks=points, bbox=[center - .15 * scale, .1, center + .15 * scale, .1 + .8 * scale], warnings=[])


def test_missing_appearance_weights_do_not_claim_detection(monkeypatch, tmp_path):
    monkeypatch.setattr(appearance, 'MODEL_DIR', tmp_path)
    result, detail = appearance.analyze_person(Image.new('RGB', (200, 400)), person())
    assert result['outfit'] == 'none' and result['accessories'] == []
    assert detail['status'] == 'unavailable'
    assert detail['confidence'] is None
    assert not appearance.status()['available']


@pytest.mark.parametrize('scores', [
    {'suit': .3, 'casual': .29, 'none': .1},  # ambiguous, even though score high
    {'suit': .12, 'casual': .02, 'none': .01},  # very weak but apparently distinct
    {'other': .5, 'suit': .2, 'none': .1},  # unsupported clothing, no invented replacement
    {'suit': float('nan'), 'casual': .1, 'none': .01},
])
def test_weak_or_unsupported_outfits_abstain(scores):
    assert not appearance.choose_label(scores, 'outfit')[1]


def test_strong_match_is_a_suggestion_not_a_probability():
    label, accepted, score, margin = appearance.choose_label({'suit': .34, 'casual': .25, 'none': .1}, 'outfit')
    assert label == 'suit' and accepted
    assert score == pytest.approx(.34) and margin == pytest.approx(.09)


def test_person_crops_exclude_other_side_of_group():
    pixels = np.zeros((400, 600, 3), dtype=np.uint8)
    pixels[:, :300, 0] = 255
    pixels[:, 300:, 2] = 255
    crops = appearance.subject_crops(Image.fromarray(pixels), person(.22, .8))
    assert set(crops) == {'outfit', 'upper', 'lower', 'hat', 'bouquet', 'veil', 'bowtie',
                          'bag', 'scarf', 'glasses', 'mobility', 'object', 'bouquet-left', 'bouquet-right'}
    for crop in crops.values():
        assert crop.width > 0 and crop.height > 0
        assert np.asarray(crop)[:, :, 2].max() == 0


def test_automatic_suit_and_bouquet_follow_shared_schema(monkeypatch):
    class FakeRuntime:
        def score(self, crops):
            return {'outfit': {'suit': .36, 'casual': .23, 'none': .1},
                    'hat': {'none': .4, 'cap': .2},
                    'bouquet': {'yes': .35, 'no': .2},
                    'veil': {'no': .4, 'yes': .2},
                    'bowtie': {'yes': .35, 'no': .2},
                    'bouquet-left': {'yes': .35, 'no': .2},
                    'bouquet-right': {'no': .35, 'yes': .2}}
    monkeypatch.setattr(appearance, 'available', lambda: (True, 'test'))
    monkeypatch.setattr(appearance, '_runtime', FakeRuntime())
    result, detail = appearance.analyze_person(Image.new('RGB', (600, 800)), person())
    assert Appearance(**result).outfit == 'suit'
    assert result['accessories'] == ['bouquet', 'bowtie']
    assert result['bouquet_hand'] == 'left'
    assert detail['status'] == 'uncertain' and detail['confidence'] is None
    assert result['garment']['upper'] == 'jacket'
    assert result['garment']['lower'] == 'pants'
    assert result['garment']['inferred']


def test_classifier_failure_preserves_explicit_manual_fallback(monkeypatch):
    class BrokenRuntime:
        def score(self, crops):
            raise RuntimeError('runtime missing')
    monkeypatch.setattr(appearance, 'available', lambda: (True, 'test'))
    monkeypatch.setattr(appearance, '_runtime', BrokenRuntime())
    result, detail = appearance.analyze_person(Image.new('RGB', (300, 600)), person())
    assert not result['accessories'] and result['outfit'] == 'none'
    assert detail['status'] == 'unavailable'


def test_major_unsupported_mobility_aid_is_explicit_not_silently_omitted(monkeypatch):
    class Runtime:
        def score(self,crops):
            return {'mobility':{'wheelchair':.38,'none':.2,'cane':.12,'other':.1}}
    monkeypatch.setattr(appearance,'available',lambda:(True,'test'))
    monkeypatch.setattr(appearance,'_runtime',Runtime())
    _,detail=appearance.analyze_person(Image.new('RGB',(600,800)),person())
    assert detail['unsupported_items']==['wheelchair']
    assert detail['blocking']
    assert any('would omit' in warning for warning in detail['warnings'])


def test_ambiguous_walking_stick_does_not_block_as_confirmed_cane():
    assert not appearance.choose_label({'cane':.3,'none':.26,'other':.24,'wheelchair':.1},'mobility')[1]


def test_neighbor_flowers_are_not_attached_without_local_visible_hand_match(monkeypatch):
    class Runtime:
        def score(self,crops):
            return {'outfit':{'suit':.36,'casual':.23,'none':.1},
                    'bouquet':{'yes':.36,'no':.2},
                    'bouquet-left':{'no':.4,'yes':.2},
                    'bouquet-right':{'no':.4,'yes':.2}}
    monkeypatch.setattr(appearance,'available',lambda:(True,'test'))
    monkeypatch.setattr(appearance,'_runtime',Runtime())
    result,detail=appearance.analyze_person(Image.new('RGB',(600,800)),person())
    assert 'bouquet' not in result['accessories']
    assert any('no bouquet was assigned' in warning for warning in detail['warnings'])


def test_layout_preserves_order_and_relative_projected_size():
    people = [person(.2, 1), person(.75, .7)]
    pose.suggest_layout(people, (1000, 1000))
    assert people[0]['suggested_transform']['x'] < 0 < people[1]['suggested_transform']['x']
    assert people[0]['suggested_height_mm'] > people[1]['suggested_height_mm']
    assert all(s['suggested_transform']['y'] == 0 for s in people)
    assert any('perspective' in warning for warning in people[0]['warnings'])


def test_hidden_legs_do_not_invent_relative_height():
    subject = person()
    for k in (27, 28):
        subject['landmarks'][k]['visibility'] = 0
    pose.suggest_layout([subject], (1000, 1000))
    assert subject['suggested_height_mm'] == 90
    warnings = pose._subject_warnings(subject['landmarks'], subject['bbox'], .6, (1000, 1000))
    assert any('Feet are hidden' in warning for warning in warnings)


def test_window_dedup_requires_matching_head_and_pelvis():
    first = person(.4)['landmarks']
    duplicate = [dict(point, x=point['x'] + .003) for point in first]
    neighbor = person(.52)['landmarks']
    assert pose._same_person(first, duplicate, (1200, 800))
    assert not pose._same_person(first, neighbor, (1200, 800))


def test_matching_visible_head_deduplicates_uncertain_occluded_body():
    first = person(.4)['landmarks']
    first[7]['x'], first[8]['x'] = .36, .44
    occluded = [dict(point, x=point['x'] + .003) for point in first]
    occluded[23]['x'] += .15
    occluded[24]['x'] += .15
    assert pose._same_person(first, occluded, (1200, 800))


def test_six_person_windows_remap_and_deduplicate_original_coordinates():
    image = Image.new('RGB', (1500, 666))
    originals = []
    for row in range(2):
        for column in range(3):
            points = person((column + .5) / 3, .3)['landmarks']
            for point in points:
                point['y'] += row * .5
            originals.append(points)
    tiles = [(left, top) for top in (0, 250) for left in (0, 437, 875)]
    class Detector:
        calls = 0
        def detect(self, data):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(pose_landmarks=[])
            left, top = tiles[self.calls - 2]
            result = []
            for points in originals:
                center_x = points[23]['x'] * image.width
                center_y = points[23]['y'] * image.height
                if left <= center_x <= left + 625 and top <= center_y <= top + 416:
                    remapped = [SimpleNamespace(x=(p['x'] * image.width - left) / 625,
                                               y=(p['y'] * image.height - top) / 416,
                                               z=0, visibility=1) for p in points]
                    # Some overlapping inference windows report the same person.
                    result.extend([remapped, remapped])
            return SimpleNamespace(pose_landmarks=result)
    mp = SimpleNamespace(Image=lambda **kwargs: kwargs['data'], ImageFormat=SimpleNamespace(SRGB='RGB'))
    detected = pose._detect_points(Detector(), image, mp)
    assert len(detected) == 6
    assert sorted(round(points[0]['x'], 5) for points in detected) == sorted(round(points[0]['x'], 5) for points in originals)
    assert sorted(round(points[0]['y'], 5) for points in detected) == sorted(round(points[0]['y'], 5) for points in originals)


def test_photo_subject_keeps_original_coordinates_and_image_aspect(monkeypatch):
    points = person()['landmarks']
    monkeypatch.setattr(pose, '_detector', object())
    monkeypatch.setattr(pose, '_detect_points', lambda *args: [points])
    monkeypatch.setattr(appearance, 'analyze_person', lambda *args: (appearance.default_appearance(), {'status': 'unavailable'}))
    detected = pose.analyze(Image.new('RGB', (800, 400)))
    assert detected[0]['photo_aspect'] == 2
    assert detected[0]['landmarks'] == points


def test_photo_quality_advisories_are_not_fake_people():
    assert any('small photo' in warning for warning in pose.photo_warnings(Image.new('RGB', (100, 100))))


def test_downloader_rejects_bad_checksum_and_keeps_no_partial_model(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/download_appearance_model.py'
    spec = importlib.util.spec_from_file_location('download_appearance_model', script)
    downloader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(downloader)
    metadata = {'sha': 'a' * 40,
                'siblings': [{'rfilename': name, 'lfs': {'sha256': '0' * 64}} for name in downloader.FILES]}
    replies = iter([BytesIO(json.dumps(metadata).encode()), BytesIO(b'bad model contents')])
    monkeypatch.setattr(downloader.urllib.request, 'urlopen', lambda *args, **kwargs: next(replies))
    with pytest.raises(RuntimeError, match='SHA-256 mismatch'):
        downloader.install(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not appearance.available()[0], reason='Optional verified local CLIP weights/runtime not installed')
def test_real_local_clip_runs_and_abstains_on_blank_image():
    # Exercises the actual tokenizer, checked model files, CPU ONNX sessions,
    # image preprocessing and score calculation. No fixture photo or network.
    runtime = appearance.ClipRuntime()
    scores = runtime.score({'outfit': Image.new('RGB', (224, 224), (240, 240, 240))})['outfit']
    assert set(scores) == set(appearance.LABELS['outfit'])
    assert all(np.isfinite(value) for value in scores.values())
    assert not appearance.choose_label(scores, 'outfit')[1]
