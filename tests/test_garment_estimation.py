"""Clothing uncertainty resolves to explicit estimates rather than a lost group."""
from copy import deepcopy
import json

import pytest

from backend.app.appearance import appearance_summary
from backend.app.garment_estimation import compact_evidence, estimate_appearance


def appearance():
    return dict(outfit='casual', accessories=['hat', 'bouquet'], hat_style='cap',
                bouquet_hand='left', garment=dict(upper='shirt', lower='unknown',
                sleeve_length=1., trouser_length=1., hem_length=.25, skirt_flare=.35,
                headwear='cap', inferred=True))


def with_garment(**changes):
    value = appearance()
    value['garment'].update(changes)
    return value


def test_reported_known_upper_unknown_lower_continues_with_labelled_estimate():
    original = appearance()
    before = deepcopy(original)
    result = estimate_appearance(original)
    assert original == before
    assert result['garment']['upper'] == 'shirt'
    assert result['garment']['lower'] == 'pants'
    assert result['garment']['sleeve_length'] == 1.
    assert 'lower' in result['garment']['estimated_fields']
    assert 'upper' not in result['garment']['estimated_fields']
    assert 'Trousers (estimated)' in appearance_summary(result)
    assert result['accessories'] == original['accessories']
    assert result['hat_style'] == 'cap' and result['bouquet_hand'] == 'left'


def test_conflicting_pants_skirt_evidence_uses_combined_support_and_notes_conflict():
    evidence = dict(lower_scores={'pants': .34, 'skirt': .30, 'other': .1},
                    lower_fractions={'skirt': .5, 'pants': .1}, lower_region='skirt')
    result = estimate_appearance(appearance(), evidence)
    assert result['garment']['lower'] == 'skirt'
    assert any('disagree' in note for note in result['garment']['estimation_notes'])
    assert 'lower' in result['garment']['estimated_fields']


def test_conflicting_signals_can_support_pants_instead_of_always_preferring_skirt():
    evidence = dict(lower_scores={'pants': .38, 'skirt': .22, 'other': .1},
                    lower_fractions={'skirt': .3, 'pants': .25}, lower_region='skirt')
    result = estimate_appearance(appearance(), evidence)
    assert result['garment']['lower'] == 'pants'
    assert any('disagree' in note for note in result['garment']['estimation_notes'])


@pytest.mark.parametrize('evidence', [dict(trouser_length=.42), {}])
def test_visible_short_coverage_survives_unknown_category(evidence):
    value = with_garment(trouser_length=.42)
    result = estimate_appearance(value, evidence)
    assert result['garment']['lower'] == 'shorts'
    assert result['garment']['trouser_length'] == .42
    assert ('trouser_length' in result['garment']['estimated_fields']) == (not evidence)


def test_a_supported_one_piece_dress_continues_below_waist():
    value = with_garment(upper='dress', hem_length=.86, skirt_flare=.7)
    result = estimate_appearance(value)
    assert result['garment']['upper'] == result['garment']['lower'] == 'dress'
    assert result['garment']['hem_length'] == .86
    assert result['garment']['skirt_flare'] == .7
    assert 'Dress (estimated)' in appearance_summary(result)


def test_bridal_and_suit_context_fill_coherent_unknown_parts():
    bridal = with_garment(upper='unknown')
    bridal['outfit'] = 'wedding-dress'
    dress = estimate_appearance(bridal)
    assert dress['garment']['upper'] == dress['garment']['lower'] == 'dress'
    assert dress['garment']['hem_length'] == 1.
    suit = with_garment(upper='unknown')
    suit['outfit'] = 'suit'
    result = estimate_appearance(suit)
    assert result['garment']['upper'] == 'jacket'
    assert result['garment']['lower'] == 'pants'


def test_legacy_suit_with_no_sleeve_dimension_estimates_long_jacket_sleeves():
    result = estimate_appearance({'outfit': 'suit', 'accessories': [],
                                 'garment': {'upper': 'unknown', 'lower': 'unknown'}})
    assert result['garment']['upper'] == 'jacket'
    assert result['garment']['sleeve_length'] == 1.
    assert 'sleeve_length' in result['garment']['estimated_fields']


def test_known_categories_dimensions_and_accessories_are_not_reinterpreted():
    value = with_garment(lower='shorts', trouser_length=.4)
    evidence = dict(upper_scores={'dress': .9}, lower_scores={'skirt': .9},
                    trouser_length=.95, sleeve_length=.1)
    assert estimate_appearance(value, evidence) == value


def test_saved_compact_evidence_is_used_without_photo_or_reupload():
    value = appearance()
    value['garment_evidence'] = compact_evidence(
        {'lower': {'pants': .22, 'shorts': .34, 'other': .1}},
        {'trouser_length': .4, 'semantics': {'lower': 'pants', 'lower_fractions': {'pants': .4}}})
    restored = json.loads(json.dumps(value))
    result = estimate_appearance(restored)
    assert result['garment']['lower'] == 'shorts'
    assert result['garment']['trouser_length'] == .4
    assert result['garment_evidence'] == value['garment_evidence']
    assert estimate_appearance(result) == result


def test_all_unknown_legacy_photo_uses_modest_shape_without_gender_or_accessory_guesses():
    value = with_garment(upper='unknown')
    value['accessories'] = []
    result = estimate_appearance(value)
    assert result['garment']['upper'] == 'shirt'
    assert result['garment']['lower'] == 'pants'
    assert result['accessories'] == []
    assert set(result['garment']['estimated_fields']) >= {'upper', 'lower'}
    assert any('no usable clothing evidence' in note for note in result['garment']['estimation_notes'])


def test_weak_or_obscured_scores_do_not_strip_clothing_or_force_a_supported_label():
    result = estimate_appearance(with_garment(upper='unknown'), {
        'upper_scores': {'bare': .4, 'shirt': .15},
        'lower_scores': {'other': .5, 'skirt': .22, 'pants': float('nan')}})
    assert result['garment']['upper'] == 'shirt'
    assert result['garment']['lower'] == 'pants'
    assert all('no usable clothing evidence' in note for note in result['garment']['estimation_notes']
               if 'clothing estimated' in note)


def test_compact_evidence_retains_only_bounded_garment_information():
    scores = {'upper': {'shirt': .3, 'fake': .5, 'dress': float('inf')},
              'lower': {'pants': .2, 'skirt': float('nan')}, 'hat': {'none': .9}}
    measurements = dict(torso_clothes=.8, torso_skin=-.1, sleeve_length=.4,
                        mask='private pixels', warnings=['not stored'], semantics={
                        'upper': 'shirt', 'upper_fractions': {'shirt': .8, 'fake': .2},
                        'lower': 'unknown', 'lower_fractions': {'pants': .2, 'dress': 2}})
    result = compact_evidence(scores, measurements)
    assert result == dict(upper_scores={'shirt': .3}, lower_scores={'pants': .2},
                          upper_fractions={'shirt': .8}, lower_fractions={'pants': .2},
                          upper_region='shirt', torso_clothes=.8, sleeve_length=.4)
    assert len(json.dumps(result)) < 400


def test_partial_dress_guess_does_not_overwrite_confirmed_separate_clothing():
    value = appearance()
    result = estimate_appearance(value, {'lower_scores': {'dress': .4, 'pants': .1}})
    assert result['garment']['upper'] == 'shirt'
    assert result['garment']['lower'] == 'skirt'
    assert any('preserve' in note for note in result['garment']['estimation_notes'])


def test_repeated_estimation_preserves_short_bounded_notes_and_provenance():
    value = appearance()
    result = estimate_appearance(value)
    assert estimate_appearance(result) == result
    assert len(result['garment']['estimated_fields']) <= 6
    assert len(result['garment']['estimation_notes']) <= 10
    assert all(len(note) <= 500 for note in result['garment']['estimation_notes'])


def test_fresh_analysis_persists_conflicting_evidence_and_returns_unblocked_estimate(monkeypatch):
    from PIL import Image
    from backend.app import appearance as analysis
    from backend.app.schemas import Appearance
    from tests.test_appearance import person

    class Runtime:
        def score(self, crops):
            return {'upper': {'shirt': .4, 'jacket': .2},
                    'lower': {'pants': .27, 'skirt': .26, 'other': .1}}

    measurements = dict(torso_clothes=.9, torso_skin=0., sleeve_length=1.,
                        trouser_length=None, hem_length=.6, skirt_flare=.4,
                        warnings=[], semantics={'upper': 'shirt', 'lower': 'skirt',
                        'lower_fractions': {'skirt': .4, 'pants': .05}})
    monkeypatch.setattr(analysis, 'available', lambda: (True, 'test'))
    monkeypatch.setattr(analysis, '_runtime', Runtime())
    monkeypatch.setattr(analysis.garment_parser, 'analyze_person',
                        lambda *args: (measurements, {'status': 'ready', 'warnings': []}))
    value, detail = analysis.analyze_person(Image.new('RGB', (600, 800)), person())
    validated = Appearance(**value)
    assert validated.garment.lower == 'skirt'
    assert validated.garment.hem_length == .6
    assert validated.garment_evidence.lower_scores == {'pants': .27, 'skirt': .26, 'other': .1}
    assert detail['warning_classes'] == ['inferred_clothing']
    assert not detail['blocking']
    assert 'Skirt (estimated)' in detail['summary']
    assert all('remains unknown' not in warning and 'no garment shape was invented' not in warning
               for warning in detail['warnings'])
