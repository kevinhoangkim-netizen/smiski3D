"""Resolve uncertain garment shapes without claiming that a guess was recognized.

Only compact numeric evidence is retained. No image, crop, mask or inferred
gender is used or saved here. Known garment categories and accessories survive
unchanged; the result is a deep copy suitable for saving and mesh generation.
"""
from __future__ import annotations

from copy import deepcopy
import math


UPPER = ('bare', 'shirt', 'jacket', 'dress')
LOWER = ('bare', 'pants', 'shorts', 'skirt', 'dress')
LENGTHS = ('sleeve_length', 'trouser_length', 'hem_length', 'skirt_flare')
ESTIMATED_FIELDS = ('upper', 'lower', *LENGTHS)
DEFAULT_GARMENT = dict(upper='unknown', lower='unknown', sleeve_length=.5,
                       hem_length=.25, skirt_flare=.35, trouser_length=1.,
                       headwear='none', inferred=True)


def _number(value, lo=0., hi=1.):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) and lo <= value <= hi else None


def compact_evidence(scores=None, measurements=None):
    """Bound the evidence carried alongside a saved appearance, never raw pixels."""
    scores, measurements = scores or {}, measurements or {}
    semantics = measurements.get('semantics') or {}
    result = {}
    for field, categories in (('upper', UPPER), ('lower', LOWER)):
        clip = {label: number for label in (*categories, 'other')
                if (number := _number((scores.get(field) or {}).get(label), -1, 1)) is not None}
        fractions = {label: number for label in categories if label != 'bare'
                     if (number := _number((semantics.get(field + '_fractions') or {}).get(label))) is not None}
        if clip:
            result[field + '_scores'] = clip
        if fractions:
            result[field + '_fractions'] = fractions
        if semantics.get(field) in categories:
            result[field + '_region'] = semantics[field]
    for field in ('torso_clothes', 'torso_skin', *LENGTHS):
        if (number := _number(measurements.get(field))) is not None:
            result[field] = number
    return result


def _rank(field, evidence):
    """Combine independent similarity/region support, not probability scores."""
    categories = UPPER if field == 'upper' else LOWER
    # Bare is intentionally excluded when resolving *unknown* clothing. Existing
    # confirmed bare classifications are preserved by estimate_appearance.
    categories = tuple(label for label in categories if label != 'bare')
    values = evidence.get(field + '_scores') or {}
    scores = {label: value for label in (*categories, 'other', 'bare')
              if (value := _number(values.get(label), -1, 1)) is not None}
    fractions = evidence.get(field + '_fractions') or {}
    region = evidence.get(field + '_region')
    best_clip = max(scores, key=scores.get) if scores else None
    clip_usable = best_clip not in ('other', 'bare') or (
        max((scores.get(label, -1) for label in categories), default=-1)
        >= scores.get(best_clip, 1) - .025)
    strength, supports = {}, {}
    for label in categories:
        score = scores.get(label, -1)
        clip = max(0., min(1., (score - .19) / .2)) if clip_usable and score >= .21 else 0.
        fraction = _number(fractions.get(label)) or 0.
        # Anatomical region occupancy has an independent spatial constraint;
        # small fragments below 8% are not enough to choose a garment.
        mask = min(1., fraction / .55) if fraction >= .08 else 0.
        if region == label:
            mask = max(mask, .5)
        if clip or mask:
            strength[label] = clip * .55 + mask * .65
            supports[label] = (bool(clip), bool(mask))
    if not strength:
        return None, 'no usable person-local clothing evidence'
    ranked = sorted(strength, key=lambda label: (-strength[label], categories.index(label)))
    selected = ranked[0]
    clip_winner = max((label for label in categories if scores.get(label, -1) >= .21),
                      key=lambda label: scores[label], default=None) if clip_usable else None
    mask_labels = [label for label in categories if (_number(fractions.get(label)) or 0.) >= .08]
    mask_winner = max(mask_labels, key=lambda label: fractions[label], default=None) or (
        region if region in categories else None)
    conflict = clip_winner and mask_winner and clip_winner != mask_winner
    near_tie = len(ranked) > 1 and strength[selected] - strength[ranked[1]] < .08
    if conflict:
        reason = 'the clothing signals disagree; their combined person-local support is strongest for this shape'
    elif near_tie:
        reason = 'the closest clothing signals are ambiguous; this is the strongest supported simple shape'
    elif supports[selected] == (True, True):
        reason = 'the clothing similarity and visible garment region support this shape'
    elif supports[selected][1]:
        reason = 'the visible garment region supports this shape'
    else:
        reason = 'the strongest usable clothing similarity supports this shape'
    return selected, reason


def estimate_appearance(appearance: dict, evidence: dict | None = None) -> dict:
    """Fill only unknown clothing, retaining provenance and all known details.

    Compact evidence can come from a fresh analysis or an older saved project.
    If it is absent, known outfit/category/length relationships come first, then
    a plainly labelled modest shirt/trousers fallback lets the group proceed.
    """
    result = deepcopy(appearance)
    garment = result.setdefault('garment', deepcopy(DEFAULT_GARMENT))
    supplied_dimensions = set(garment).intersection(LENGTHS)
    for field, value in DEFAULT_GARMENT.items():
        garment.setdefault(field, value)
    evidence = deepcopy(evidence if evidence is not None else result.get('garment_evidence') or {})
    fields = [field for field in garment.get('estimated_fields', []) if field in ESTIMATED_FIELDS]
    notes = [str(note)[:500] for note in garment.get('estimation_notes', [])][:10]

    def note(field, message):
        if field not in fields:
            fields.append(field)
        if message not in notes:
            notes.append(message[:500])

    def dimension(field, default=None):
        measured = _number(evidence.get(field))
        if measured is not None:
            # This dimension belongs to the previously unknown garment only.
            garment[field] = measured
            return
        if default is not None:
            garment[field] = default
        note(field, f'{field.replace("_", " ").capitalize()} is estimated where the photo does not supply a measurement.')

    unknown_upper = garment['upper'] == 'unknown'
    unknown_lower = garment['lower'] == 'unknown'
    if not unknown_upper and not unknown_lower:
        # Do not add empty metadata to legacy known appearances unnecessarily.
        return result

    outfit = result.get('outfit', 'none')
    for field, unknown in (('upper', unknown_upper), ('lower', unknown_lower)):
        if not unknown:
            continue
        other = 'lower' if field == 'upper' else 'upper'
        if garment[other] == 'dress' or outfit == 'wedding-dress':
            selected, reason = 'dress', 'continued the supported one-piece dress instead of inventing separate clothing'
        elif outfit == 'suit':
            selected = 'jacket' if field == 'upper' else 'pants'
            reason = 'continued the supported suit with a coherent garment shape'
        else:
            selected, reason = _rank(field, evidence)
            if selected == 'dress' and garment[other] not in ('dress', 'unknown'):
                # Preserve a confirmed separate garment. A low-confidence dress
                # guess must not overwrite a recognized shirt or trousers.
                selected = 'skirt' if field == 'lower' else 'shirt'
                reason = 'used a separate garment to preserve the already identified clothing'
            if field == 'upper' and selected is None and (_number(evidence.get('torso_clothes')) or 0.) > .2:
                selected, reason = 'shirt', 'visible torso clothing supports a simple upper garment'
            if field == 'lower' and (selected in (None, 'pants', 'shorts')):
                length = _number(evidence.get('trouser_length'))
                # Older projects may retain a valid short coverage measurement
                # even if their category was discarded after model conflict.
                if length is None and _number(garment.get('trouser_length')) is not None and garment['trouser_length'] < .62:
                    length = garment['trouser_length']
                if length is not None:
                    covered_shape = 'shorts' if length < .62 else 'pants'
                    coverage_reason = 'visible leg coverage supports short trousers' if covered_shape == 'shorts' else 'visible leg coverage supports long trousers'
                    reason = reason + '; ' + coverage_reason if selected is not None else coverage_reason
                    selected = covered_shape
            if selected is None:
                selected = 'shirt' if field == 'upper' else 'pants'
                reason = 'no usable clothing evidence remains; used a modest simple garment, without assumptions about gender'
        garment[field] = selected
        note(field, f'{field.capitalize()} clothing estimated as {selected}: {reason}.')
        if field == 'upper' and selected not in ('bare',):
            dimension('sleeve_length', 1. if selected == 'jacket' and 'sleeve_length' not in supplied_dimensions else None)
        if field == 'lower' and selected in ('pants', 'shorts'):
            dimension('trouser_length')
        elif field == 'lower' and selected in ('skirt', 'dress'):
            dimension('hem_length', 1. if outfit == 'wedding-dress' and evidence.get('hem_length') is None else None)
            dimension('skirt_flare')
    garment['inferred'] = True
    garment['estimated_fields'] = list(dict.fromkeys(fields))[:6]
    garment['estimation_notes'] = list(dict.fromkeys(notes))[:10]
    return result
