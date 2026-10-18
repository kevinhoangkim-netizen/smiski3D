"""Photo measurements change real garment solids while retaining bent limbs."""
from io import BytesIO

import numpy as np
import pytest
import trimesh

from backend.app.figure_style import accessory_meshes
from backend.app.geometry import body_frame, generate, joints
from backend.app.schemas import Figure


def photo_style(*, pose='standing', accessories=(), **garment):
    return Figure(id='measured', pose=pose, base={'shape': 'none'}, appearance={
        'outfit': 'none', 'accessories': list(accessories),
        'garment': {'inferred': True, **garment}})


def additions(figure):
    pose = joints(figure)
    _, shoulder, up, lateral, forward = body_frame(pose)
    return accessory_meshes(figure, pose, shoulder + up * 18, up, lateral, forward)


def pieces(figure, feature):
    return [mesh for mesh in additions(figure) if mesh.metadata['style_feature'] == feature]


def test_inferred_photo_clothing_is_geometry_even_when_legacy_outfit_is_none():
    bare = generate(Figure(id='bare', base={'shape': 'none'}), transformed=False)
    wearing = generate(photo_style(upper='shirt', lower='pants', sleeve_length=1,
                                   trouser_length=1), transformed=False)
    assert wearing.is_watertight and len(wearing.split()) == 1
    assert wearing.volume > bare.volume * 1.04
    assert pieces(photo_style(upper='shirt'), 'upper-garment')


def test_unknown_measurements_do_not_select_a_stock_costume():
    figure = photo_style(upper='unknown', lower='unknown')
    figure.appearance.outfit = 'wedding-dress'
    assert additions(figure) == []


def test_sleeve_coverage_extends_along_the_bent_forearm():
    short = photo_style(pose='arms-raised', upper='shirt', sleeve_length=.25)
    long = photo_style(pose='arms-raised', upper='shirt', sleeve_length=1)
    short_sleeves, long_sleeves = pieces(short, 'sleeve'), pieces(long, 'sleeve')
    short_volume = sum(mesh.volume for mesh in short_sleeves)
    long_volume = sum(mesh.volume for mesh in long_sleeves)
    assert long_volume > short_volume * 2
    assert max(mesh.bounds[1, 2] for mesh in long_sleeves) > max(
        mesh.bounds[1, 2] for mesh in short_sleeves) + 10
    # Both forearm coverings remain aligned with the photographed joint chain.
    pose = joints(long)
    for side in ('l', 'r'):
        arm = pose['wrist_' + side] - pose['elbow_' + side]
        matching = [mesh for mesh in long_sleeves
                    if np.linalg.norm(mesh.center_mass - (pose['wrist_' + side]
                                                          + pose['elbow_' + side]) / 2) < 1]
        assert matching
        axis = matching[0].principal_inertia_vectors[0]
        assert abs(np.dot(axis, arm) / np.linalg.norm(arm)) > .95


def test_measured_short_and_long_trousers_change_the_leg_coverage():
    short = pieces(photo_style(lower='shorts', trouser_length=.35), 'trouser-leg')
    long = pieces(photo_style(lower='pants', trouser_length=1), 'trouser-leg')
    assert sum(mesh.volume for mesh in long) > sum(mesh.volume for mesh in short) * 1.7
    assert min(mesh.bounds[0, 2] for mesh in long) < min(
        mesh.bounds[0, 2] for mesh in short) - 12


def test_measured_hem_and_flare_change_real_skirt_silhouette():
    short = pieces(photo_style(lower='skirt', hem_length=.25, skirt_flare=.2), 'skirt')[0]
    long = pieces(photo_style(lower='skirt', hem_length=1, skirt_flare=.2), 'skirt')[0]
    wide = pieces(photo_style(lower='skirt', hem_length=1, skirt_flare=.9), 'skirt')[0]
    assert short.is_watertight and long.is_watertight and wide.is_watertight
    assert long.bounds[0, 2] < short.bounds[0, 2] - 15
    assert wide.extents[0] > long.extents[0] + 12
    assert wide.volume > long.volume * 1.4


def test_seated_skirt_moves_forward_with_the_knees():
    skirt = pieces(photo_style(pose='seated', lower='skirt', hem_length=.7), 'skirt')[0]
    standing = pieces(photo_style(lower='skirt', hem_length=.7), 'skirt')[0]
    assert skirt.center_mass[1] < standing.center_mass[1] - 8


@pytest.mark.parametrize('pose,upper,lower,sleeve,hem', [
    ('standing', 'shirt', 'pants', 1, .25),
    ('arms-raised', 'shirt', 'shorts', .25, .25),
    ('seated', 'dress', 'dress', .4, .9),
    ('crouching', 'jacket', 'skirt', 1, .6),
    ('crouching', 'dress', 'dress', 0, 1),
    ('seated', 'shirt', 'skirt', .5, 0),
])
def test_photo_garments_round_trip_as_connected_printable_solids(pose, upper, lower, sleeve, hem):
    figure = photo_style(pose=pose, upper=upper, lower=lower, sleeve_length=sleeve,
                         hem_length=hem, skirt_flare=.5, trouser_length=.8)
    mesh = generate(figure)
    restored = trimesh.load(BytesIO(mesh.export(file_type='stl')), file_type='stl', force='mesh')
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0
    assert restored.is_watertight and restored.is_winding_consistent
    assert len(restored.split()) == 1
    assert restored.extents[2] == pytest.approx(figure.height_mm, abs=.01)


@pytest.mark.parametrize('accessory', ['bag', 'scarf', 'glasses'])
def test_new_visible_accessories_are_fused_into_the_actual_export(accessory):
    plain = photo_style(upper='shirt', lower='pants')
    adorned = photo_style(upper='shirt', lower='pants', accessories=[accessory])
    mesh = generate(adorned)
    bare = generate(plain)
    restored = trimesh.load(BytesIO(mesh.export(file_type='stl')), file_type='stl', force='mesh')
    assert len(restored.split()) == 1 and restored.is_watertight
    assert pieces(adorned, accessory)
    assert abs(mesh.volume - bare.volume) > 5


def test_measured_beanie_does_not_turn_into_a_top_hat():
    beanie = photo_style(headwear='beanie')
    meshes = pieces(beanie, 'hat')
    assert len(meshes) == 2
    assert all(mesh.is_watertight and mesh.volume > 0 for mesh in meshes)
    mesh = generate(beanie)
    assert mesh.is_watertight and len(mesh.split()) == 1


def test_unknown_photo_hat_does_not_fall_back_to_legacy_top_hat():
    figure = photo_style(headwear='unknown', accessories=['hat'])
    assert pieces(figure, 'hat') == []
