"""Actual export checks for garments, their supports and round-trip topology."""
from io import BytesIO

import numpy as np
import pytest
import trimesh

from backend.app.figure_style import accessory_meshes
from backend.app.geometry import body_frame, generate, joints
from backend.app.schemas import Figure


@pytest.mark.parametrize('pose,outfit,accessories,hand', [
    ('standing', 'suit', ['hat', 'bowtie'], 'both'),
    ('standing', 'wedding-dress', ['veil', 'bouquet'], 'both'),
    ('seated', 'wedding-dress', ['veil', 'bouquet'], 'left'),
    ('crouching', 'wedding-dress', ['veil', 'bouquet'], 'right'),
    ('arms-raised', 'casual', ['hat', 'bouquet'], 'both'),
])
def test_styled_poses_export_as_one_watertight_printable_solid(pose, outfit, accessories, hand):
    figure = Figure(id='styled', pose=pose, appearance={
        'outfit': outfit, 'accessories': accessories, 'bouquet_hand': hand})
    mesh = generate(figure)
    restored = trimesh.load(BytesIO(mesh.export(file_type='stl')), file_type='stl', force='mesh')
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert restored.is_watertight and restored.is_winding_consistent
    assert mesh.volume > 0 and len(mesh.split()) == 1
    assert restored.extents[2] == pytest.approx(figure.height_mm, abs=.01)


@pytest.mark.parametrize('style', ['cap', 'top-hat', 'wide-brim', 'beanie'])
def test_each_hat_is_a_real_connected_addition(style):
    figure = Figure(id='hat', base={'shape':'none'}, appearance={'accessories': ['hat'], 'hat_style': style})
    bare = generate(Figure(id='bare', base={'shape':'none'}), transformed=False)
    wearing = generate(figure, transformed=False)
    assert wearing.is_watertight and len(wearing.split()) == 1
    # Same target height means a hat changes actual geometry/proportions, not
    # just metadata or material. Its surface survives the export path.
    assert not np.allclose(bare.extents, wearing.extents, atol=.1)


def test_hat_follows_head_tilt_without_mutating_pose():
    figure = Figure(id='tilt', appearance={'accessories': ['hat']})
    j = joints(figure)
    _, shoulder, up, lateral, forward = body_frame(j)
    angle = np.radians(25)
    head_up = np.array([np.sin(angle), 0, np.cos(angle)])
    j['head_up'] = head_up.copy()
    j['head_lateral'] = np.array([np.cos(angle), 0, -np.sin(angle)])
    head = shoulder + head_up * 18
    additions = accessory_meshes(figure, j, head, up, lateral, forward)
    assert all(m.is_watertight and m.volume > 0 for m in additions)
    np.testing.assert_array_equal(j['head_up'], head_up)
    # The crown's top moves in the same lateral direction as the tilted head.
    assert additions[-1].center_mass[0] > head[0] + 3


def test_bouquet_remains_visible_in_front_of_default_body():
    # Scaling would hide a buried accessory in a volume-only assertion. Use
    # equal canonical geometry at the same target height and no pedestal.
    bare = generate(Figure(id='bare-body', base={'shape': 'none'}), transformed=False)
    holding = generate(Figure(id='flowers', base={'shape': 'none'},
                              appearance={'accessories': ['bouquet']}), transformed=False)
    assert holding.bounds[0, 1] < bare.bounds[0, 1] - 3
    assert holding.is_watertight and len(holding.split()) == 1


@pytest.mark.parametrize('head_up,head_lateral', [
    ([0, 0, 0], [0, 0, 0]),
    ([1, 0, 0], [1, 0, 0]),
])
def test_uncertain_head_axes_still_produce_closed_hat_solids(head_up, head_lateral):
    figure = Figure(id='uncertain-hat', appearance={'accessories': ['hat']})
    j = joints(figure)
    _, shoulder, up, lateral, forward = body_frame(j)
    j['head_up'] = np.array(head_up, float)
    j['head_lateral'] = np.array(head_lateral, float)
    additions = accessory_meshes(figure, j, shoulder + up * 18, up, lateral, forward)
    assert all(np.isfinite(m.vertices).all() and m.is_watertight and m.volume > 0
               for m in additions)
