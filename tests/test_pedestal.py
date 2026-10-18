"""Exercise geometry consumers actually send to printers, including STL welding."""
import io

import numpy as np
import pytest
import trimesh

from backend.app.pedestal import normalized_inscription, pedestal_mesh


def _figures():
    figures = []
    for x in (-24, 24):
        mesh = trimesh.creation.box(extents=[18, 12, 70])
        mesh.apply_translation([x, 0, 37.5])  # Bottom 2.5 overlaps a 3 mm base.
        figures.append(mesh)
    return figures


@pytest.mark.parametrize('shape', ['round', 'rectangle'])
@pytest.mark.parametrize('label', ['', 'ANNA & LEO 10/18/2026', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 !?'])
def test_shared_base_and_inscription_remain_one_closed_printable_solid(shape, label):
    figures = _figures()
    base = pedestal_mesh(figures, dict(shape=shape, inscription=label))
    assert base.is_watertight
    assert len(base.split()) == 1
    assert np.isclose(base.bounds[0, 2], 0)
    assert np.isclose(base.bounds[1, 2], 3.7 if label else 3)
    assert base.bounds[0, 0] <= -41
    assert base.bounds[1, 0] >= 41
    assert base.bounds[0, 1] <= -14
    assert base.bounds[1, 1] >= 14
    combined = trimesh.boolean.union([*figures, base], engine='manifold')
    restored = trimesh.load(io.BytesIO(combined.export(file_type='stl')), file_type='stl')
    assert restored.is_watertight
    assert restored.is_winding_consistent
    assert restored.volume > 0
    assert len(restored.split()) == 1


def test_disabled_or_empty_shared_base():
    assert pedestal_mesh(_figures(), {'shape': 'none'}) is None
    assert pedestal_mesh([], {'shape': 'rectangle'}) is None


def test_portable_lettering_is_explicit_about_normalization():
    assert normalized_inscription('José & Zoë ♥') == 'JOSE & ZOE ?'
    base = pedestal_mesh(_figures(), dict(shape='rectangle', inscription='José & Zoë ♥'))
    assert base.metadata['inscription'] == 'JOSE & ZOE ?'
