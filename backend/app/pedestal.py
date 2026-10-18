"""Shared printable pedestals with portable, raised bitmap lettering.

Lettering uses an embedded 5 x 7 font, so exports do not depend on OS fonts.
Lowercase is rendered uppercase; accented Latin letters are transliterated.
Unsupported characters become ``?``. The inscription is fitted to the front
of the top surface and can be smaller than the requested text size on a
narrow base. Every letter overlaps the base, so no glyph is a floating part.
"""
from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping

import numpy as np
import trimesh


# Rows run from the top of a glyph to the bottom. These intentionally simple
# strokes retain the rounded figures' restrained, toy-like visual language.
_FONT_ROWS = {
    'A': '01110/10001/10001/11111/10001/10001/10001',
    'B': '11110/10001/10001/11110/10001/10001/11110',
    'C': '01111/10000/10000/10000/10000/10000/01111',
    'D': '11110/10001/10001/10001/10001/10001/11110',
    'E': '11111/10000/10000/11110/10000/10000/11111',
    'F': '11111/10000/10000/11110/10000/10000/10000',
    'G': '01111/10000/10000/10111/10001/10001/01111',
    'H': '10001/10001/10001/11111/10001/10001/10001',
    'I': '11111/00100/00100/00100/00100/00100/11111',
    'J': '00111/00010/00010/00010/10010/10010/01100',
    'K': '10001/10010/10100/11000/10100/10010/10001',
    'L': '10000/10000/10000/10000/10000/10000/11111',
    'M': '10001/11011/10101/10101/10001/10001/10001',
    'N': '10001/11001/10101/10011/10001/10001/10001',
    'O': '01110/10001/10001/10001/10001/10001/01110',
    'P': '11110/10001/10001/11110/10000/10000/10000',
    'Q': '01110/10001/10001/10001/10101/10011/01111',
    'R': '11110/10001/10001/11110/10100/10010/10001',
    'S': '01111/10000/10000/01110/00001/00001/11110',
    'T': '11111/00100/00100/00100/00100/00100/00100',
    'U': '10001/10001/10001/10001/10001/10001/01110',
    'V': '10001/10001/10001/10001/10001/01010/00100',
    'W': '10001/10001/10001/10101/10101/10101/01010',
    'X': '10001/10001/01010/00100/01010/10001/10001',
    'Y': '10001/10001/01010/00100/00100/00100/00100',
    'Z': '11111/00001/00010/00100/01000/10000/11111',
    '0': '01110/10001/10011/10101/11001/10001/01110',
    '1': '00100/01100/00100/00100/00100/00100/01110',
    '2': '01110/10001/00001/00010/00100/01000/11111',
    '3': '11110/00001/00001/01110/00001/00001/11110',
    '4': '00010/00110/01010/10010/11111/00010/00010',
    '5': '11111/10000/10000/11110/00001/00001/11110',
    '6': '01110/10000/10000/11110/10001/10001/01110',
    '7': '11111/00001/00010/00100/01000/01000/01000',
    '8': '01110/10001/10001/01110/10001/10001/01110',
    '9': '01110/10001/10001/01111/00001/00001/01110',
    '&': '01100/10010/10100/01000/10101/10010/01101',
    '-': '00000/00000/00000/11111/00000/00000/00000',
    '+': '00000/00100/00100/11111/00100/00100/00000',
    '.': '00000/00000/00000/00000/00000/00110/00110',
    ',': '00000/00000/00000/00000/00110/00100/01000',
    ':': '00000/00110/00110/00000/00110/00110/00000',
    "'": '00100/00100/01000/00000/00000/00000/00000',
    '/': '00001/00010/00010/00100/01000/01000/10000',
    '(': '00010/00100/01000/01000/01000/00100/00010',
    ')': '01000/00100/00010/00010/00010/00100/01000',
    '!': '00100/00100/00100/00100/00100/00000/00100',
    '?': '01110/10001/00001/00010/00100/00000/00100',
    ' ': '000/000/000/000/000/000/000',
}


def normalized_inscription(text: str) -> str:
    """Return exactly the printable label used in the exported mesh."""
    text = ' '.join(str(text).split())[:40]
    text = ''.join(c for c in unicodedata.normalize('NFKD', text)
                   if not unicodedata.combining(c)).upper()
    return ''.join(c if c in _FONT_ROWS else '?' for c in text)


def _setting(settings, name, default):
    return settings.get(name, default) if isinstance(settings, Mapping) else getattr(settings, name, default)


def _glyph_rectangles(text):
    """Merge equal horizontal runs in successive rows to limit boolean work."""
    offset = 0
    rectangles = []
    for character in text:
        rows = _FONT_ROWS[character].split('/')
        active = {}
        for row_index, row in enumerate([*rows, '0' * len(rows[0])]):
            runs = []
            start = None
            for col, filled in enumerate(row + '0'):
                if filled == '1' and start is None:
                    start = col
                elif filled == '0' and start is not None:
                    runs.append((start, col))
                    start = None
            for run in list(active):
                if run not in runs:
                    start_row = active.pop(run)
                    rectangles.append((offset + run[0], offset + run[1], start_row, row_index))
            for run in runs:
                if run not in active:
                    active[run] = row_index
        offset += len(rows[0]) + 1
    return rectangles, max(0, offset - 1)


def _closed_union(meshes):
    """Boolean union followed by the same float32 welding STL will perform."""
    from manifold3d import Manifold, Mesh, OpType
    solids = [m if isinstance(m, Manifold) else
              Manifold(Mesh(vert_properties=np.asarray(m.vertices, dtype=np.float32),
                            tri_verts=np.asarray(m.faces, dtype=np.uint32))) for m in meshes]
    result = Manifold.batch_boolean(solids, OpType.Add)
    raw = result.set_tolerance(1e-4).simplify(1e-4).to_mesh64()
    mesh = trimesh.Trimesh(vertices=np.asarray(raw.vert_properties, dtype=np.float32),
                           faces=raw.tri_verts)
    mesh.update_faces(mesh.area_faces > 0)
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight or mesh.volume <= 0:
        raise ValueError('The shared pedestal could not be created as a closed printable solid.')
    return mesh


def pedestal_mesh(meshes: Iterable[trimesh.Trimesh], settings) -> trimesh.Trimesh | None:
    """Create only the shared base; caller unions it with the world-space figures.

    Figure meshes must already be raised by ``thickness_mm - .5`` so their
    lowest supports penetrate the pedestal. The base spans Z=0..thickness;
    optional lettering projects 0.7 mm above its top with 0.15 mm overlap.
    Both round and rectangular footprints encompass all figure XY bounds plus
    the requested margin. A label may extend the footprint toward the front.
    """
    shape = _setting(settings, 'shape', 'none')
    figures = list(meshes)
    if shape == 'none' or not figures:
        return None
    if shape not in ('round', 'rectangle'):
        raise ValueError('Shared pedestal shape must be none, round, or rectangle.')
    margin = float(_setting(settings, 'margin_mm', 8))
    thickness = float(_setting(settings, 'thickness_mm', 3))
    requested_size = float(_setting(settings, 'text_size_mm', 4))
    text = normalized_inscription(_setting(settings, 'inscription', ''))
    bounds = np.asarray([m.bounds for m in figures])
    low = np.min(bounds[:, 0, :2], axis=0) - margin
    high = np.max(bounds[:, 1, :2], axis=0) + margin
    rectangles, text_columns = _glyph_rectangles(text)
    text_size = requested_size
    if text_columns:
        # Keep at least 2 mm clear at each end of the inscription.
        text_size = min(requested_size, (high[0] - low[0] - 4) * 7 / text_columns)
        low[1] = min(low[1], float(np.min(bounds[:, 0, 1])) - text_size - 4)
    center = (low + high) / 2
    size = high - low
    if shape == 'round':
        radius = float(np.linalg.norm(size / 2))
        # Circumscribing the rectangle also places the complete label safely
        # inside the top face, away from its circular boundary.
        base = trimesh.creation.cylinder(radius=radius, height=thickness, sections=96)
    else:
        base = trimesh.creation.box(extents=[*size, thickness])
    base.apply_translation([*center, thickness / 2])
    primitives = [base]
    if rectangles:
        unit = text_size / 7
        origin_x = center[0] - text_columns * unit / 2
        baseline_y = low[1] + 2
        # Adjacent strokes overlap by 4% of a pixel. The relief overlaps the
        # base itself, including isolated punctuation such as a period.
        overlap = unit * .04
        from manifold3d import CrossSection, OpType
        strokes = []
        for x0, x1, row0, row1 in rectangles:
            stroke = CrossSection.square([(x1 - x0) * unit + overlap,
                                          (row1 - row0) * unit + overlap])
            strokes.append(stroke.translate([origin_x + x0 * unit - overlap / 2,
                                             baseline_y + (7 - row1) * unit - overlap / 2]))
        # Resolve coplanar stroke overlaps in 2D before extrusion. This avoids
        # microscopic intersection triangles when many narrow letters meet.
        lettering = CrossSection.batch_boolean(strokes, OpType.Add).simplify(1e-5)
        primitives.append(lettering.extrude(.85).translate([0, 0, thickness - .15]))
    result = _closed_union(primitives)
    result.metadata.update(inscription=text,
                           inscription_height_mm=round(text_size, 3) if text else 0,
                           pedestal_thickness_mm=thickness,
                           label_note='Raised uppercase bitmap text; accented Latin letters are simplified and unsupported characters become ?.')
    return result
