"""Rounded, printable garments and accessories for the original studio figures.

These are solid additions to the sculpt, not textures.  Every detail deliberately
overlaps its support so the caller can fuse it into the body before exporting.
The local frame follows the torso; limbs and held objects follow the pose joints.
"""
from __future__ import annotations

import math
import numpy as np
import trimesh


def _value(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def accessory_meshes(figure, j, head, up, lateral, forward):
    """Return closed garment/accessory solids in canonical figure coordinates.

    ``up``, ``lateral`` and ``forward`` are the body's orthonormal frame.  The
    caller performs the final union with the implicit body, scales and grounding.
    A modest amount of relief makes the simplified details survive an STL export.
    """
    appearance = _value(figure, 'appearance', {})
    outfit = _value(appearance, 'outfit', 'none')
    accessories = set(_value(appearance, 'accessories', []) or [])
    head = np.asarray(head, float)
    up, lateral, forward = [np.asarray(v, float) for v in (up, lateral, forward)]
    frame = np.column_stack((lateral, forward, up))
    head_up = np.array(j.get('head_up', up), dtype=float, copy=True)
    if np.linalg.norm(head_up) < 1e-8:
        head_up = up.copy()
    head_up /= max(float(np.linalg.norm(head_up)), 1e-8)
    head_lateral = np.array(j.get('head_lateral', lateral), dtype=float, copy=True)
    if head_lateral @ lateral < 0:
        head_lateral = -head_lateral
    head_lateral -= head_up * (head_lateral @ head_up)
    if np.linalg.norm(head_lateral) < 1e-8:
        head_lateral = lateral - head_up * (lateral @ head_up)
    if np.linalg.norm(head_lateral) < 1e-8:
        fallback = np.eye(3)[int(np.argmin(np.abs(head_up)))]
        head_lateral = fallback - head_up * (fallback @ head_up)
    head_lateral /= max(float(np.linalg.norm(head_lateral)), 1e-8)
    head_forward = np.cross(head_lateral, head_up)
    head_frame = np.column_stack((head_lateral, head_forward, head_up))
    hip = (j['hip_l'] + j['hip_r']) / 2
    shoulder = (j['shoulder_l'] + j['shoulder_r']) / 2
    meshes = []

    def ellipsoid(center, radii, subdivisions=2, axes=frame):
        mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
        mesh.vertices = (mesh.vertices * np.asarray(radii)) @ axes.T + center
        # The canonical left/right frame may be reflected by photo orientation.
        if mesh.volume < 0:
            mesh.invert()
        meshes.append(mesh)
        return mesh

    def capsule(a, b, radius):
        a, b = np.asarray(a, float), np.asarray(b, float)
        distance = np.linalg.norm(b - a)
        if distance < 1e-6:
            return ellipsoid(a, [radius] * 3)
        mesh = trimesh.creation.capsule(height=distance, radius=radius, count=[12, 16])
        transform = trimesh.geometry.align_vectors([0, 0, 1], (b - a) / distance)
        mesh.apply_transform(transform)
        # Trimesh capsules are centered at their end-point midpoint.
        mesh.apply_translation((a + b) / 2)
        meshes.append(mesh)
        return mesh

    def loft(rings, sections=48, axes=frame):
        """Closed elliptical loft, including a solid top and bottom."""
        angles = np.arange(sections) * (2 * math.pi / sections)
        vertices = []
        for center, rx, ry in rings:
            vertices.extend(np.asarray(center) + np.cos(angles)[:, None] * rx * axes[:, 0]
                            + np.sin(angles)[:, None] * ry * axes[:, 1])
        faces = []
        for level in range(len(rings) - 1):
            for n in range(sections):
                a = level * sections + n
                b = level * sections + (n + 1) % sections
                faces.extend(((a, b, b + sections), (a, b + sections, a + sections)))
        for level, reverse in ((0, True), (len(rings) - 1, False)):
            center_index = len(vertices)
            vertices.append(np.asarray(rings[level][0]))
            for n in range(sections):
                a, b = level * sections + n, level * sections + (n + 1) % sections
                faces.append((center_index, b, a) if reverse else (center_index, a, b))
        mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=faces)
        if mesh.volume < 0:
            mesh.invert()
        meshes.append(mesh)
        return mesh

    if outfit in ('casual', 'suit'):
        # A rounded sweater/coat follows the body rather than replacing its pose.
        ellipsoid(hip + up * 13, [12.7, 10.1, 17], subdivisions=3)
        for side in ('l', 'r'):
            # Sleeves attach at the shoulder and taper toward the existing arm.
            capsule(j['shoulder_' + side], j['elbow_' + side], 4.25)
        if outfit == 'casual':
            ellipsoid(shoulder - up * 1.6 + forward * 6.4, [6, 2.7, 2.4])
        else:
            for sign in (-1, 1):
                capsule(shoulder + lateral * sign * 5 - up * 2 + forward * 8.6,
                        hip + up * 11 + lateral * sign * 1.2 + forward * 10, 1.65)
            # A bow tie replaces the standard necktie rather than stacking both.
            if 'bowtie' not in accessories:
                capsule(shoulder - up * 5 + forward * 9.4,
                        hip + up * 12 + forward * 10.5, 1.65)
                ellipsoid(shoulder - up * 4.5 + forward * 9, [2.2, 2.4, 2.4])

    if outfit == 'wedding-dress':
        ellipsoid(hip + up * 13, [12.5, 9.9, 16], subdivisions=3)
        knee = (j['knee_l'] + j['knee_r']) / 2
        ankle = (j['ankle_l'] + j['ankle_r']) / 2
        waist = hip + up * 7
        # Knee/ankle centers carry seated and bent-leg gestures into the skirt.
        # Keep rings ordered along the torso to avoid a folded/self-crossing loft.
        drop = max(float((waist - ankle) @ up), 12)
        knee_drop = float(np.clip((waist - knee) @ up, drop * .25, drop * .75))
        middle = knee + up * (float((waist - knee) @ up) - knee_drop)
        hem = ankle - up * 2
        hem += up * (float((waist - hem) @ up) - drop)
        loft([(hem, 22, 17), (middle, 17, 13), (waist, 11.5, 9)])
        # Shallow waist band gives the bodice and skirt a legible transition.
        ellipsoid(waist, [12.5, 10.2, 2.3])

    head_size = float(_value(figure, 'head_size', 1))
    if 'hat' in accessories:
        style = _value(appearance, 'hat_style', 'top-hat')
        if style == 'cap':
            ellipsoid(head + head_up * 13 * head_size,
                      np.array([17.2, 16, 9.2]) * head_size, subdivisions=3, axes=head_frame)
            # A solid, short forward visor joins the crown and the forehead.
            ellipsoid(head + head_up * 11 * head_size + head_forward * 11 * head_size,
                      np.array([15.5, 12, 2.5]) * head_size, axes=head_frame)
        else:
            radius = (25 if style == 'wide-brim' else 21) * head_size
            brim_center = head + head_up * 11.5 * head_size
            loft([(brim_center - head_up * 1.7 * head_size, radius, radius * .88),
                  (brim_center + head_up * 1.7 * head_size, radius, radius * .88)], axes=head_frame)
            height = (10 if style == 'wide-brim' else 17) * head_size
            loft([(brim_center, 15 * head_size, 13.8 * head_size),
                  (brim_center + head_up * height, 12.8 * head_size, 12 * head_size)], axes=head_frame)

    if 'veil' in accessories:
        # A thick soft drape anchors well inside the back of the scalp. It remains
        # one solid sheet, with no lace fragments or a separate unsupported shell.
        ellipsoid(head - head_forward * 12.5 * head_size - head_up * 11,
                  [17 * head_size, 4.3, 23], subdivisions=3, axes=head_frame)

    if 'bowtie' in accessories:
        center = shoulder - up * 3.5 + forward * 8.2
        for sign in (-1, 1):
            ellipsoid(center + lateral * sign * 3, [3.8, 2.4, 2.6])
        ellipsoid(center + forward * .6, [2, 2.6, 2.4])

    if 'bouquet' in accessories:
        hand = _value(appearance, 'bouquet_hand', 'both')
        if hand == 'left':
            wrists = [j['wrist_l']]
        elif hand == 'right':
            wrists = [j['wrist_r']]
        else:
            wrists = [j['wrist_l'], j['wrist_r']]
        anchor = np.mean(wrists, axis=0)
        # A two-hand bouquet uses both actual hands as supports, even when their
        # estimated positions are apart. This retains the pose and avoids islands.
        # A photo or neutral pose can place the wrists against the belly. Keep
        # the flower heads in front of the torso instead of burying the entire
        # bouquet inside it; thick stems still connect to the unchanged hands.
        torso_front = float(((hip + shoulder) / 2) @ forward) + 14
        offset = max(3, torso_front - float(anchor @ forward))
        flower_center = anchor + up * 8 + forward * offset
        stem_top = flower_center - up * 2
        for wrist in wrists:
            capsule(wrist, stem_top, 2.5)
        ellipsoid(flower_center, [6.2, 5.6, 5.3])
        for dx, dy, dz in [(-4.3, 0, 1), (4.3, 0, 1), (0, 3.4, 2),
                           (-2.2, -2.8, 3), (2.2, -2.8, 3)]:
            ellipsoid(flower_center + lateral * dx + forward * dy + up * dz,
                      [3.5, 3.5, 3.3])

    return meshes
