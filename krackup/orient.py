"""Stand each piece in the pose that leaves the least overhang."""

import numpy as np

from krackup.geom import fits
from krackup.split import orientations_for


def place(solid, normals, limit):
    mesh = solid.to_mesh()
    verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    best = None
    for name, frame in orientations_for(normals):
        rotated = verts @ frame.T
        size = rotated.max(axis=0) - rotated.min(axis=0)
        if not fits(size, limit):
            continue
        placed = rotated.copy()
        low = placed.min(axis=0)
        high = placed.max(axis=0)
        placed[:, 0] -= 0.5 * (low[0] + high[0])
        placed[:, 1] -= 0.5 * (low[1] + high[1])
        placed[:, 2] -= low[2]
        overhang, contact = _support(placed, faces)
        rank = (overhang, -contact, size[2])
        if best is None or rank < best[0]:
            best = (rank, name, frame, size, overhang, contact, placed, faces)
    if best is None:
        return None
    _, name, frame, size, overhang, contact, placed, faces = best
    return {
        "name": name,
        "frame": frame,
        "size": size,
        "overhang": overhang,
        "contact": contact,
        "verts": placed,
        "faces": faces,
    }


def _support(verts, faces):
    triangles = verts[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = 0.5 * np.linalg.norm(normals, axis=1)
    scale = np.maximum(np.linalg.norm(normals, axis=1), 1e-12)
    unit_normals = normals / scale[:, None]
    angle = np.degrees(
        np.arctan2(np.hypot(unit_normals[:, 0], unit_normals[:, 1]), np.abs(unit_normals[:, 2]))
    )
    # The face sitting on the bed is not support.
    height = triangles[:, :, 2].mean(axis=1)
    overhang = (unit_normals[:, 2] < 0) & (angle < 30.0) & (height > 0.6)
    contact = (unit_normals[:, 2] < -0.97) & (height < 0.8)
    return float(area[overhang].sum()), float(area[contact].sum())
