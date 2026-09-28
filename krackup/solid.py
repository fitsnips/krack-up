"""Turn triangle soup into a Manifold solid and back."""

import manifold3d as mf
import numpy as np


class KrackError(Exception):
    """A problem with the input or the cut that the user can act on."""


def manifold_from(verts, faces):
    verts, faces = _weld(verts, faces)
    verts, faces = _fill_small_holes(verts, faces)
    mesh = mf.Mesh(
        vert_properties=np.ascontiguousarray(verts, dtype=np.float32),
        tri_verts=np.ascontiguousarray(faces, dtype=np.uint32),
    )
    solid = mf.Manifold(mesh)
    status = str(solid.status())
    if "NoError" not in status:
        raise KrackError(f"mesh is not a solid ({status})")
    return solid


def mesh_arrays(solid):
    mesh = solid.to_mesh()
    verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    return verts, faces


def _weld(verts, faces):
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    _, index, inverse = np.unique(np.round(verts, 5), axis=0, return_index=True, return_inverse=True)
    welded = verts[index]
    faces = inverse.reshape(-1)[faces]
    keep = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 2] != faces[:, 0])
    )
    return welded, faces[keep]


def _fill_small_holes(verts, faces):
    """Close gaps of a few edges. A missing triangle makes Manifold reject the part."""
    faces = np.asarray(faces, dtype=np.int64)
    directed = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    undirected = np.sort(directed, axis=1)
    _, counts = np.unique(undirected, axis=0, return_counts=True)
    if not np.any(counts == 1):
        return verts, faces
    # Keep the direction used by the existing face. The patch uses the opposite.
    packed = directed[:, 0].astype(np.int64) << 32 | directed[:, 1].astype(np.int64)
    reverse = directed[:, 1].astype(np.int64) << 32 | directed[:, 0].astype(np.int64)
    boundary = set(packed.tolist()) - set(reverse.tolist())
    if not boundary:
        return verts, faces
    nxt = {}
    for key in boundary:
        nxt[int(key >> 32)] = int(key & 0xFFFFFFFF)
    seen = set()
    extra = []
    for start in list(nxt):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        cursor = nxt[start]
        while cursor != start and cursor not in seen and cursor in nxt:
            loop.append(cursor)
            seen.add(cursor)
            cursor = nxt[cursor]
        if cursor != start or len(loop) < 3 or len(loop) > 12:
            continue
        loop = loop[::-1]
        for i in range(1, len(loop) - 1):
            extra.append((loop[0], loop[i], loop[i + 1]))
    if not extra:
        return verts, faces
    return verts, np.vstack([faces, np.asarray(extra, dtype=np.int64)])
