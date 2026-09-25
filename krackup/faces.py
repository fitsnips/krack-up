"""Add dowels to flat cut faces that were already in the file.

krack-up only used to pin cuts it made itself. A project like the Jimmy
file is already split, and some of those faces have no dowel, or only one.
Each cut face gets the same spacing rule as a new cut, and at least
`min_pins` dowels. A matching face on the other part gets the same holes.
"""

import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.validation import make_valid

from krackup.geom import plane_frame, unit
from krackup.pins import RADII, WALL, _carve, _distance, _grid, _section_shape

MIN_FACE = 700.0
SHELL_GAP = 22.0


def pin_existing_faces(pieces, length, tolerance, pitch, min_pins, log=print):
    faces = []
    for piece in pieces:
        faces.extend(_cut_faces(piece))
    pairs, singles = _match(faces)
    pins = []
    for face_a, face_b in pairs:
        pins.extend(_fill_pair(face_a, face_b, length, tolerance, pitch, min_pins, log))
    for face in singles:
        pins.extend(_fill_pair(face, None, length, tolerance, pitch, min_pins, log))
    if pins:
        _carve(pieces, pins)
    return pins


def _cut_faces(piece):
    mesh = piece.solid.to_mesh()
    verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    if len(faces) == 0:
        return []
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normals, axis=1)
    area = 0.5 * length
    good = length > 1e-8
    unit_n = np.zeros_like(normals)
    unit_n[good] = normals[good] / length[good, None]
    centers = tri.mean(axis=1)
    offset = np.sum(unit_n * centers, axis=1)
    groups = {}
    for index in np.flatnonzero(good & (area > 0.4)):
        normal = unit_n[index]
        key = (
            int(np.round(normal[0] * 20)),
            int(np.round(normal[1] * 20)),
            int(np.round(normal[2] * 20)),
            int(np.round(offset[index])),
        )
        bucket = groups.setdefault(key, [])
        bucket.append(index)
    patches = []
    for indexes in groups.values():
        indexes = np.asarray(indexes)
        total = float(area[indexes].sum())
        if total < 30:
            continue
        normal = unit(unit_n[indexes].T @ area[indexes])
        plane = float(np.average(offset[indexes], weights=area[indexes]))
        centroid = np.average(centers[indexes], axis=0, weights=area[indexes])
        patches.append(
            {
                "area": total,
                "normal": normal,
                "offset": plane,
                "centroid": centroid,
            }
        )
    cuts = []
    for patch in patches:
        if patch["area"] < MIN_FACE:
            continue
        if _thin_wall(patch, patches):
            continue
        cuts.append(
            {
                "piece": piece,
                "area": patch["area"],
                "normal": patch["normal"],
                "offset": patch["offset"],
                "existing": _existing_pins(patch, patches),
            }
        )
    return cuts


def _thin_wall(patch, patches):
    for other in patches:
        if other is patch or other["area"] < MIN_FACE:
            continue
        if float(np.dot(patch["normal"], other["normal"])) > -0.9:
            continue
        gap = abs(patch["offset"] + other["offset"])
        ratio = min(patch["area"], other["area"]) / max(patch["area"], other["area"])
        if gap < SHELL_GAP and ratio > 0.8:
            return True
    return False


def _existing_pins(patch, patches):
    found = []
    frame = plane_frame(patch["normal"])
    for other in patches:
        if other is patch or not (40 <= other["area"] <= 500):
            continue
        if float(np.dot(patch["normal"], other["normal"])) < 0.9:
            continue
        inward = float(np.dot(patch["centroid"] - other["centroid"], patch["normal"]))
        if not 2.5 <= inward <= 12:
            continue
        local = frame @ other["centroid"]
        found.append((float(local[0]), float(local[1])))
    return found


def _match(faces):
    scored = []
    for i, face_a in enumerate(faces):
        shape_a = _shape(face_a)
        if shape_a is None:
            continue
        for face_b in faces[i + 1 :]:
            if face_a["piece"] is face_b["piece"]:
                continue
            if abs(face_a["area"] - face_b["area"]) / max(face_a["area"], face_b["area"]) > 0.2:
                continue
            shape_b = _shape(face_b)
            if shape_b is None:
                continue
            score, signs = _best_iou(shape_a, shape_b)
            if score >= 0.55:
                scored.append((score, face_a, face_b, signs, shape_a, shape_b))
    scored.sort(key=lambda item: item[0], reverse=True)
    used = set()
    pairs = []
    for _, face_a, face_b, signs, shape_a, shape_b in scored:
        if id(face_a) in used or id(face_b) in used:
            continue
        used.add(id(face_a))
        used.add(id(face_b))
        face_a["shape"] = shape_a
        face_b["shape"] = shape_b
        face_b["signs"] = signs
        pairs.append((face_a, face_b))
    singles = [face for face in faces if id(face) not in used]
    for face in singles:
        face["shape"] = _shape(face)
    return pairs, singles


def _shape(face):
    polygon = _section_shape(face["piece"].solid, plane_frame(face["normal"]), face["offset"] - 0.4)
    if polygon is None or polygon.is_empty or polygon.area < MIN_FACE * 0.5:
        return None
    coords = np.asarray(polygon.convex_hull.exterior.coords[:-1], dtype=np.float64)
    center = coords.mean(axis=0)
    shifted = coords - center
    _, vectors = np.linalg.eigh(shifted.T @ shifted)
    axes = vectors[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, 1] *= -1
    return {"polygon": polygon, "center": center, "axes": axes, "frame": plane_frame(face["normal"])}


def _best_iou(shape_a, shape_b):
    local_a = _local_polygon(shape_a)
    best = (0.0, (1.0, 1.0))
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            local_b = _local_polygon(shape_b, (sx, sy))
            score = _iou(local_a, local_b)
            if score > best[0]:
                best = (score, (sx, sy))
    return best


def _local_polygon(shape, signs=(1.0, 1.0)):
    from shapely.geometry import Polygon

    coords = np.asarray(shape["polygon"].convex_hull.exterior.coords[:-1], dtype=np.float64)
    local = (coords - shape["center"]) @ shape["axes"]
    local = local * np.array(signs)
    polygon = Polygon(local)
    return polygon if polygon.is_valid else make_valid(polygon)


def _iou(one, two):
    union = one.union(two).area
    if union <= 0:
        return 0.0
    return float(one.intersection(two).area / union)


def _fill_pair(face_a, face_b, length, tolerance, pitch, min_pins, log):
    if face_a.get("shape") is None:
        return []
    depth = length / 2 + tolerance
    positions = _layout(face_a, depth, tolerance, pitch, min_pins)
    if len(positions) < min_pins and len(face_a["existing"]) < min_pins:
        log(f"  cut face {face_a['area']:.0f} mm2 could only take {len(positions)} new dowels")
    fresh = [pos for pos in positions if _far(pos, face_a["existing"])]
    pins = [
        _socket_pin(face_a, x, y, radius, length, tolerance, depth, counts=True)
        for x, y, radius in fresh
    ]
    if face_b is not None and face_b.get("shape") is not None:
        mapped = []
        for x, y, radius in fresh:
            other = _map_point(x, y, face_a["shape"], face_b["shape"], face_b["signs"])
            if _far(other, face_b["existing"]):
                mapped.append((other[0], other[1], radius))
        pins.extend(
            _socket_pin(face_b, x, y, radius, length, tolerance, depth, counts=False)
            for x, y, radius in mapped
        )
        log(
            f"  matched faces {face_a['area']:.0f} mm2, "
            f"{len(fresh)} new dowels on each side"
        )
    elif fresh:
        log(f"  bare face {face_a['area']:.0f} mm2, {len(fresh)} new dowels")
    return [pin for pin in pins if pin is not None]


def _layout(face, depth, tolerance, pitch, min_pins):
    shape = face["shape"]["polygon"]
    existing = list(face["existing"])
    for radius in RADII:
        region = shape.buffer(-(radius + tolerance + WALL), join_style=2)
        if region.is_empty:
            continue
        points = list(_grid(region, pitch))
        _top_up(points, region, radius, tolerance, max(min_pins, len(points)))
        kept = [point for point in points if _holds(face, point, radius, tolerance, depth)]
        # Existing dowels count toward the minimum.
        total = len(existing) + len([point for point in kept if _far(point, existing)])
        if total >= min_pins or radius == RADII[-1]:
            return [(x, y, radius) for x, y in kept if _far((x, y), existing)]
    return []


def _top_up(points, region, radius, tolerance, target):
    separation = 2.0 * (radius + tolerance) + 1.0
    if len(points) >= target:
        return
    minx, miny, maxx, maxy = region.bounds
    step = max(4.0, separation / 2)
    candidates = []
    x = minx
    while x <= maxx:
        y = miny
        while y <= maxy:
            if region.contains(Point(float(x), float(y))):
                candidates.append((float(x), float(y)))
            y += step
        x += step
    while len(points) < target:
        best = None
        best_dist = -1.0
        for x, y in candidates:
            dist = min((_distance(x, y, px, py) for px, py in points), default=1e9)
            if dist < separation or dist <= best_dist:
                continue
            best = (x, y)
            best_dist = dist
        if best is None:
            break
        points.append(best)


def _holds(face, point, radius, tolerance, depth):
    shape = _section_shape(
        face["piece"].solid,
        face["shape"]["frame"],
        face["offset"] - depth,
    )
    if shape is None:
        return False
    inner = shape.buffer(-(radius + tolerance), join_style=2)
    return (not inner.is_empty) and inner.contains(Point(point[0], point[1]))


def _far(point, others, gap=8.0):
    return all(_distance(point[0], point[1], other[0], other[1]) >= gap for other in others)


def _map_point(x, y, shape_a, shape_b, signs):
    local = (np.array([x, y]) - shape_a["center"]) @ shape_a["axes"]
    local = local * np.array(signs)
    mapped = shape_b["center"] + shape_b["axes"] @ local
    return float(mapped[0]), float(mapped[1])


def _socket_pin(face, x, y, radius, length, tolerance, depth, counts=True):
    # The face normal points out of the part, so the hole goes the other way.
    # _carve treats side -1 as "into the negative half", which is this direction
    # when `positive` is a dummy and `negative` is the real piece.
    return {
        "x": float(x),
        "y": float(y),
        "radius": radius,
        "length": length,
        "tolerance": tolerance,
        "depth": depth,
        "cut_id": -1,
        "frame": face["shape"]["frame"],
        "normal": face["normal"],
        "offset": face["offset"],
        "positive": _Dummy(),
        "negative": face["piece"],
        "counts": counts,
    }


class _Dummy:
    pass
