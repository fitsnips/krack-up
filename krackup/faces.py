"""Add dowels to flat cut faces that were already in the file.

A project like the Jimmy file is already split into objects, and some of
those joints have no dowel, or only one. Two faces count as a joint only when
they sit on different objects of the file and have the same outline. When
Bambu's cut_information.xml says two objects are halves of one cut, their
faces are paired before any other. Faces
that krack-up cut itself are left alone because `add_dowels` pinned them, and
so are outside walls with no partner.

The objects in the file are laid out on the plate, not assembled, so the two
faces of a joint are matched by outline. Seen from outside each part, the two
faces are mirror images, so the map between them is a reflection. When the
outline is symmetric, more than one reflection fits; the dowels are then laid
out with the same symmetry so the holes line up whichever one is right.
"""

import numpy as np
from shapely import affinity
from shapely.geometry import Point

from krackup.geom import plane_frame, unit
from krackup.pins import RADII, WALL, _carve, _distance, _grid, _section_shape, socket
from krackup.solid import mesh_arrays

MIN_FACE = 700.0
SHELL_GAP = 22.0
MATCH_IOU = 0.8
# A face this far from flat is a curved surface, not a cut.
FLAT = 0.3
# Outlines within this IoU of the best fit are treated as equally good.
TIE = 0.015
# More equally good fits than this means the face is round.
ROUND = 8


def pin_existing_faces(pieces, length, tolerance, pitch, min_pins, log=print, scale=1.0):
    """`scale` is how much the model was scaled, so connectors in the file
    (sized for the unscaled model) are still recognised."""
    faces = []
    for piece in pieces:
        faces.extend(_cut_faces(piece, scale))
    pairs = _match(faces)
    pins = []
    for number, (face_a, face_b, fit) in enumerate(pairs):
        pins.extend(
            _fill_pair(face_a, face_b, fit, f"file{number}", length, tolerance, pitch, min_pins, log)
        )
    if pins:
        failed = _carve(pieces, pins, log)
        pins = [pin for pin in pins if id(pin) not in failed]
    return pins, [(face_a, face_b) for face_a, face_b, _ in pairs]


def _cut_faces(piece, scale=1.0):
    verts, faces = mesh_arrays(piece.solid)
    if len(faces) == 0:
        return []
    patches = _flat_patches(verts, faces)
    planes = getattr(piece, "planes", [])
    merged = _merge_coplanar([p for p in patches if not _on_plane(p, planes)])
    cuts = []
    for patch in merged:
        if patch["area"] < MIN_FACE:
            continue
        if _thin_wall(patch, merged):
            continue
        cuts.append(
            {
                "piece": piece,
                "area": patch["area"],
                "normal": patch["normal"],
                "offset": patch["offset"],
                "existing": _existing_pins(patch, patches, scale),
            }
        )
    return cuts


def _flat_patches(verts, faces):
    """Connected groups of coplanar triangles with at least 30 mm2 of area."""
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normals, axis=1)
    area = 0.5 * length
    good = length > 1e-8
    unit_n = np.zeros_like(normals)
    unit_n[good] = normals[good] / length[good, None]
    centers = tri.mean(axis=1)
    offset = np.sum(unit_n * centers, axis=1)

    count = len(faces)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    owner = np.tile(np.arange(count), 3)
    edges = np.sort(edges, axis=1)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    owner = owner[order]
    shared = np.all(edges[1:] == edges[:-1], axis=1)
    first = owner[:-1][shared]
    second = owner[1:][shared]
    coplanar = (
        good[first]
        & good[second]
        & (np.sum(unit_n[first] * unit_n[second], axis=1) > 0.9998)
        & (np.abs(np.sum(unit_n[first] * centers[second], axis=1) - offset[first]) < 0.05)
    )
    labels = _components(count, first[coplanar], second[coplanar])

    labels = np.where(good, labels, -1)
    ids, inverse = np.unique(labels, return_inverse=True)
    totals = np.bincount(inverse, weights=area)
    patches = []
    for index in np.flatnonzero((totals >= 30) & (ids >= 0)):
        members = np.flatnonzero(inverse == index)
        weights = area[members]
        normal = unit(unit_n[members].T @ weights)
        plane = float(np.average(centers[members] @ normal, weights=weights))
        corners = tri[members].reshape(-1, 3) @ normal
        if float(np.max(np.abs(corners - plane))) > FLAT:
            continue
        patches.append(
            {
                "area": float(weights.sum()),
                "normal": normal,
                "offset": plane,
                "centroid": np.average(centers[members], axis=0, weights=weights),
            }
        )
    return patches


def _components(count, first, second):
    labels = np.arange(count)
    if len(first) == 0:
        return labels
    while True:
        low = np.minimum(labels[first], labels[second])
        new = labels.copy()
        np.minimum.at(new, first, low)
        np.minimum.at(new, second, low)
        new = new[new]
        if np.array_equal(new, labels):
            return labels
        labels = new


def _merge_coplanar(patches):
    """One cut can leave several islands on one plane. Treat them as one face."""
    merged = []
    for patch in sorted(patches, key=lambda item: item["area"], reverse=True):
        for group in merged:
            if (
                float(np.dot(group["normal"], patch["normal"])) > 0.9995
                and abs(group["offset"] - patch["offset"]) < 0.3
            ):
                total = group["area"] + patch["area"]
                group["centroid"] = (
                    group["centroid"] * group["area"] + patch["centroid"] * patch["area"]
                ) / total
                group["area"] = total
                break
        else:
            merged.append(dict(patch))
    return merged


def _on_plane(patch, planes):
    """True for a face krack-up cut itself; add_dowels already pinned those."""
    for normal, offset in planes:
        normal = np.asarray(normal, dtype=np.float64)
        dot = float(np.dot(patch["normal"], normal))
        if abs(dot) > 0.999 and abs(patch["offset"] - np.sign(dot) * offset) < 0.5:
            return True
    return False


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


def _existing_pins(patch, patches, scale=1.0):
    """Sockets (a pocket floor) and pegs (a flat top) already on this face."""
    found = []
    frame = plane_frame(patch["normal"])
    for other in patches:
        if other is patch or not (40 * scale**2 <= other["area"] <= 500 * scale**2):
            continue
        if float(np.dot(patch["normal"], other["normal"])) < 0.9:
            continue
        step = abs(float(np.dot(patch["centroid"] - other["centroid"], patch["normal"])))
        if not 2.5 * scale <= step <= 12 * scale:
            continue
        local = frame @ other["centroid"]
        found.append((float(local[0]), float(local[1])))
    return found


def _match(faces):
    scored = []
    shapes = {}

    def shape(face):
        if id(face) not in shapes:
            shapes[id(face)] = _shape(face)
        return shapes[id(face)]

    for i, face_a in enumerate(faces):
        for face_b in faces[i + 1 :]:
            if face_a["piece"] is face_b["piece"]:
                continue
            # A joint is between two objects of the file. Pieces krack-up cut
            # from one object were joined by add_dowels.
            if getattr(face_a["piece"], "origin", None) == getattr(face_b["piece"], "origin", None):
                continue
            if abs(face_a["area"] - face_b["area"]) / max(face_a["area"], face_b["area"]) > 0.2:
                continue
            shape_a = shape(face_a)
            shape_b = shape(face_b)
            if shape_a is None or shape_b is None:
                continue
            fit = _align(shape_a, shape_b)
            if fit["score"] >= MATCH_IOU:
                scored.append((_same_cut(face_a, face_b), fit["score"], face_a, face_b, fit))
    # The two halves of one Bambu cut take their faces first, so a look-alike
    # face elsewhere cannot claim them.
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    used = set()
    pairs = []
    for _, _, face_a, face_b, fit in scored:
        if id(face_a) in used or id(face_b) in used:
            continue
        used.add(id(face_a))
        used.add(id(face_b))
        face_a["shape"] = shapes[id(face_a)]
        face_b["shape"] = shapes[id(face_b)]
        pairs.append((face_a, face_b, fit))
    return pairs


def _same_cut(face_a, face_b):
    cut_a = getattr(face_a["piece"], "cut_id", 0)
    return bool(cut_a) and cut_a == getattr(face_b["piece"], "cut_id", 0)


def _shape(face):
    frame = plane_frame(face["normal"])
    polygon = _section_shape(face["piece"].solid, frame, face["offset"] - 0.4)
    if polygon is None or polygon.is_empty or polygon.area < MIN_FACE * 0.5:
        return None
    center = polygon.centroid
    local = affinity.translate(polygon, -center.x, -center.y).simplify(0.3)
    return {
        "polygon": polygon,
        "local": local,
        "center": np.array([center.x, center.y]),
        "frame": frame,
    }


def _reflection(theta):
    """Reflection across the line at theta / 2. Its own inverse."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [s, -c]])


def _iou_at(local_a, local_b, theta):
    m = _reflection(theta)
    moved = affinity.affine_transform(local_b, [m[0, 0], m[0, 1], m[1, 0], m[1, 1], 0.0, 0.0])
    union = local_a.union(moved).area
    if union <= 0:
        return 0.0
    return float(local_a.intersection(moved).area / union)


def _align(shape_a, shape_b):
    """Find the reflections that lay face B's outline on face A's."""
    local_a, local_b = shape_a["local"], shape_b["local"]
    steps = np.radians(np.arange(0.0, 360.0, 3.0))
    scores = np.array([_iou_at(local_a, local_b, t) for t in steps])
    best = float(scores.max())
    if best < MATCH_IOU:
        return {"score": best}
    if float(scores.min()) >= best - TIE:
        return {"score": best, "theta": 0.0, "ties": [0.0], "round": True}
    peaks = []
    count = len(steps)
    for i in range(count):
        if scores[i] < best - 0.05:
            continue
        if scores[i] >= scores[i - 1] and scores[i] >= scores[(i + 1) % count]:
            peaks.append(_refine(local_a, local_b, float(steps[i]), float(scores[i])))
    peaks.sort(key=lambda item: item[1], reverse=True)
    top = peaks[0][1]
    ties = []
    for theta, score in peaks:
        if score < top - TIE:
            continue
        if all(_angle_gap(theta, other) > np.radians(4.0) for other in ties):
            ties.append(theta)
    return {
        "score": top,
        "theta": peaks[0][0],
        "ties": ties,
        "round": len(ties) > ROUND,
    }


def _refine(local_a, local_b, theta, score):
    step = np.radians(1.5)
    while step > np.radians(0.05):
        moved = False
        for candidate in (theta - step, theta + step):
            value = _iou_at(local_a, local_b, candidate)
            if value > score:
                theta, score, moved = candidate, value, True
        if not moved:
            step /= 2.0
    return theta, score


def _angle_gap(one, two):
    gap = (one - two) % (2 * np.pi)
    return min(gap, 2 * np.pi - gap)


def _map_point(point, shape_a, shape_b, theta):
    local = np.asarray(point, dtype=np.float64) - shape_a["center"]
    mapped = shape_b["center"] + _reflection(theta) @ local
    return float(mapped[0]), float(mapped[1])


def _symmetry(fit):
    """Rotations about face A's center that the dowel layout must keep."""
    rotations = []
    for theta in fit["ties"]:
        angle = fit["theta"] - theta
        c, s = np.cos(angle), np.sin(angle)
        rotations.append(np.array([[c, -s], [s, c]]))
    return rotations or [np.eye(2)]


def _fill_pair(face_a, face_b, fit, joint, length, tolerance, pitch, min_pins, log):
    depth = length / 2 + tolerance
    existing = face_a["existing"]
    if fit["round"]:
        positions = _center_only(face_a, face_b, fit, depth, tolerance)
        if positions and min_pins > 1:
            log(
                f"  round face {face_a['area']:.0f} mm2: one centered dowel, because "
                "the turn between the two parts cannot be read from a round outline"
            )
    else:
        positions = _layout(face_a, face_b, fit, depth, tolerance, pitch, min_pins)
    if len(positions) + len(existing) < min_pins:
        log(
            f"  cut face {face_a['area']:.0f} mm2 has {len(existing)} dowels "
            f"and could only take {len(positions)} more"
        )
    if not positions:
        return []
    pins = []
    for x, y, radius in positions:
        bx, by = _map_point((x, y), face_a["shape"], face_b["shape"], fit["theta"])
        pins.append(
            {
                "x": x,
                "y": y,
                "radius": radius,
                "length": length,
                "tolerance": tolerance,
                "depth": depth,
                "cut_id": joint,
                "area": face_a["area"],
                "positive": face_a["piece"],
                "negative": face_b["piece"],
                # Each face normal points out of its part, so both holes go
                # against it.
                "holes": [
                    socket(face_a["piece"], face_a["shape"]["frame"], x, y, face_a["offset"], -1.0),
                    socket(face_b["piece"], face_b["shape"]["frame"], bx, by, face_b["offset"], -1.0),
                ],
            }
        )
    log(
        f"  matched faces {face_a['area']:.0f} mm2 on "
        f"{face_a['piece'].source} and {face_b['piece'].source}, {len(pins)} new dowels"
    )
    return pins


def _center_only(face_a, face_b, fit, depth, tolerance):
    center = tuple(face_a["shape"]["center"])
    for radius in RADII:
        if _fits_both(face_a, face_b, fit, center, radius, depth, tolerance, []):
            return [(center[0], center[1], radius)]
    return []


def _layout(face_a, face_b, fit, depth, tolerance, pitch, min_pins):
    shape = face_a["shape"]["polygon"]
    existing = list(face_a["existing"])
    rotations = _symmetry(fit)
    center = face_a["shape"]["center"]
    for radius in RADII:
        region = shape.buffer(-(radius + tolerance + WALL), join_style=2)
        if region.is_empty:
            continue
        grid = _grid(region, pitch)
        # One dowel per `pitch`, at least `min_pins`, and the ones already
        # on the face count.
        need = max(min_pins, len(grid)) - len(existing)
        if need <= 0:
            return []
        separation = 2.0 * (radius + tolerance) + 1.0
        chosen = []

        def take(point):
            orbit = _orbit(point, rotations, center)
            if all(
                _fits_both(
                    face_a, face_b, fit, p, radius, depth, tolerance,
                    existing + chosen + orbit[:i], separation,
                )
                for i, p in enumerate(orbit)
            ):
                chosen.extend(orbit)
                return True
            return False

        for point in grid:
            if len(chosen) >= need:
                break
            take(point)
        candidates = _candidates(region, max(4.0, separation / 2.0))
        while len(chosen) < need and candidates:
            taken = existing + chosen
            best = max(
                candidates,
                key=lambda p: min((_distance(p[0], p[1], q[0], q[1]) for q in taken), default=1e9),
            )
            candidates.remove(best)
            take(best)
        if len(chosen) + len(existing) >= min_pins or radius == RADII[-1]:
            return [(x, y, radius) for x, y in chosen]
    return []


def _orbit(point, rotations, center):
    orbit = []
    for rotation in rotations:
        moved = center + rotation @ (np.asarray(point, dtype=np.float64) - center)
        moved = (float(moved[0]), float(moved[1]))
        # The center of a symmetric face maps to itself.
        if all(_distance(moved[0], moved[1], p[0], p[1]) > 1.0 for p in orbit):
            orbit.append(moved)
    return orbit


def _fits_both(face_a, face_b, fit, point, radius, depth, tolerance, taken, separation=0.0):
    if any(_distance(point[0], point[1], q[0], q[1]) < separation for q in taken):
        return False
    if not _holds(face_a, point, radius, tolerance, depth):
        return False
    other = _map_point(point, face_a["shape"], face_b["shape"], fit["theta"])
    if any(_distance(other[0], other[1], q[0], q[1]) < separation for q in face_b["existing"]):
        return False
    return _holds(face_b, other, radius, tolerance, depth)


def _candidates(region, step):
    minx, miny, maxx, maxy = region.bounds
    found = []
    for x in np.arange(minx, maxx + 1e-6, step):
        for y in np.arange(miny, maxy + 1e-6, step):
            if region.contains(Point(float(x), float(y))):
                found.append((float(x), float(y)))
    return found


def _holds(face, point, radius, tolerance, depth):
    """The hole wall stays inside the part all the way to the hole floor."""
    cache = face.setdefault("floor", {})
    key = (radius, round(depth, 4))
    if key not in cache:
        section = _section_shape(face["piece"].solid, face["shape"]["frame"], face["offset"] - depth)
        inner = None if section is None else section.buffer(-(radius + tolerance), join_style=2)
        cache[key] = None if inner is None or inner.is_empty else inner
    inner = cache[key]
    return inner is not None and inner.contains(Point(point[0], point[1]))
