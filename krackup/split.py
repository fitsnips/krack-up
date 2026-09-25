"""Cut a solid until every piece fits on the bed.

Cuts stay away from the tips of a part. Inside that band the narrowest
cross-section wins, which is where a limb joint usually is.
"""

import numpy as np

from krackup.geom import apply_frame, extents, fits, pca, unit

MIN_CUT = 36.0


class Piece:
    def __init__(self, solid, cuts, stall=0):
        self.solid = solid
        self.cuts = list(cuts)
        self.stall = stall


def split_to_fit(solid, limit, log=print):
    accepted = []
    cuts = []
    queue = [Piece(solid, [])]
    steps = 0
    while queue:
        steps += 1
        if steps > 2000:
            raise RuntimeError("split did not finish")
        piece = queue.pop()
        volume = piece.solid.volume()
        if volume < 200.0:
            continue
        if _fits_any(piece.solid, limit, [cuts[i]["normal"] for i in piece.cuts]):
            accepted.append(piece)
            size = extents(piece.solid.bounding_box())
            log(
                f"keep  {size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} mm"
                f"  {volume / 1000:.0f} cm3"
            )
            continue
        if piece.stall >= 6:
            size = extents(piece.solid.bounding_box())
            raise RuntimeError(f"could not fit a piece on the bed: {size}")
        direction, offset, reason = (
            _forced_plane(piece.solid, piece.stall - 1)
            if piece.stall
            else _choose_plane(piece.solid, limit)
        )
        direction = unit(direction)
        positive, negative = piece.solid.split_by_plane(direction.tolist(), float(offset))
        if positive.volume() < 1.0 or negative.volume() < 1.0:
            direction, offset, reason = _forced_plane(piece.solid, 3)
            direction = unit(direction)
            positive, negative = piece.solid.split_by_plane(direction.tolist(), float(offset))
        cut_id = len(cuts)
        cuts.append(
            {
                "id": cut_id,
                "normal": direction.tolist(),
                "offset": float(offset),
                "reason": reason,
            }
        )
        log(f"cut {cut_id}: {reason}")
        parent_size = float(np.max(extents(piece.solid.bounding_box())))
        for side in (positive, negative):
            for component in side.decompose():
                child_size = float(np.max(extents(component.bounding_box())))
                stall = 0 if child_size < parent_size - 20.0 else piece.stall + 1
                queue.append(Piece(component, piece.cuts + [cut_id], stall))
    return accepted, cuts


def _fits_any(solid, limit, normals):
    if fits(extents(solid.bounding_box()), limit):
        return True
    for _, frame in _orientations(normals):
        if fits(extents(apply_frame(solid, frame).bounding_box()), limit):
            return True
    return False


def _orientations(normals):
    found = [("upright", np.eye(3))]
    seen = [np.eye(3)]

    def add(name, frame):
        frame = np.asarray(frame, dtype=np.float64)
        if np.linalg.det(frame) < 0:
            frame = frame.copy()
            frame[0] *= -1
        for previous in seen:
            if np.allclose(previous, frame, atol=1e-5):
                return
        seen.append(frame)
        found.append((name, frame))

    for index, axis in enumerate(
        (np.array([1.0, 0, 0]), np.array([0.0, 1, 0]), np.array([0.0, 0, 1]))
    ):
        for sign, tag in ((1.0, "+"), (-1.0, "-")):
            add(f"world{('XYZ')[index]}{tag}", _map_up(sign * axis))
    for index, normal in enumerate(normals):
        add(f"cut{index} down", _map_up(-np.asarray(normal, dtype=np.float64)))
        add(f"cut{index} up", _map_up(np.asarray(normal, dtype=np.float64)))
    return found


def _map_up(direction):
    from krackup.geom import rot_from_to

    return rot_from_to(direction, [0, 0, 1])


def _sample(solid, count=40000):
    mesh = solid.to_mesh()
    verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    if len(verts) > count:
        pick = np.random.default_rng(0).choice(len(verts), count, replace=False)
        verts = verts[pick]
    return verts


def _choose_plane(solid, limit):
    points = _sample(solid)
    best = None
    for direction, axis_limit, name in _axes(points, limit):
        found = _position(points @ unit(direction), axis_limit)
        if found is None:
            continue
        score, position, narrow = found
        if name == "worldZ":
            score *= 0.92
        if best is None or score < best[0]:
            best = (score, unit(direction), position, name, narrow)
    if best is None:
        return _forced_plane(solid, 3)
    _, direction, position, name, narrow = best
    return direction, position, f"{name} narrow={narrow:.2f}"


def _axes(points, limit):
    world = points.max(axis=0) - points.min(axis=0)
    axes = []
    for index, name in enumerate(("worldX", "worldY", "worldZ")):
        cap = limit[2] if index == 2 else limit[0]
        if world[index] > cap + 0.5:
            direction = np.zeros(3)
            direction[index] = 1.0
            axes.append((direction, cap, name))
    _, pca_axes, pca_ext = pca(points)
    over = pca_ext > (limit[0] + 0.5)
    for index in range(3):
        too_tall = pca_ext[index] > limit[2] + 0.5
        blocks = pca_ext[index] > limit[0] + 0.5 and int(np.count_nonzero(over)) >= 2
        if too_tall or blocks or (index == 0 and pca_ext[0] > limit[0] + 0.5):
            cap = limit[2] if index == 0 else limit[0]
            axes.append((pca_axes[:, index], cap, f"pca{index}"))
    return axes


def _position(projection, limit):
    low = float(projection.min())
    high = float(projection.max())
    span = high - low
    if span <= limit + 0.5:
        return None
    windows = [(low + 0.28 * span, low + 0.72 * span)]
    if span > limit + 90:
        windows.append((low + limit - 50, low + limit + 30))
        windows.append((high - limit - 30, high - limit + 50))
    positions = []
    for start, end in windows:
        start = max(start, low + MIN_CUT)
        end = min(end, high - MIN_CUT)
        if end > start + 1:
            positions.extend(np.arange(start, end + 1e-6, 12.0))
    if not positions:
        positions = [low + 0.5 * span]
    positions = np.unique(np.round(positions, 3))
    counts = np.array(
        [np.count_nonzero(np.abs(projection - pos) <= 8.0) for pos in positions],
        dtype=np.float64,
    )
    counts = np.maximum(counts, 1.0)
    baseline = float(np.median(counts))
    ideal = min(limit, span / 2.0)
    best = None
    for pos, count in zip(positions, counts):
        smaller = min(pos - low, high - pos)
        if smaller < MIN_CUT:
            continue
        score = (0.45 + abs(smaller - ideal) / ideal) * (count / baseline)
        if best is None or score < best[0]:
            best = (score, float(pos), float(count / baseline))
    if best is None:
        return (1.0, float(low + 0.5 * span), 1.0)
    return best


def _forced_plane(solid, attempt):
    points = _sample(solid, 25000)
    _, pca_axes, pca_ext = pca(points)
    world = points.max(axis=0) - points.min(axis=0)
    choices = []
    for index in range(3):
        choices.append((float(pca_ext[index]), unit(pca_axes[:, index]), f"pca{index}"))
    for index, name in enumerate("XYZ"):
        direction = np.zeros(3)
        direction[index] = 1.0
        choices.append((float(world[index]), direction, f"world{name}"))
    choices.sort(key=lambda item: item[0], reverse=True)
    direction, name = choices[min(attempt, len(choices) - 1)][1:]
    projection = points @ direction
    position = float(0.5 * (projection.min() + projection.max()))
    return unit(direction), position, f"forced {name} midpoint"


def orientations_for(normals):
    return _orientations(normals)
