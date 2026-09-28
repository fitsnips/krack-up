"""Pentagon dowels, sized the way the Jimmy P1S project was cut by hand.

That file uses Bambu cut connectors with height 10 mm and a radial
clearance of 0.1 mm. The printed dowels are pentagonal prisms. The three
sizes in the project are circumradius 5, 7.5, and 10 mm. A joint gets the
largest of those that fits, repeated every `pitch` millimetres.
"""

import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.validation import make_valid

import manifold3d as mf

from krackup.geom import apply_frame, plane_frame, unit

RADII = (10.0, 7.5, 5.0)
WALL = 3.5


def add_dowels(pieces, cuts, length, tolerance, pitch, min_pins=2, log=print):
    planned = []
    areas = {}
    for cut in cuts:
        pins, area = _plan_cut(pieces, cut, length, tolerance, pitch, min_pins)
        areas[cut["id"]] = area
        planned.extend(pins)
        if area >= 80:
            log(f"joint {cut['id']}: {area:.0f} mm2, {len(pins)} dowels")
    failed = _carve(pieces, planned, log)
    planned = [pin for pin in planned if id(pin) not in failed]
    return planned, areas


def dowel_solid(radius, length):
    """Pentagon prism. Circumradius and length are millimeters."""
    section = mf.CrossSection([_pentagon(radius)])
    solid = section.extrude(length).translate((0, 0, -length / 2))
    return solid


def _plan_cut(pieces, cut, length, tolerance, pitch, min_pins):
    normal = unit(np.array(cut["normal"], dtype=np.float64))
    offset = float(cut["offset"])
    frame = plane_frame(normal)
    positive, negative = [], []
    for piece in pieces:
        if cut["id"] not in piece.cuts:
            continue
        bounds = piece.solid.bounding_box()
        center = np.array(
            [
                (bounds[0] + bounds[3]) / 2,
                (bounds[1] + bounds[4]) / 2,
                (bounds[2] + bounds[5]) / 2,
            ]
        )
        side = 1.0 if float(np.dot(center, normal) - offset) >= 0 else -1.0
        (positive if side > 0 else negative).append(piece)
    if not positive or not negative:
        return [], 0.0
    pos_shape = _union(_sections(positive, frame, offset + 0.6))
    neg_shape = _union(_sections(negative, frame, offset - 0.6))
    if pos_shape is None or neg_shape is None:
        return [], 0.0
    mating = pos_shape.intersection(neg_shape)
    if mating.is_empty or mating.area < 80:
        return [], 0.0
    depth = length / 2 + tolerance
    parts = [part for part in _polygons(mating) if part.area >= 80]
    best = []
    for radius in RADII:
        pins = _pins_at_radius(
            parts, radius, pitch, min_pins, length, tolerance, depth,
            positive, negative, frame, offset, cut["id"], normal,
        )
        if len(pins) >= min_pins:
            return pins, float(mating.area)
        if len(pins) > len(best):
            best = pins
    return best, float(mating.area)


def _pins_at_radius(parts, radius, pitch, min_pins, length, tolerance, depth,
                    positive, negative, frame, offset, cut_id, normal):
    pins = []
    regions = []
    for part in parts:
        region = part.buffer(-(radius + tolerance + WALL), join_style=2)
        if region.is_empty or region.area < 1.0:
            continue
        regions.append(region)
        for x, y in _grid(region, pitch):
            pin = _make_pin(
                x, y, radius, length, tolerance, depth,
                positive, negative, frame, offset, cut_id, normal,
            )
            if pin is not None:
                pins.append(pin)
    if len(pins) >= min_pins or not regions:
        return pins
    separation = 2.0 * (radius + tolerance) + 1.0
    candidates = _candidates(regions, max(4.0, separation / 2.0))
    while len(pins) < min_pins:
        taken = [(pin["x"], pin["y"]) for pin in pins]
        best = None
        best_dist = -1.0
        for x, y in candidates:
            dist = min((_distance(x, y, px, py) for px, py in taken), default=1e9)
            if dist + 1e-6 < separation or dist <= best_dist:
                continue
            best = (x, y)
            best_dist = dist
        if best is None:
            break
        pin = _make_pin(
            best[0], best[1], radius, length, tolerance, depth,
            positive, negative, frame, offset, cut_id, normal,
        )
        if pin is None:
            candidates = [point for point in candidates if point != best]
            continue
        pins.append(pin)
    return pins


def _make_pin(x, y, radius, length, tolerance, depth, positive, negative, frame, offset, cut_id, normal):
    if not _both_sides_hold(positive, negative, frame, offset, depth, radius, tolerance, x, y):
        return None
    owner_pos = _owner(positive, frame, offset + depth, x, y, radius)
    owner_neg = _owner(negative, frame, offset - depth, x, y, radius)
    if owner_pos is None or owner_neg is None:
        return None
    return {
        "x": x,
        "y": y,
        "radius": radius,
        "length": length,
        "tolerance": tolerance,
        "depth": depth,
        "cut_id": cut_id,
        "frame": frame,
        "normal": normal,
        "offset": offset,
        "positive": owner_pos,
        "negative": owner_neg,
        "holes": [
            socket(owner_pos, frame, x, y, offset, 1.0),
            socket(owner_neg, frame, x, y, offset, -1.0),
        ],
    }


def socket(piece, frame, x, y, offset, side):
    """One hole. (x, y) are in `frame`, on the plane z' = offset.

    The hole runs along side * frame[2] from that plane into the piece.
    """
    return {
        "piece": piece,
        "frame": frame,
        "x": float(x),
        "y": float(y),
        "offset": float(offset),
        "side": float(side),
    }


def _candidates(regions, step):
    found = []
    for region in regions:
        minx, miny, maxx, maxy = region.bounds
        x = minx
        while x <= maxx:
            y = miny
            while y <= maxy:
                if region.contains(Point(float(x), float(y))):
                    found.append((float(x), float(y)))
                y += step
            x += step
    return found


def _distance(x1, y1, x2, y2):
    return float(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5)


def _both_sides_hold(positive, negative, frame, offset, depth, radius, tolerance, x, y):
    return (
        _owner(positive, frame, offset + depth, x, y, radius + tolerance) is not None
        and _owner(negative, frame, offset - depth, x, y, radius + tolerance) is not None
    )


def _owner(group, frame, height, x, y, radius):
    point = Point(x, y)
    for piece in group:
        shape = _section_shape(piece.solid, frame, height)
        if shape is None:
            continue
        inner = shape.buffer(-radius, join_style=2)
        if not inner.is_empty and inner.contains(point):
            return piece
    return None


def _carve(pieces, pins, log=print):
    """Cut every pin's holes. Returns ids of pins that lost a hole.

    A dowel is only useful when both of its holes exist, so the caller drops
    those pins from the print count.
    """
    jobs = {}
    for pin in pins:
        for hole in pin["holes"]:
            jobs.setdefault(id(hole["piece"]), []).append((pin, hole))
    failed = set()
    for piece in pieces:
        group = jobs.get(id(piece))
        if not group:
            continue
        tools = [_hole(pin, hole) for pin, hole in group]
        tool = tools[0]
        for extra in tools[1:]:
            tool = tool + extra
        carved = _cut(piece.solid, tool)
        if carved is not None:
            piece.solid = carved
            continue
        # One bad hole should not cost the part all of its sockets.
        for (pin, _), single in zip(group, tools):
            carved = _cut(piece.solid, single)
            if carved is None:
                failed.add(id(pin))
            else:
                piece.solid = carved
    if failed:
        log(
            f"warning: {len(failed)} dowel holes could not be cut; those dowels are "
            "left out and their other socket stays empty"
        )
    return failed


def _cut(solid, tool):
    carved = solid - tool
    if "NoError" in str(carved.status()) and carved.volume() > 0:
        return carved
    return None


def _hole(pin, hole):
    frame = hole["frame"]
    x_axis, y_axis, normal = frame[0], frame[1], frame[2]
    origin = frame.T @ np.array([hole["x"], hole["y"], hole["offset"]])
    into = hole["side"] * normal
    depth = pin["depth"]
    length = depth + 0.3
    center = origin + into * (depth / 2.0 - 0.15)
    matrix = np.column_stack([x_axis, y_axis, into, center])
    tool = dowel_solid(pin["radius"] + pin["tolerance"], length)
    # dowel_solid is centered on Z. Move that axis onto `into`.
    return tool.transform(matrix)


def _pentagon(radius):
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 2
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def _sections(pieces, frame, height):
    return [_section_shape(piece.solid, frame, height) for piece in pieces]


def _section_shape(solid, frame, height):
    section = apply_frame(solid, frame).slice(float(height))
    return _to_shape(section)


def _to_shape(section):
    if section is None or section.is_empty() or section.area() < 1.0:
        return None
    shells = []
    holes = []
    for raw in section.to_polygons():
        polygon = np.asarray(raw, dtype=np.float64)
        if len(polygon) < 3:
            continue
        area = _signed_area(polygon)
        if abs(area) < 1.0:
            continue
        (shells if area > 0 else holes).append(polygon)
    geoms = []
    used = set()
    for shell in shells:
        shape = make_valid(__polygon(shell))
        kept = []
        for index, hole in enumerate(holes):
            if index in used:
                continue
            hole_shape = __polygon(hole)
            if shape.contains(hole_shape.representative_point()):
                kept.append(hole)
                used.add(index)
        geoms.append(make_valid(__polygon(shell, kept)))
    if not geoms:
        return None
    merged = unary_union(geoms)
    return None if merged.is_empty else merged


def __polygon(shell, holes=()):
    from shapely.geometry import Polygon

    return Polygon(shell, holes=list(holes) or None)


def _signed_area(polygon):
    return 0.5 * float(
        np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1))
    )


def _union(shapes):
    shapes = [shape for shape in shapes if shape is not None and not shape.is_empty]
    if not shapes:
        return None
    return unary_union(shapes)


def _polygons(shape):
    if shape.geom_type == "Polygon":
        return [shape]
    if shape.geom_type == "MultiPolygon":
        return list(shape.geoms)
    return [geom for geom in getattr(shape, "geoms", []) if geom.geom_type == "Polygon"]


def _grid(region, pitch):
    minx, miny, maxx, maxy = region.bounds
    best = []
    for ox in np.linspace(0, pitch, 6, endpoint=False):
        xs = np.arange(np.floor((minx - ox) / pitch) * pitch + ox, maxx + pitch, pitch)
        for oy in np.linspace(0, pitch, 6, endpoint=False):
            ys = np.arange(np.floor((miny - oy) / pitch) * pitch + oy, maxy + pitch, pitch)
            found = [
                (float(x), float(y))
                for x in xs
                for y in ys
                if region.contains(Point(float(x), float(y)))
            ]
            if len(found) > len(best):
                best = found
    if best:
        return best
    point = region.representative_point()
    if region.contains(point):
        return [(float(point.x), float(point.y))]
    return []
