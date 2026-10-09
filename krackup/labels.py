"""Engrave each part's number into one of its joint faces.

The number goes on a mating face, so it is hidden once the model is glued
up. Digits are seven-segment strokes with a bar under them, which keeps 6
and 9 apart. Text reads the right way round seen from outside the face.
"""

import numpy as np
from shapely.affinity import rotate, scale, translate
from shapely.geometry import box
from shapely.ops import polylabel, unary_union

import manifold3d as mf

from krackup.geom import apply_frame, plane_frame, unit
from krackup.pins import _cut, _polygons, _section_shape

DEPTH = 0.6
MARGIN = 2.0
HEIGHTS = (10.0, 8.0, 6.0, 5.0)

SEGMENTS = {
    "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg",
    "5": "acdfg", "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg",
}


def label_parts(records, log=print):
    """Engrave rec['id'] into rec['piece'].solid. Returns ids left blank."""
    missed = []
    for rec in records:
        piece = rec["piece"]
        faces = _faces(piece, rec.get("holes", []), rec.get("cuts", []))
        if not faces:
            continue
        if not _engrave(piece, str(rec["id"]), faces):
            missed.append(rec["id"])
    if missed:
        log(f"no room for a label on parts {missed}")
    return missed


def text_outline(text, height):
    """Rectangles for `text`, lower left at the origin, as one shape."""
    stroke = max(0.15 * height, 1.0)
    width = 0.6 * height
    gap = 0.25 * height
    rects = []
    x = 0.0
    for char in text:
        for seg in SEGMENTS[char]:
            rects.append(_segment(seg, x, width, height, stroke))
        x += width + gap
    # The bar marks the bottom, so an upside-down 6 is not a 9.
    rects.append(box(0.0, -2.0 * stroke, x - gap, -stroke))
    return unary_union(rects)


def _segment(seg, x, w, h, t):
    half = h / 2.0
    return {
        "a": box(x, h - t, x + w, h),
        "b": box(x + w - t, half, x + w, h),
        "c": box(x + w - t, 0.0, x + w, half),
        "d": box(x, 0.0, x + w, t),
        "e": box(x, 0.0, x + t, half),
        "f": box(x, half, x + t, h),
        "g": box(x, half - t / 2.0, x + w, half + t / 2.0),
    }[seg]


def _faces(piece, holes, cuts):
    """Flat joint faces of this piece as (frame, offset, side).

    `side` is +1 when the piece lies on the +normal side of the plane.
    """
    found = []

    def add(frame, offset, side):
        for other, other_offset, other_side in found:
            if (
                other_side * float(np.dot(other[2], frame[2])) * side > 0.999
                and abs(other_offset * other_side - offset * side) < 0.3
            ):
                return
        found.append((np.asarray(frame, dtype=np.float64), float(offset), float(side)))

    for hole in holes:
        add(hole["frame"], hole["offset"], hole["side"])
    bounds = piece.solid.bounding_box()
    center = np.array([(bounds[i] + bounds[i + 3]) / 2 for i in range(3)])
    for cut in cuts:
        normal = unit(cut["normal"])
        side = 1.0 if float(np.dot(center, normal)) >= cut["offset"] else -1.0
        add(plane_frame(normal), cut["offset"], side)
    return found


def _engrave(piece, text, faces):
    regions = []
    for frame, offset, side in faces:
        region = _room(piece.solid, frame, offset, side)
        if region is not None:
            regions.append((region.area, region, frame, offset, side))
    regions.sort(key=lambda item: item[0], reverse=True)
    for height in HEIGHTS:
        outline = text_outline(text, height)
        for _, region, frame, offset, side in regions:
            placed = _place(outline, region, side)
            if placed is None:
                continue
            carved = _cut(piece.solid, _tool(placed, frame, offset, side))
            if carved is not None:
                piece.solid = carved
                return True
    return False


def _room(solid, frame, offset, side):
    """Where on the face a label can go: solid under it, nothing above it."""
    shallow = _section_shape(solid, frame, offset + side * 0.3)
    deep = _section_shape(solid, frame, offset + side * (DEPTH + 0.6))
    if shallow is None or deep is None:
        return None
    region = shallow.intersection(deep).buffer(-MARGIN, join_style=2)
    above = _section_shape(solid, frame, offset - side * 0.3)
    if above is not None:
        region = region.difference(above.buffer(MARGIN))
    if region.is_empty or region.area < 20.0:
        return None
    return region


def _place(outline, region, side):
    if side > 0:
        # Seen from outside this face, frame x runs right to left.
        outline = scale(outline, xfact=-1.0, origin=(0, 0))
    for polygon in sorted(_polygons(region), key=lambda p: p.area, reverse=True):
        spot = polylabel(polygon, tolerance=0.5)
        for angle in (0.0, 90.0):
            turned = rotate(outline, angle, origin="centroid")
            middle = turned.centroid
            moved = translate(turned, spot.x - middle.x, spot.y - middle.y)
            if polygon.contains(moved):
                return moved
    return None


def _tool(shape, frame, offset, side):
    contours = []
    for polygon in _polygons(shape):
        ring = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
        if _area(ring) < 0:
            ring = ring[::-1]
        contours.append(ring)
        for interior in polygon.interiors:
            hole = np.asarray(interior.coords[:-1], dtype=np.float64)
            if _area(hole) > 0:
                hole = hole[::-1]
            contours.append(hole)
    height = DEPTH + 0.2
    tool = mf.CrossSection(contours).extrude(height)
    bottom = offset - 0.2 if side > 0 else offset - DEPTH
    tool = tool.translate((0.0, 0.0, bottom))
    return apply_frame(tool, np.asarray(frame).T)


def _area(ring):
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))
