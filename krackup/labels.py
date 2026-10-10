"""Engrave every joint face with its part number and the part it meets.

Part 3's face that meets part 5 reads 3-5, and the face on part 5 reads
5-3, so any face says what it is and where it goes. The text is on mating
faces, so it is hidden once the model is glued up. Digits are seven-segment
strokes with a bar under them, which keeps 6 and 9 apart, and they read the
right way round seen from outside the face. A face too small for both
numbers gets the part's own number, and one too small for that gets none.
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
# Contact smaller than this is a touch, not a joint.
MIN_CONTACT = 80.0

SEGMENTS = {
    "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg",
    "5": "acdfg", "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg",
    "-": "g",
}


def cut_faces(pieces, cuts):
    """Joint faces on krack-up's own cuts, one per pair of parts that touch.

    The label region is where the two parts actually meet, so a face that
    meets two parts gets a label for each.
    """
    found = []
    for cut in cuts:
        normal = unit(cut["normal"])
        offset = float(cut["offset"])
        frame = plane_frame(normal)
        sides = {1.0: [], -1.0: []}
        for piece in pieces:
            if cut["id"] not in piece.cuts:
                continue
            bounds = piece.solid.bounding_box()
            center = np.array([(bounds[i] + bounds[i + 3]) / 2 for i in range(3)])
            side = 1.0 if float(np.dot(center, normal)) >= offset else -1.0
            section = _section_shape(piece.solid, frame, offset + side * 0.6)
            if section is not None:
                sides[side].append((piece, section))
        for above, top in sides[1.0]:
            for below, bottom in sides[-1.0]:
                contact = top.intersection(bottom)
                if contact.is_empty or contact.area < MIN_CONTACT:
                    continue
                found.append(_face(above, below, frame, offset, 1.0, contact))
                found.append(_face(below, above, frame, offset, -1.0, contact))
    return found


def file_faces(pairs):
    """Joint faces matched between objects that came cut in the file.

    Each face's frame normal points out of its part, so the part is on -1.
    """
    found = []
    for face_a, face_b in pairs:
        for face, mate in ((face_a, face_b), (face_b, face_a)):
            found.append(_face(face["piece"], mate["piece"], face["shape"]["frame"], face["offset"], -1.0))
    return found


def _face(piece, mate, frame, offset, side, region=None):
    return {
        "piece": piece,
        "mate": mate,
        "frame": np.asarray(frame, dtype=np.float64),
        "offset": float(offset),
        "side": float(side),
        "region": region,
    }


def label_joints(faces, ids, log=print):
    """Engrave each face. `ids` maps id(piece) to its part number.

    Returns (labeled, short, missing): counts of full labels, and the
    (part, mate) pairs that got only the part's number or nothing.
    """
    rooms = [_room(face["piece"].solid, face["frame"], face["offset"], face["side"]) for face in faces]
    labeled = 0
    short = []
    missing = []
    for face, room in zip(faces, rooms):
        own = ids[id(face["piece"])]
        mate = ids[id(face["mate"])]
        if room is not None and face["region"] is not None:
            room = room.intersection(face["region"].buffer(-MARGIN, join_style=2))
        text = _engrave(face, room, (f"{own}-{mate}", str(own)))
        if text == f"{own}-{mate}":
            labeled += 1
        elif text:
            short.append([own, mate])
        else:
            missing.append([own, mate])
    if short:
        log(f"only the part number fit on {len(short)} joint faces")
    if missing:
        log(f"no room for a label on {len(missing)} joint faces")
    return labeled, short, missing


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


def _engrave(face, room, texts):
    """Cut the first of `texts` that fits. Returns it, or None."""
    if room is None or room.is_empty:
        return None
    piece = face["piece"]
    for text in texts:
        for height in HEIGHTS:
            placed = _place(text_outline(text, height), room, face["side"])
            if placed is None:
                continue
            carved = _cut(piece.solid, _tool(placed, face["frame"], face["offset"], face["side"]))
            if carved is not None:
                piece.solid = carved
                return text
    return None


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
