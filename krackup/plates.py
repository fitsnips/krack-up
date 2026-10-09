"""Lay parts out on as few plates as will hold them.

Footprints are bounding rectangles packed with MaxRects (best short side
fit), so a part may be turned a quarter turn about Z. Bambu Studio puts
plate n of a project at a grid position, and `plate_origin` follows it.
"""

import math

GAP = 6.0
# Bambu keeps plates a fifth of a bed apart.
PLATE_GAP = 0.2


def pack(sizes, bed, margin, exclude=(), gap=GAP):
    """Place (width, depth) footprints on beds of size `bed`.

    Returns one (plate, x, y, turned) per size, where (x, y) is the
    footprint center on that plate's bed and `turned` means a quarter turn.
    """
    width = bed[0] - 2 * margin + gap
    depth = bed[1] - 2 * margin + gap
    order = sorted(range(len(sizes)), key=lambda i: (-max(sizes[i]), -sizes[i][0] * sizes[i][1]))
    bins = []
    placed = [None] * len(sizes)
    for index in order:
        w, d = sizes[index][0] + gap, sizes[index][1] + gap
        spot = None
        for plate, free in enumerate(bins):
            spot = free.find(w, d)
            if spot is not None:
                break
        if spot is None:
            free = _Bin(width, depth)
            for x0, y0, x1, y1 in exclude:
                free.block(x0 - margin, y0 - margin, x1 - x0 + gap, y1 - y0 + gap)
            spot = free.find(w, d)
            if spot is None:
                # Too big to clear the excluded corner by its box; its own
                # plate, centered, where the real outline usually misses it.
                free = _Bin(width, depth)
                spot = free.find(w, d)
                if spot is None:
                    raise ValueError(
                        f"a {sizes[index][0]:.0f} x {sizes[index][1]:.0f} mm part does not fit the bed"
                    )
                _, _, rw, rd, turned = spot
                spot = ((width - rw) / 2, (depth - rd) / 2, rw, rd, turned)
                free.free = []
            bins.append(free)
            plate = len(bins) - 1
        x, y, rw, rd, turned = spot
        free.block(x, y, rw, rd)
        placed[index] = (plate, margin + x + (rw - gap) / 2, margin + y + (rd - gap) / 2, turned)
    return placed


def plate_origin(plate, count, bed):
    """Where Bambu Studio puts plate `plate` (0-based) of `count`."""
    columns = max(1, math.ceil(math.sqrt(count) - 1e-9))
    row, column = divmod(plate, columns)
    return column * bed[0] * (1 + PLATE_GAP), -row * bed[1] * (1 + PLATE_GAP)


class _Bin:
    def __init__(self, width, depth):
        self.free = [(0.0, 0.0, width, depth)]

    def find(self, w, d):
        best = None
        for fx, fy, fw, fd in self.free:
            for rw, rd, turned in ((w, d, False), (d, w, True)):
                if rw <= fw + 1e-6 and rd <= fd + 1e-6:
                    score = (min(fw - rw, fd - rd), max(fw - rw, fd - rd))
                    if best is None or score < best[0]:
                        best = (score, (fx, fy, rw, rd, turned))
        return None if best is None else best[1]

    def block(self, x, y, w, d):
        pieces = []
        for rect in self.free:
            pieces.extend(_split(rect, (x, y, w, d)))
        pieces = list(dict.fromkeys(pieces))
        self.free = [
            rect for rect in pieces
            if not any(other != rect and _inside(rect, other) for other in pieces)
        ]


def _split(rect, used):
    fx, fy, fw, fd = rect
    ux, uy, uw, ud = used
    if ux >= fx + fw or ux + uw <= fx or uy >= fy + fd or uy + ud <= fy:
        return [rect]
    out = []
    if ux > fx:
        out.append((fx, fy, ux - fx, fd))
    if ux + uw < fx + fw:
        out.append((ux + uw, fy, fx + fw - ux - uw, fd))
    if uy > fy:
        out.append((fx, fy, fw, uy - fy))
    if uy + ud < fy + fd:
        out.append((fx, uy + ud, fw, fy + fd - uy - ud))
    return [r for r in out if r[2] > 1e-6 and r[3] > 1e-6]


def _inside(rect, other):
    return (
        rect[0] >= other[0] - 1e-9
        and rect[1] >= other[1] - 1e-9
        and rect[0] + rect[2] <= other[0] + other[2] + 1e-9
        and rect[1] + rect[3] <= other[1] + other[3] + 1e-9
    )

