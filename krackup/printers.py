"""Bed sizes. Usable volume leaves room for a 5 mm brim and the P1S corner."""

from krackup.solid import KrackError

PRINTERS = {
    # The corner at the origin is the nozzle wiper, from the Jimmy project settings.
    "p1s": {"name": "Bambu Lab P1S", "bed": (256.0, 256.0, 250.0), "exclude": [(0.0, 0.0, 18.0, 28.0)]},
    "x1c": {"name": "Bambu Lab X1C", "bed": (256.0, 256.0, 256.0), "exclude": [(0.0, 0.0, 18.0, 28.0)]},
    "a1": {"name": "Bambu Lab A1", "bed": (256.0, 256.0, 256.0)},
    "a1mini": {"name": "Bambu Lab A1 mini", "bed": (180.0, 180.0, 180.0)},
    "mini": {"name": "Prusa MINI", "bed": (180.0, 180.0, 180.0)},
    "mk4": {"name": "Prusa MK4", "bed": (250.0, 210.0, 220.0)},
    "ender3": {"name": "Creality Ender-3", "bed": (220.0, 220.0, 250.0)},
    "k1": {"name": "Creality K1", "bed": (220.0, 220.0, 250.0)},
}

# Jimmy's cut file and the alien job were both arranged on a P1S.
DEFAULT_PRINTER = "p1s"
MARGIN_XY = 16.0
MARGIN_Z = 6.0
# Smaller than this and the splitter cannot keep cuts away from the tips.
MIN_USABLE = 80.0


def parse_bed(text):
    """'300x300x250' -> (300.0, 300.0, 250.0)."""
    parts = str(text).lower().replace("×", "x").replace(" ", "").split("x")
    if len(parts) != 3:
        raise ValueError(f"bed must be W x D x H in mm, like 300x300x250, not {text!r}")
    try:
        bed = tuple(float(part) for part in parts)
    except ValueError:
        raise ValueError(f"bed must be W x D x H in mm, like 300x300x250, not {text!r}") from None
    if min(bed) <= 0:
        raise ValueError("bed sizes must be greater than 0")
    return bed


def resolve(printer, bed=None):
    """Name and bed size for a preset, or for a custom bed when `bed` is given."""
    if bed is not None:
        return "Custom", tuple(float(value) for value in bed)
    if printer not in PRINTERS:
        raise KrackError(f"unknown printer {printer!r}")
    return PRINTERS[printer]["name"], PRINTERS[printer]["bed"]


def bed_exclude(printer, custom=False):
    """Rectangles (x0, y0, x1, y1) on the bed where nothing may be placed."""
    if custom or printer not in PRINTERS:
        return []
    return PRINTERS[printer].get("exclude", [])


def usable_box(bed, margin_xy=MARGIN_XY, margin_z=MARGIN_Z):
    if isinstance(bed, str):
        bed = PRINTERS[bed]["bed"]
    if margin_xy < 0 or margin_z < 0:
        raise KrackError("margins cannot be negative")
    usable = (bed[0] - margin_xy, bed[1] - margin_xy, bed[2] - margin_z)
    if min(usable) < MIN_USABLE:
        raise KrackError(
            f"usable volume {usable[0]:g} x {usable[1]:g} x {usable[2]:g} mm is too small; "
            f"each side needs at least {MIN_USABLE:g} mm after the margins"
        )
    return usable
