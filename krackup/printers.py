"""Bed sizes. Usable volume leaves room for a 5 mm brim and the P1S corner."""

PRINTERS = {
    "p1s": {"name": "Bambu Lab P1S", "bed": (256.0, 256.0, 250.0)},
    "x1c": {"name": "Bambu Lab X1C", "bed": (256.0, 256.0, 256.0)},
    "mini": {"name": "Prusa MINI", "bed": (180.0, 180.0, 180.0)},
}

# Jimmy's cut file and the alien job were both arranged on a P1S.
DEFAULT_PRINTER = "p1s"
MARGIN_XY = 16.0
MARGIN_Z = 6.0


def usable_box(printer):
    bed = PRINTERS[printer]["bed"]
    return (
        bed[0] - MARGIN_XY,
        bed[1] - MARGIN_XY,
        bed[2] - MARGIN_Z,
    )
