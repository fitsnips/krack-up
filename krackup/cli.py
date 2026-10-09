"""Command line for krack-up."""

import argparse
import sys
from pathlib import Path

from krackup import __version__
from krackup.printers import DEFAULT_PRINTER, MARGIN_XY, MARGIN_Z, PRINTERS, parse_bed
from krackup.run import describe, krack
from krackup.solid import KrackError


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] in ("ui", "--ui"):
        from krackup.ui import serve

        return serve()
    parser = argparse.ArgumentParser(
        prog="krack-up",
        description=(
            "Cut a model into pieces that fit on a printer bed and add "
            "pentagon dowels. Writes meshes only, not gcode."
        ),
    )
    parser.add_argument("input", help="STL or 3MF to cut")
    parser.add_argument("-o", "--output", help="output directory (default: <name>-krackup)")
    parser.add_argument(
        "--printer",
        default=DEFAULT_PRINTER,
        choices=sorted(PRINTERS),
        help="bed to fit (default: p1s, 256 x 256 x 250)",
    )
    parser.add_argument(
        "--bed",
        type=_bed,
        help="custom bed W x D x H in mm, like 300x300x250; overrides --printer",
    )
    parser.add_argument(
        "--margin-xy",
        type=float,
        default=MARGIN_XY,
        help=f"mm taken off the bed width and depth for brim and clips (default: {MARGIN_XY:g})",
    )
    parser.add_argument(
        "--margin-z",
        type=float,
        default=MARGIN_Z,
        help=f"mm taken off the bed height (default: {MARGIN_Z:g})",
    )
    parser.add_argument(
        "--length",
        type=float,
        default=10.0,
        help="dowel length in mm (default: 10, from the Jimmy cut)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="extra mm in the hole, radial and on each end (default: 0.1)",
    )
    parser.add_argument(
        "--every",
        "--pitch",
        dest="pitch",
        type=float,
        default=100.0,
        help="one dowel per this many mm of mating face (default: 100)",
    )
    parser.add_argument(
        "--min-pins",
        type=int,
        default=2,
        help="at least this many dowels on each cut face (default: 2)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="scale the model about its center before cutting (default: 1)",
    )
    parser.add_argument(
        "--no-labels",
        dest="labels",
        action="store_false",
        help="do not engrave part numbers on the joint faces",
    )
    parser.add_argument("--3mf", action="store_true", help="also write project.3mf")
    parser.add_argument("--info", action="store_true", help="print part sizes and exit")
    parser.add_argument("--version", action="version", version=f"krack-up {__version__}")
    args = parser.parse_args(argv)
    if args.min_pins < 1:
        parser.error("--min-pins must be at least 1")
    for name in ("length", "pitch", "scale"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be greater than 0")
    if args.tolerance < 0:
        parser.error("--tolerance cannot be negative")
    if args.margin_xy < 0 or args.margin_z < 0:
        parser.error("margins cannot be negative")
    try:
        return _run(args)
    except KrackError as exc:
        print(f"krack-up: {exc}", file=sys.stderr)
        return 1


def _bed(text):
    try:
        return parse_bed(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _run(args):
    if args.info:
        for line in describe(args.input):
            print(line)
        return 0
    output = args.output
    if not output:
        output = str(Path(args.input).with_suffix("")) + "-krackup"
    krack(
        args.input,
        output,
        printer=args.printer,
        length=args.length,
        tolerance=args.tolerance,
        pitch=args.pitch,
        min_pins=args.min_pins,
        scale=args.scale,
        bed=args.bed,
        margin_xy=args.margin_xy,
        margin_z=args.margin_z,
        labels=args.labels,
        write_project=args.__dict__["3mf"],
        log=lambda msg: print(msg, file=sys.stderr),
    )
    print(output)
    return 0
