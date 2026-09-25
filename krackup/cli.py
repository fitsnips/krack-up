"""Command line for krack-up."""

import argparse
import sys

from krackup import __version__
from krackup.printers import DEFAULT_PRINTER, PRINTERS
from krackup.run import describe, krack


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
    parser.add_argument("--3mf", action="store_true", help="also write project.3mf")
    parser.add_argument("--info", action="store_true", help="print part sizes and exit")
    parser.add_argument("--version", action="version", version=f"krack-up {__version__}")
    args = parser.parse_args(argv)
    if args.info:
        for line in describe(args.input):
            print(line)
        return 0
    output = args.output
    if not output:
        stem = args.input.rsplit(".", 1)[0]
        output = stem + "-krackup"
    krack(
        args.input,
        output,
        printer=args.printer,
        length=args.length,
        tolerance=args.tolerance,
        pitch=args.pitch,
        min_pins=args.min_pins,
        scale=args.scale,
        write_project=args.__dict__["3mf"],
        log=lambda msg: print(msg, file=sys.stderr),
    )
    print(output)
    return 0
