"""Load, cut, dowel, orient, and write a folder of meshes."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from krackup.faces import pin_existing_faces
from krackup.labels import DEPTH, label_parts
from krackup.meshio import load_models, write_3mf, write_stl
from krackup.orient import place
from krackup.pins import RADII, add_dowels, dowel_solid
from krackup.plates import pack
from krackup.printers import MARGIN_XY, MARGIN_Z, bed_exclude, resolve, usable_box
from krackup.solid import KrackError, manifold_from, mesh_arrays
from krackup.split import split_to_fit

__all__ = ["KrackError", "Stopped", "describe", "krack", "manifold_from", "scale_vertices"]


def describe(path):
    lines = []
    for name, verts, faces in load_models(path):
        size = verts.max(axis=0) - verts.min(axis=0)
        lines.append(
            f"{name}: {len(faces)} triangles, "
            f"{size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} mm"
        )
    return lines


class Stopped(Exception):
    """The UI asked the running cut to stop."""


def scale_vertices(verts, scale):
    if scale == 1:
        return verts
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2
    return (verts - center) * scale + center


def krack(
    path,
    output,
    printer,
    length,
    tolerance,
    pitch,
    write_project,
    min_pins=2,
    scale=1.0,
    bed=None,
    margin_xy=MARGIN_XY,
    margin_z=MARGIN_Z,
    labels=True,
    should_stop=None,
    log=print,
):
    custom = bed is not None
    printer_name, bed = resolve(printer, bed)
    limit = np.array(usable_box(bed, margin_xy, margin_z), dtype=np.float64)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    part_dir = output / "parts"
    dowel_dir = output / "dowels"
    part_dir.mkdir(exist_ok=True)
    dowel_dir.mkdir(exist_ok=True)
    for old in list(part_dir.glob("*.stl")) + list(dowel_dir.glob("*.stl")):
        old.unlink()

    def emit(message):
        if should_stop and should_stop():
            raise Stopped()
        log(message)

    all_pins = []
    pending = []
    for origin, (source_name, verts, faces) in enumerate(load_models(path)):
        emit(f"\n{source_name}")
        verts = scale_vertices(verts, scale)
        solid = manifold_from(verts, faces)
        pieces, cuts = split_to_fit(solid, limit, log=emit)
        pins, areas = add_dowels(pieces, cuts, length, tolerance, pitch, min_pins, log=emit)
        for piece in pieces:
            piece.source = source_name
            piece.origin = origin
            piece.planes = [(cuts[i]["normal"], cuts[i]["offset"]) for i in piece.cuts]
        pending.append((source_name, pieces, cuts, pins, areas))
        all_pins.extend(pins)

    every_piece = [piece for _, pieces, _, _, _ in pending for piece in pieces]
    file_pins = []
    if len({piece.origin for piece in every_piece}) > 1:
        emit("checking cut faces that came in with the file")
        file_pins = pin_existing_faces(every_piece, length, tolerance, pitch, min_pins, log=emit)
        all_pins.extend(file_pins)

    holes_by_piece = {}
    for pin in all_pins:
        for hole in pin["holes"]:
            holes_by_piece.setdefault(id(hole["piece"]), []).append(hole)
    records = []
    part_index = 1
    for source_name, pieces, cuts, _, _ in pending:
        group = [
            {
                "source": source_name,
                "piece": piece,
                "cuts": [cuts[i] for i in piece.cuts],
                "holes": holes_by_piece.get(id(piece), []),
                "centroid": _centroid(piece.solid),
            }
            for piece in pieces
        ]
        # Number the parts of each source bottom to top.
        group.sort(key=lambda rec: (rec["centroid"][2], rec["centroid"][0]))
        for rec in group:
            rec["id"] = part_index
            part_index += 1
        records.extend(group)
    id_by_piece = {id(rec["piece"]): rec["id"] for rec in records}

    for rec in records:
        piece = rec["piece"]
        posed = place(piece.solid, [cut["normal"] for cut in rec["cuts"]], limit)
        if posed is None:
            size = piece.solid.bounding_box()
            raise KrackError(f"a piece of {rec['source']} does not fit: {size}")
        rec["posed"] = posed
    unlabeled = []
    if labels and len(records) > 1:
        emit("engraving part numbers")
        # Pose first, so the number can go on a face that is not on the bed.
        unlabeled = label_parts(records, log=emit)
        for rec in records:
            normals = [cut["normal"] for cut in rec["cuts"]]
            posed = place(rec["piece"].solid, normals, limit, only=rec["posed"]["name"])
            if posed is not None:
                rec["posed"] = posed

    all_joints = []
    for source_name, _, cuts, pins, areas in pending:
        for cut in cuts:
            joint_pins = [pin for pin in pins if pin["cut_id"] == cut["id"]]
            if areas.get(cut["id"], 0) < 80 and not joint_pins:
                continue
            all_joints.append(
                _joint(source_name, cut["id"], cut["reason"], areas.get(cut["id"], 0), joint_pins, id_by_piece)
            )
    by_joint = {}
    for pin in file_pins:
        by_joint.setdefault(pin["cut_id"], []).append(pin)
    for joint_id, joint_pins in by_joint.items():
        first = joint_pins[0]
        all_joints.append(
            _joint(
                f"{first['positive'].source} + {first['negative'].source}",
                joint_id,
                "cut face in the file",
                first["area"],
                joint_pins,
                id_by_piece,
            )
        )

    manifest_parts = []
    project = []
    for rec in records:
        posed = rec["posed"]
        filename = f"part_{rec['id']:03d}.stl"
        write_stl(part_dir / filename, posed["verts"], posed["faces"])
        size = [round(float(v), 1) for v in posed["size"]]
        log(
            f"part_{rec['id']:03d}  {size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f}"
            f"  {posed['name']}  overhang {posed['overhang']:.0f} mm2"
        )
        manifest_parts.append(
            {
                "id": rec["id"],
                "file": f"parts/{filename}",
                "source": rec["source"],
                "print_size_mm": size,
                "orientation": posed["name"],
                "overhang_area_mm2": round(posed["overhang"], 1),
                "assembly_centroid_mm": [round(float(v), 1) for v in rec["centroid"]],
            }
        )
        project.append((filename, posed["verts"], posed["faces"]))

    counts = Counter((pin["radius"], pin["length"]) for pin in all_pins)
    dowel_files = []
    for (radius, pin_length), count in sorted(counts.items()):
        verts, faces = mesh_arrays(dowel_solid(radius, pin_length))
        verts = verts.copy()
        verts[:, 2] -= verts[:, 2].min()
        # :g keeps 7.5 distinguishable from 7 and 10.
        filename = f"pentagon_R{radius:g}_L{pin_length:g}.stl"
        write_stl(dowel_dir / filename, verts, faces)
        dowel_files.append(
            {"file": f"dowels/{filename}", "radius_mm": radius, "length_mm": pin_length, "count": count}
        )
        log(f"dowel {filename} x{count}")
        stem = filename[: -len(".stl")]
        project.extend((f"{stem}_{n}.stl", verts, faces) for n in range(1, count + 1))

    layout = None
    plates = 0
    if write_project and project:
        sizes = [tuple(np.ptp(verts[:, :2], axis=0)) for _, verts, _ in project]
        layout = pack(sizes, bed, margin_xy / 2, exclude=bed_exclude(printer, custom))
        plates = 1 + max(spot[0] for spot in layout)

    manifest = {
        "source": str(path),
        "printer": printer_name,
        "bed_mm": list(bed),
        "usable_mm": [round(float(v), 1) for v in limit],
        "margin_xy_mm": margin_xy,
        "margin_z_mm": margin_z,
        "labels": {
            "engraved": bool(labels and len(records) > 1),
            "depth_mm": DEPTH,
            "unlabeled_parts": unlabeled,
        },
        "dowel": {
            "shape": "pentagon",
            "length_mm": length,
            "radial_tolerance_mm": tolerance,
            "axial_tolerance_mm": tolerance,
            "pitch_mm": pitch,
            "min_pins_per_plane": min_pins,
            "scale": scale,
            "radii_mm": list(RADII),
        },
        "parts": manifest_parts,
        "joints": all_joints,
        "dowels": dowel_files,
        "plates": plates,
        "gcode": False,
    }
    (output / "assembly.json").write_text(json.dumps(manifest, indent=2))
    (output / "ASSEMBLY.txt").write_text(_text(manifest))
    if layout:
        write_3mf(output / "project.3mf", project, layout=layout, bed=bed)
        log(f"project.3mf: {len(project)} objects on {_plates(plates)}")
    return manifest


def _joint(source, joint_id, reason, area, pins, id_by_piece):
    return {
        "source": source,
        "cut_id": joint_id,
        "reason": reason,
        "mating_area_mm2": round(float(area), 1),
        "dowels": len(pins),
        "radii_mm": sorted({pin["radius"] for pin in pins}),
        "parts_a": sorted({id_by_piece[id(pin["positive"])] for pin in pins}),
        "parts_b": sorted({id_by_piece[id(pin["negative"])] for pin in pins}),
    }


def _plates(count):
    return f"{count} plate" if count == 1 else f"{count} plates"


def _label_lines(labels):
    if not labels["engraved"]:
        return []
    lines = [
        f"Each part number is engraved {labels['depth_mm']:g} mm deep on one of its "
        "joint faces, with a bar under the digits.",
    ]
    if labels["unlabeled_parts"]:
        lines.append(f"No room for a number on parts {labels['unlabeled_parts']}.")
    return lines


def _centroid(solid):
    verts, _ = mesh_arrays(solid)
    return verts.mean(axis=0)


def _text(manifest):
    dowel = manifest["dowel"]
    radii = ", ".join(f"{radius:g}" for radius in dowel["radii_mm"][:-1])
    radii = f"{radii}, or {dowel['radii_mm'][-1]:g} mm"
    lines = [
        "krack-up",
        "",
        f"Printer: {manifest['printer']} bed {manifest['bed_mm'][0]:.0f} x "
        f"{manifest['bed_mm'][1]:.0f} x {manifest['bed_mm'][2]:.0f} mm.",
        f"Each part fits in {manifest['usable_mm'][0]:.0f} x "
        f"{manifest['usable_mm'][1]:.0f} x {manifest['usable_mm'][2]:.0f} mm "
        "so a brim still lands on the bed.",
        "",
        f"Pieces: {len(manifest['parts'])}",
        f"Dowels: {sum(item['count'] for item in manifest['dowels'])}",
        f"Shape: pentagon, {dowel['length_mm']:g} mm long.",
        f"Circumradius is {radii}, whichever is the largest that fits the joint.",
        f"One dowel every {dowel['pitch_mm']:g} mm across each cut, "
        f"and at least {dowel['min_pins_per_plane']} dowels on every cut plane.",
        f"The hole is {dowel['radial_tolerance_mm']:g} mm larger in radius and "
        f"{dowel['axial_tolerance_mm']:g} mm deeper on each side.",
        "Sockets are cut into both faces. The dowels are separate STLs.",
        "",
        "Print each part in the pose it was exported. Z = 0 is the bed.",
        *_label_lines(manifest["labels"]),
        *(
            [f"project.3mf has every part and dowel laid out on {_plates(manifest['plates'])}."]
            if manifest["plates"]
            else []
        ),
        "This folder is meshes only. It is not gcode.",
        "",
        "Joints:",
    ]
    for joint in manifest["joints"]:
        if not joint["dowels"]:
            continue
        lines.append(
            f"  {joint['source']} cut {joint['cut_id']}: "
            f"parts {joint['parts_a']} mate with {joint['parts_b']}  "
            f"{joint['mating_area_mm2']:.0f} mm2, {joint['dowels']} dowels, "
            f"radius {joint['radii_mm']}"
        )
    lines.append("")
    lines.append("Dowels to print:")
    for item in manifest["dowels"]:
        lines.append(
            f"  {item['count']} x {item['file']}  "
            f"(circumradius {item['radius_mm']:g} mm, {item['length_mm']:.0f} mm long)"
        )
    return "\n".join(lines) + "\n"
