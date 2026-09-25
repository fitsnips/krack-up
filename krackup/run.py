"""Load, cut, dowel, orient, and write a folder of meshes."""

import json
from collections import Counter
from pathlib import Path

import manifold3d as mf
import numpy as np

from krackup.meshio import load_models, write_3mf, write_stl
from krackup.orient import place
from krackup.pins import add_dowels, dowel_solid
from krackup.printers import MARGIN_XY, MARGIN_Z, PRINTERS, usable_box
from krackup.split import split_to_fit


def manifold_from(verts, faces):
    verts, faces = _weld(verts, faces)
    verts, faces = _fill_small_holes(verts, faces)
    mesh = mf.Mesh(
        vert_properties=np.ascontiguousarray(verts, dtype=np.float32),
        tri_verts=np.ascontiguousarray(faces, dtype=np.uint32),
    )
    solid = mf.Manifold(mesh)
    status = str(solid.status())
    if "NoError" not in status:
        raise SystemExit(f"mesh is not a solid ({status})")
    return solid


def _weld(verts, faces):
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    _, index, inverse = np.unique(np.round(verts, 5), axis=0, return_index=True, return_inverse=True)
    welded = verts[index]
    faces = inverse[faces]
    keep = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 2] != faces[:, 0])
    )
    return welded, faces[keep]


def _fill_small_holes(verts, faces):
    """Close gaps of a few edges. A missing triangle makes Manifold reject the part."""
    faces = np.asarray(faces, dtype=np.int64)
    directed = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    undirected = np.sort(directed, axis=1)
    _, counts = np.unique(undirected, axis=0, return_counts=True)
    if not np.any(counts == 1):
        return verts, faces
    # Keep the direction used by the existing face. The patch uses the opposite.
    packed = directed[:, 0].astype(np.int64) << 32 | directed[:, 1].astype(np.int64)
    reverse = directed[:, 1].astype(np.int64) << 32 | directed[:, 0].astype(np.int64)
    boundary = set(packed.tolist()) - set(reverse.tolist())
    if not boundary:
        return verts, faces
    nxt = {}
    for key in boundary:
        nxt[int(key >> 32)] = int(key & 0xFFFFFFFF)
    seen = set()
    extra = []
    for start in list(nxt):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        cursor = nxt[start]
        while cursor != start and cursor not in seen and cursor in nxt:
            loop.append(cursor)
            seen.add(cursor)
            cursor = nxt[cursor]
        if cursor != start or len(loop) < 3 or len(loop) > 12:
            continue
        loop = loop[::-1]
        for i in range(1, len(loop) - 1):
            extra.append((loop[0], loop[i], loop[i + 1]))
    if not extra:
        return verts, faces
    return verts, np.vstack([faces, np.asarray(extra, dtype=np.int64)])


def describe(path):
    lines = []
    for name, verts, faces in load_models(path):
        size = verts.max(axis=0) - verts.min(axis=0)
        lines.append(
            f"{name}: {len(faces)} triangles, "
            f"{size[0]:.0f} x {size[1]:.0f} x {size[2]:.0f} mm"
        )
    return lines


def krack(path, output, printer, length, tolerance, pitch, write_project, min_pins=2, log=print):
    limit = np.array(usable_box(printer), dtype=np.float64)
    bed = PRINTERS[printer]["bed"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    part_dir = output / "parts"
    dowel_dir = output / "dowels"
    part_dir.mkdir(exist_ok=True)
    dowel_dir.mkdir(exist_ok=True)
    for old in list(part_dir.glob("*.stl")) + list(dowel_dir.glob("*.stl")):
        old.unlink()

    records = []
    all_pins = []
    all_joints = []
    part_index = 1
    for source_name, verts, faces in load_models(path):
        log(f"\n{source_name}")
        solid = manifold_from(verts, faces)
        pieces, cuts = split_to_fit(solid, limit, log=log)
        pins, areas = add_dowels(pieces, cuts, length, tolerance, pitch, min_pins, log=log)
        for piece in pieces:
            normals = [cuts[i]["normal"] for i in piece.cuts]
            posed = place(piece.solid, normals, limit)
            if posed is None:
                size = piece.solid.bounding_box()
                raise SystemExit(f"a piece of {source_name} does not fit: {size}")
            records.append(
                {
                    "source": source_name,
                    "piece": piece,
                    "posed": posed,
                    "centroid": _centroid(piece.solid),
                }
            )
        id_by_piece = {}
        # ids assigned after all pieces of this source exist, sorted by height
        group = records[-len(pieces):]
        group.sort(key=lambda rec: (rec["centroid"][2], rec["centroid"][0]))
        for rec in group:
            rec["id"] = part_index
            id_by_piece[id(rec["piece"])] = part_index
            part_index += 1
        for cut in cuts:
            joint_pins = [pin for pin in pins if pin["cut_id"] == cut["id"]]
            if areas.get(cut["id"], 0) < 80 and not joint_pins:
                continue
            all_joints.append(
                {
                    "source": source_name,
                    "cut_id": cut["id"],
                    "reason": cut["reason"],
                    "mating_area_mm2": round(areas.get(cut["id"], 0), 1),
                    "dowels": len(joint_pins),
                    "radii_mm": sorted({pin["radius"] for pin in joint_pins}),
                    "parts_a": sorted({id_by_piece[id(pin["positive"])] for pin in joint_pins}),
                    "parts_b": sorted({id_by_piece[id(pin["negative"])] for pin in joint_pins}),
                }
            )
        all_pins.extend(pins)

    records.sort(key=lambda rec: rec["id"])
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
        solid = dowel_solid(radius, pin_length)
        mesh = solid.to_mesh()
        verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
        faces = np.asarray(mesh.tri_verts, dtype=np.int64)
        verts = verts.copy()
        verts[:, 2] -= verts[:, 2].min()
        filename = f"pentagon_R{radius:.1f}_L{pin_length:.0f}.stl".replace(".0", "")
        # 7.5 must stay distinguishable from 7 and 10.
        filename = f"pentagon_R{radius:g}_L{pin_length:.0f}.stl"
        write_stl(dowel_dir / filename, verts, faces)
        dowel_files.append(
            {"file": f"dowels/{filename}", "radius_mm": radius, "length_mm": pin_length, "count": count}
        )
        log(f"dowel {filename} x{count}")

    manifest = {
        "source": str(path),
        "printer": PRINTERS[printer]["name"],
        "bed_mm": list(bed),
        "usable_mm": [round(float(v), 1) for v in limit],
        "margin_xy_mm": MARGIN_XY,
        "margin_z_mm": MARGIN_Z,
        "dowel": {
            "shape": "pentagon",
            "length_mm": length,
            "radial_tolerance_mm": tolerance,
            "axial_tolerance_mm": tolerance,
            "pitch_mm": pitch,
            "min_pins_per_plane": min_pins,
            "radii_mm": list(RADII_NOTE()),
        },
        "parts": manifest_parts,
        "joints": all_joints,
        "dowels": dowel_files,
        "gcode": False,
    }
    (output / "assembly.json").write_text(json.dumps(manifest, indent=2))
    (output / "ASSEMBLY.txt").write_text(_text(manifest))
    if write_project and project:
        write_3mf(output / "project.3mf", project)
    return manifest


def RADII_NOTE():
    return [10.0, 7.5, 5.0]


def _centroid(solid):
    mesh = solid.to_mesh()
    verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    return verts.mean(axis=0)


def _text(manifest):
    lines = [
        "krack-up",
        "",
        f"Printer: {manifest['printer']} bed {manifest['bed_mm'][0]:.0f} x "
        f"{manifest['bed_mm'][1]:.0f} x {manifest['bed_mm'][2]:.0f} mm.",
        f"Each part fits in {manifest['usable_mm'][0]:.0f} x "
        f"{manifest['usable_mm'][1]:.0f} x {manifest['usable_mm'][2]:.0f} mm "
        "so a 5 mm brim still lands on the bed.",
        "",
        f"Pieces: {len(manifest['parts'])}",
        f"Dowels: {sum(item['count'] for item in manifest['dowels'])}",
        "Shape: pentagon. Length 10 mm unless you changed --length.",
        "Circumradius is 10, 7.5, or 5 mm, whichever is the largest that fits the joint.",
        f"One dowel every {manifest['dowel']['pitch_mm']:.0f} mm across each cut, "
        f"and at least {manifest['dowel']['min_pins_per_plane']} dowels on every cut plane.",
        "The hole is 0.1 mm larger in radius and 0.1 mm deeper on each side.",
        "Sockets are cut into both faces. The dowels are separate STLs.",
        "",
        "Print each part in the pose it was exported. Z = 0 is the bed.",
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
