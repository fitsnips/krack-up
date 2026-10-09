"""Read STL and 3MF meshes, write binary STL and a plain 3MF project."""

import struct
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from krackup.solid import KrackError, manifold_from, mesh_arrays

CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
PROD = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
MODEL_REL = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
UNITS = {
    "micron": 0.001,
    "millimeter": 1.0,
    "centimeter": 10.0,
    "inch": 25.4,
    "foot": 304.8,
    "meter": 1000.0,
}
# 3MF object types that are not printed as part of the model.
SKIP_TYPES = {"support", "solidsupport", "surface", "other"}
# Bambu part subtypes. Modifiers and support volumes are not geometry.
SOLID_PART = "normal_part"
NEGATIVE_PART = "negative_part"


def read_stl(path):
    data = Path(path).read_bytes()
    count = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else -1
    # Some binary exporters also start the header with "solid", so trust the size.
    if len(data) != 84 + 50 * count:
        if data[:5].lower() == b"solid":
            return _read_ascii_stl(data.decode("utf-8", "replace"))
        raise KrackError(f"{path} is not a valid STL")
    record = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
    tris = np.frombuffer(data, dtype=record, count=count, offset=84)
    verts = np.ascontiguousarray(tris["v"].reshape(-1, 3), dtype=np.float64)
    faces = np.arange(len(verts), dtype=np.uint32).reshape(-1, 3)
    return verts, faces


def write_stl(path, verts, faces):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    triangles = np.asarray(verts, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    record = np.zeros(
        len(faces),
        dtype=np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")]),
    )
    record["n"] = (normals / length).astype(np.float32)
    record["v"] = triangles.astype(np.float32)
    with path.open("wb") as handle:
        handle.write(b"krack-up".ljust(80, b"\0"))
        handle.write(struct.pack("<I", len(faces)))
        handle.write(record.tobytes())


def load_models(path):
    """Return a list of (name, vertices, faces) in millimeters."""
    path = Path(path)
    if path.suffix.lower() == ".stl":
        verts, faces = read_stl(path)
        return [(path.stem, verts, faces)]
    if path.suffix.lower() == ".3mf":
        return _load_3mf(path)
    raise KrackError(f"unsupported file type: {path.suffix}")


def write_3mf(path, objects, layout=None, bed=None):
    """Write a 3MF. Objects are (name, verts, faces) already in print position.

    With `layout` (one (plate, x, y, turned) per object, from plates.pack)
    and `bed`, each object goes on its plate and Bambu Studio reads the
    plates from Metadata/model_settings.config. Without it they sit in a row.
    """
    from krackup.plates import plate_origin

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    resources = []
    items = []
    cursor_x = 0.0
    count = 1 + max((spot[0] for spot in layout), default=0) if layout else 0
    for index, (name, verts, faces) in enumerate(objects, start=1):
        resources.append(_object_xml(index, name, verts, faces))
        low = verts.min(axis=0) if len(verts) else np.zeros(3)
        high = verts.max(axis=0) if len(verts) else np.zeros(3)
        if layout:
            plate, x, y, turned = layout[index - 1]
            origin = plate_origin(plate, count, bed)
            middle = (low + high) / 2
            # Row-vector 3MF matrix; a turn sends +X to +Y.
            matrix = "0 1 0 -1 0 0 0 0 1" if turned else "1 0 0 0 1 0 0 0 1"
            if turned:
                middle = np.array([-middle[1], middle[0], middle[2]])
            move = (origin[0] + x - middle[0], origin[1] + y - middle[1], -low[2])
            items.append(
                f'<item objectid="{index}" transform="{matrix} '
                f'{move[0]:.3f} {move[1]:.3f} {move[2]:.3f}"/>'
            )
            continue
        # Put this object's left edge at the cursor so neighbours never overlap.
        items.append(
            f'<item objectid="{index}" transform="1 0 0 0 1 0 0 0 1 {cursor_x - low[0]:.3f} 0 0"/>'
        )
        cursor_x += (high[0] - low[0]) + 10.0
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<model unit="millimeter" xml:lang="en-US" xmlns="{CORE}">\n'
        " <metadata name=\"Application\">krack-up</metadata>\n"
        " <resources>\n"
        + "\n".join(resources)
        + "\n </resources>\n <build>\n  "
        + "\n  ".join(items)
        + "\n </build>\n</model>\n"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
            "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            f'<Relationships xmlns="{REL}">'
            f'<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="{MODEL_REL}"/>'
            "</Relationships>",
        )
        archive.writestr("3D/3dmodel.model", xml)
        if layout:
            archive.writestr("Metadata/model_settings.config", _plate_config(objects, layout, count))


def _plate_config(objects, layout, count):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<config>"]
    for index, (name, _, _) in enumerate(objects, start=1):
        safe = _escape(name)
        lines += [
            f'  <object id="{index}">',
            f'    <metadata key="name" value="{safe}"/>',
            '    <metadata key="extruder" value="1"/>',
            f'    <part id="1" subtype="{SOLID_PART}">',
            f'      <metadata key="name" value="{safe}"/>',
            '      <metadata key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>',
            "    </part>",
            "  </object>",
        ]
    for plate in range(count):
        lines += [
            "  <plate>",
            f'    <metadata key="plater_id" value="{plate + 1}"/>',
            '    <metadata key="plater_name" value=""/>',
            '    <metadata key="locked" value="false"/>',
        ]
        for index, spot in enumerate(layout, start=1):
            if spot[0] == plate:
                lines += [
                    "    <model_instance>",
                    f'      <metadata key="object_id" value="{index}"/>',
                    '      <metadata key="instance_id" value="0"/>',
                    "    </model_instance>",
                ]
        lines.append("  </plate>")
    lines.append("</config>")
    return "\n".join(lines) + "\n"


def _escape(text):
    return str(text).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def _load_3mf(path):
    with zipfile.ZipFile(path) as archive:
        root_path = _root_model_path(archive)
        models = {}
        root = _ensure_model(archive, models, root_path)
        names = _object_names(archive)
        subtypes = _part_subtypes(archive)
        loaded = []
        for item in root["build"]:
            chunks = _resolve(
                archive, models, root_path, item["id"], item["transform"],
                kinds=subtypes.get(item["id"], {}),
            )
            verts, faces = _combine(chunks)
            if len(faces) == 0:
                continue
            verts = verts * root["scale"]
            loaded.append((names.get(item["id"]) or item["name"] or f"object_{item['id']}", verts, faces))
    if not loaded:
        raise KrackError(f"no meshes found in {path}")
    return loaded


def _root_model_path(archive):
    try:
        rels = ET.fromstring(archive.read("_rels/.rels"))
    except KeyError:
        return "/3D/3dmodel.model"
    for rel in rels.iter():
        if _tag(rel) == "Relationship" and rel.attrib.get("Type") == MODEL_REL:
            target = rel.attrib.get("Target", "")
            return target if target.startswith("/") else "/" + target
    return "/3D/3dmodel.model"


def _parse_model(data):
    # Namespaces vary. Strip them so the tree is easy to walk.
    root = ET.fromstring(data)
    scale = UNITS.get(root.attrib.get("unit", "millimeter"), 1.0)
    objects = {}
    for obj in root.iter():
        if _tag(obj) != "object":
            continue
        components = []
        vertices = []
        faces = []
        for child in obj:
            if _tag(child) == "components":
                for comp in child:
                    if _tag(comp) != "component":
                        continue
                    components.append(
                        {
                            "id": int(comp.attrib["objectid"]),
                            "path": comp.attrib.get(f"{{{PROD}}}path", ""),
                            "transform": _transform(comp.attrib.get("transform")),
                        }
                    )
            elif _tag(child) == "mesh":
                for node in child.iter():
                    if _tag(node) == "vertex":
                        vertices.append(
                            (float(node.attrib["x"]), float(node.attrib["y"]), float(node.attrib["z"]))
                        )
                    elif _tag(node) == "triangle":
                        faces.append(
                            (int(node.attrib["v1"]), int(node.attrib["v2"]), int(node.attrib["v3"]))
                        )
        mesh = None
        if faces:
            mesh = (
                np.asarray(vertices, dtype=np.float64),
                np.asarray(faces, dtype=np.int64),
            )
        objects[int(obj.attrib["id"])] = {
            "mesh": mesh,
            "components": components,
            "name": obj.attrib.get("name", ""),
            "type": obj.attrib.get("type", "model"),
        }
    build = []
    for item in root.iter():
        if _tag(item) != "item":
            continue
        build.append(
            {
                "id": int(item.attrib["objectid"]),
                "name": item.attrib.get("name", ""),
                "transform": _transform(item.attrib.get("transform")),
            }
        )
    if not build:
        # Some files omit <build> and just store one mesh.
        for obj_id, obj in objects.items():
            if obj["mesh"] is not None:
                build.append({"id": obj_id, "name": obj["name"], "transform": _transform(None)})
    return {"objects": objects, "build": build, "scale": scale}


def _resolve(
    archive, models, model_path, obj_id, transform,
    kinds=None, kind=SOLID_PART, root_scale=None, explicit=False,
):
    """Return (verts, faces, kind) chunks for an object and its components.

    `kinds` maps component object ids to Bambu part subtypes. It only applies to
    the first level of components, which is where Bambu Studio puts parts.
    Bambu marks every part but the body type="other", so when it names a
    subtype that decides, not the 3MF object type.
    """
    model = _ensure_model(archive, models, model_path)
    if root_scale is None:
        root_scale = model["scale"]
    obj = model["objects"][obj_id]
    if obj["type"] in SKIP_TYPES and not explicit:
        return []
    chunks = []
    if obj["mesh"] is not None:
        verts, faces = obj["mesh"]
        # Vertices in another model file may use another unit.
        verts = verts * (model["scale"] / root_scale)
        moved, faces = _apply(verts, faces, transform)
        chunks.append((moved, faces, kind))
    for comp in obj["components"]:
        path = comp["path"] or model_path
        combined = _compose(transform, comp["transform"])
        named = bool(kinds) and comp["id"] in kinds
        comp_kind = kinds[comp["id"]] if named else kind
        chunks.extend(
            _resolve(
                archive, models, path, comp["id"], combined,
                None, comp_kind, root_scale, explicit or named,
            )
        )
    return chunks


def _combine(chunks):
    """Union the solid parts of one object and subtract its negative parts."""
    solids = [(v, f) for v, f, kind in chunks if kind == SOLID_PART and len(f)]
    holes = [(v, f) for v, f, kind in chunks if kind == NEGATIVE_PART and len(f)]
    if not solids:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    if len(solids) == 1 and not holes:
        return solids[0]
    try:
        body = _union([manifold_from(v, f) for v, f in solids])
        if holes:
            body = body - _union([manifold_from(v, f) for v, f in holes])
        verts, faces = mesh_arrays(body)
        if len(faces):
            return verts, faces
    except KrackError:
        pass
    # A part that is not a closed solid cannot go through booleans. Keep the
    # solid parts as they are, which is what the file shows without its cuts.
    return _merge(solids)


def _union(solids):
    body = solids[0]
    for solid in solids[1:]:
        body = body + solid
    return body


def _ensure_model(archive, models, path):
    key = path if path.startswith("/") else "/" + path
    if key not in models:
        try:
            data = archive.read(key.lstrip("/"))
        except KeyError:
            raise KrackError(f"3MF is missing {key}") from None
        models[key] = _parse_model(data)
    return models[key]


def _apply(verts, faces, transform):
    if len(verts) == 0:
        return verts, faces
    linear = transform[:, :3]
    moved = verts @ linear.T + transform[:, 3]
    return moved, faces


def _compose(parent, child):
    """Apply child first, then parent. Both are 3x4."""
    parent_r, parent_t = parent[:, :3], parent[:, 3]
    child_r, child_t = child[:, :3], child[:, 3]
    out = np.zeros((3, 4), dtype=np.float64)
    out[:, :3] = parent_r @ child_r
    out[:, 3] = parent_r @ child_t + parent_t
    return out


def _merge(chunks):
    verts = []
    faces = []
    offset = 0
    for vert, face in chunks:
        if len(face) == 0:
            continue
        verts.append(vert)
        faces.append(face + offset)
        offset += len(vert)
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.vstack(verts), np.vstack(faces).astype(np.int64)


def _transform(text):
    """Return a 3x4 [R | t] that maps column vectors: p' = R p + t.

    3MF writes "m00 m01 m02 m10 m11 m12 m20 m21 m22 m30 m31 m32" and applies
    it to row vectors, p' = p M, so R is the transpose of the first nine.
    """
    out = np.zeros((3, 4), dtype=np.float64)
    out[:, :3] = np.eye(3)
    if not text:
        return out
    nums = [float(part) for part in text.split()]
    if len(nums) != 12:
        raise KrackError(f"transform has {len(nums)} values, expected 12")
    out[:, :3] = np.asarray(nums[:9], dtype=np.float64).reshape(3, 3).T
    out[:, 3] = nums[9:12]
    return out


def _object_names(archive):
    if "Metadata/model_settings.config" not in archive.namelist():
        return {}
    root = ET.fromstring(archive.read("Metadata/model_settings.config"))
    names = {}
    for obj in root.iter():
        if _tag(obj) != "object" or "id" not in obj.attrib:
            continue
        for meta in obj:
            if _tag(meta) == "metadata" and meta.attrib.get("key") == "name":
                names[int(obj.attrib["id"])] = meta.attrib.get("value", "")
                break
    return names


def _part_subtypes(archive):
    """Bambu Studio keeps each part's role in model_settings.config.

    Returns {object id: {component object id: subtype}}.
    """
    if "Metadata/model_settings.config" not in archive.namelist():
        return {}
    root = ET.fromstring(archive.read("Metadata/model_settings.config"))
    found = {}
    for obj in root.iter():
        if _tag(obj) != "object" or "id" not in obj.attrib:
            continue
        parts = {}
        for part in obj:
            if _tag(part) == "part" and "id" in part.attrib:
                parts[int(part.attrib["id"])] = part.attrib.get("subtype", SOLID_PART)
        found[int(obj.attrib["id"])] = parts
    return found


def _tag(node):
    tag = node.tag
    return tag.rsplit("}", 1)[-1]


def _object_xml(index, name, verts, faces):
    vert_lines = "\n".join(
        f'    <vertex x="{v[0]:.5f}" y="{v[1]:.5f}" z="{v[2]:.5f}"/>' for v in verts
    )
    face_lines = "\n".join(
        f'    <triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}"/>' for f in faces
    )
    safe = (
        str(name)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
    )
    return (
        f'  <object id="{index}" type="model" name="{safe}">\n'
        "   <mesh>\n    <vertices>\n"
        + vert_lines
        + "\n    </vertices>\n    <triangles>\n"
        + face_lines
        + "\n    </triangles>\n   </mesh>\n  </object>"
    )


def _read_ascii_stl(text):
    verts = []
    for line in text.splitlines():
        parts = line.split()
        if parts and parts[0] == "vertex":
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
    arr = np.asarray(verts, dtype=np.float64)
    faces = np.arange(len(arr), dtype=np.uint32).reshape(-1, 3)
    return arr, faces
