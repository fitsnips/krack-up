"""Read STL and 3MF meshes, write binary STL and a plain 3MF project."""

import struct
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
PROD = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
MODEL_REL = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"


def read_stl(path):
    data = Path(path).read_bytes()
    if data[:5].lower() == b"solid" and b"\0" not in data[:200]:
        return _read_ascii_stl(data.decode("utf-8", "replace"))
    count = struct.unpack_from("<I", data, 80)[0]
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
    raise SystemExit(f"unsupported file type: {path.suffix}")


def write_3mf(path, objects):
    """Write a plain 3MF. Objects are (name, verts, faces) already in print position."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    resources = []
    items = []
    cursor_x = 0.0
    for index, (name, verts, faces) in enumerate(objects, start=1):
        resources.append(_object_xml(index, name, verts, faces))
        items.append(
            f'<item objectid="{index}" transform="1 0 0 0 1 0 0 0 1 {cursor_x:.3f} 0 0"/>'
        )
        width = float(np.max(verts[:, 0]) - np.min(verts[:, 0])) if len(verts) else 0.0
        cursor_x += width + 10.0
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


def _load_3mf(path):
    with zipfile.ZipFile(path) as archive:
        models = {"/3D/3dmodel.model": _parse_model(archive.read("3D/3dmodel.model"))}
        root = models["/3D/3dmodel.model"]
        names = _object_names(archive)
        loaded = []
        for item in root["build"]:
            verts, faces = _resolve(archive, models, "/3D/3dmodel.model", item["id"], item["transform"])
            if len(faces) == 0:
                continue
            loaded.append((names.get(item["id"]) or item["name"] or f"object_{item['id']}", verts, faces))
    if not loaded:
        raise SystemExit(f"no meshes found in {path}")
    return loaded


def _parse_model(data):
    # Namespaces vary. Strip them so the tree is easy to walk.
    text = data.decode("utf-8")
    root = ET.fromstring(text)
    objects = {}
    for obj in root.iter():
        if _tag(obj) != "object":
            continue
        mesh = None
        components = []
        for child in obj.iter():
            tag = _tag(child)
            if tag == "vertex":
                pass
            elif tag == "component":
                components.append(
                    {
                        "id": int(child.attrib["objectid"]),
                        "path": child.attrib.get(f"{{{PROD}}}path", ""),
                        "transform": _transform(child.attrib.get("transform")),
                    }
                )
        vertices = []
        faces = []
        mesh_node = next((c for c in obj if _tag(c) == "mesh"), None)
        if mesh_node is not None:
            for node in mesh_node.iter():
                if _tag(node) == "vertex":
                    vertices.append(
                        (float(node.attrib["x"]), float(node.attrib["y"]), float(node.attrib["z"]))
                    )
                elif _tag(node) == "triangle":
                    faces.append(
                        (int(node.attrib["v1"]), int(node.attrib["v2"]), int(node.attrib["v3"]))
                    )
            mesh = (
                np.asarray(vertices, dtype=np.float64),
                np.asarray(faces, dtype=np.int64),
            )
        objects[int(obj.attrib["id"])] = {"mesh": mesh, "components": components, "name": obj.attrib.get("name", "")}
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
                build.append({"id": obj_id, "name": obj["name"], "transform": np.eye(4)[:3]})
    return {"objects": objects, "build": build}


def _resolve(archive, models, model_path, obj_id, transform):
    model = _ensure_model(archive, models, model_path)
    obj = model["objects"][obj_id]
    chunks = []
    if obj["mesh"] is not None and len(obj["mesh"][0]) >= 50:
        verts, faces = obj["mesh"]
        chunks.append(_apply(verts, faces, transform))
    for comp in obj["components"]:
        path = comp["path"] or model_path
        combined = _compose(transform, comp["transform"])
        chunks.append(_resolve(archive, models, path, comp["id"], combined))
    if not chunks:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return _merge(chunks)


def _ensure_model(archive, models, path):
    key = path if path.startswith("/") else "/" + path
    if key not in models:
        models[key] = _parse_model(archive.read(key.lstrip("/")))
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
    """3MF transforms in these Bambu files are 9 matrix values, then tx ty tz."""
    out = np.zeros((3, 4), dtype=np.float64)
    out[:, :3] = np.eye(3)
    if not text:
        return out
    nums = [float(part) for part in text.split()]
    if len(nums) != 12:
        raise SystemExit(f"transform has {len(nums)} values, expected 12")
    out[:, :3] = np.asarray(nums[:9], dtype=np.float64).reshape(3, 3)
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
