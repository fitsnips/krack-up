"""Write project.3mf as a Bambu Studio project: plates, printer and settings.

The settings come from the input file when it is a Bambu project for the
same printer, so its filaments and process carry over. Otherwise they are
built from the Bambu Studio profiles installed on this machine. With
neither, the caller falls back to a plain 3MF.
"""

import glob
import json
import os
import uuid
import zipfile
from pathlib import Path

from krackup.meshio import CORE, MODEL_REL, PROD, REL, SOLID_PART, _escape
from krackup.plates import plate_origin

BAMBU_NS = "http://schemas.bambulab.com/package/2021"
# krack-up printer -> Bambu machine preset.
MACHINES = {
    "p1s": "Bambu Lab P1S 0.4 nozzle",
    "x1c": "Bambu Lab X1 Carbon 0.4 nozzle",
    "a1": "Bambu Lab A1 0.4 nozzle",
    "a1mini": "Bambu Lab A1 mini 0.4 nozzle",
}
PROFILE_DIRS = (
    "~/.var/app/com.bambulab.BambuStudio/config/BambuStudio/system",
    "~/.config/BambuStudio/system",
    "~/Library/Application Support/BambuStudio/system",
    "%APPDATA%/BambuStudio/system",
    "/var/lib/flatpak/app/com.bambulab.BambuStudio/current/active/files/share/BambuStudio/profiles",
    "~/.local/share/flatpak/app/com.bambulab.BambuStudio/current/active/files/share/BambuStudio/profiles",
    "/usr/share/BambuStudio/profiles",
    "/Applications/BambuStudio.app/Contents/Resources/profiles",
)
# Keys that describe a profile file, not a print setting.
PROFILE_KEYS = {"type", "inherits", "include", "instantiation", "setting_id", "base_id"}


def project_settings(printer, source=None):
    """Settings for `printer`, or None when it is not a Bambu printer we can set up.

    Returns (settings, where) with `where` saying which source was used.
    """
    machine = MACHINES.get(printer)
    if machine is None:
        return None
    found = _from_source(source, machine)
    if found is not None:
        return found, f"settings from {Path(source).name}"
    root = _profile_root()
    if root is None:
        return None
    return _from_profiles(root, machine), f"settings from Bambu Studio profiles ({machine})"


def spacing(settings, base):
    """Gap between parts so brims and supports from neighbours do not touch."""
    gap = base
    if settings is None:
        return gap
    if str(settings.get("brim_type", "auto_brim")) != "no_brim":
        brim = _number(settings.get("brim_width"), 5.0) + _number(settings.get("brim_object_gap"), 0.0)
        gap = max(gap, 2 * brim + 2.0)
    if str(settings.get("enable_support", "0")) == "1":
        gap += 4.0
    return gap


def _number(value, default):
    if isinstance(value, list):
        value = value[0] if value else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_project(path, objects, layout, bed, settings, title=""):
    """Objects are (name, verts, faces) in print position; layout from plates.pack."""
    count = 1 + max(spot[0] for spot in layout)
    version = str(settings.get("version") or "02.08.00.00")
    resources, items, configs, files = [], [], [], {}
    for k, ((name, verts, faces), spot) in enumerate(zip(objects, layout), start=1):
        mesh_id, object_id = 2 * k - 1, 2 * k
        part_path = f"3D/Objects/object_{k}.model"
        files[part_path] = _mesh_model(mesh_id, verts, faces)
        resources.append(
            f'  <object id="{object_id}" p:UUID="{uuid.uuid4()}" type="model">\n'
            "   <components>\n"
            f'    <component p:path="/{part_path}" objectid="{mesh_id}" p:UUID="{uuid.uuid4()}" '
            'transform="1 0 0 0 1 0 0 0 1 0 0 0"/>\n'
            "   </components>\n"
            "  </object>"
        )
        items.append(
            f'  <item objectid="{object_id}" p:UUID="{uuid.uuid4()}" '
            f'transform="{_placement(verts, spot, count, bed)}" printable="1"/>'
        )
        configs.append((object_id, mesh_id, name, len(faces)))
    root = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<model unit="millimeter" xml:lang="en-US" xmlns="{CORE}" '
        f'xmlns:BambuStudio="{BAMBU_NS}" xmlns:p="{PROD}" requiredextensions="p">\n'
        # Bambu Studio only reads plates and settings from files that name it here.
        f' <metadata name="Application">BambuStudio-{_escape(version)}</metadata>\n'
        ' <metadata name="BambuStudio:3mfVersion">1</metadata>\n'
        f' <metadata name="Title">{_escape(title)}</metadata>\n'
        " <resources>\n" + "\n".join(resources) + "\n </resources>\n"
        f' <build p:UUID="{uuid.uuid4()}">\n' + "\n".join(items) + "\n </build>\n</model>\n"
    )
    rels = "\n".join(
        f' <Relationship Target="/{name}" Id="rel-{i}" Type="{MODEL_REL}"/>'
        for i, name in enumerate(files, start=1)
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
            ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
            ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
            ' <Default Extension="png" ContentType="image/png"/>\n'
            "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<Relationships xmlns="{REL}">\n'
            f' <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="{MODEL_REL}"/>\n'
            "</Relationships>",
        )
        archive.writestr("3D/3dmodel.model", root)
        archive.writestr(
            "3D/_rels/3dmodel.model.rels",
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<Relationships xmlns="{REL}">\n{rels}\n</Relationships>',
        )
        for name, body in files.items():
            archive.writestr(name, body)
        archive.writestr("Metadata/project_settings.config", json.dumps(settings, indent=4))
        archive.writestr("Metadata/model_settings.config", _model_settings(configs, layout, count))
        archive.writestr(
            "Metadata/slice_info.config",
            '<?xml version="1.0" encoding="UTF-8"?>\n<config>\n  <header>\n'
            '    <header_item key="X-BBL-Client-Type" value="slicer"/>\n'
            f'    <header_item key="X-BBL-Client-Version" value="{_escape(version)}"/>\n'
            "  </header>\n</config>\n",
        )
    return count


def _placement(verts, spot, count, bed):
    plate, x, y, turned = spot
    origin = plate_origin(plate, count, bed)
    low, high = verts.min(axis=0), verts.max(axis=0)
    mx, my = (low[0] + high[0]) / 2, (low[1] + high[1]) / 2
    if turned:
        # Row-vector 3MF matrix; a turn sends +X to +Y.
        matrix = "0 1 0 -1 0 0 0 0 1"
        mx, my = -my, mx
    else:
        matrix = "1 0 0 0 1 0 0 0 1"
    return f"{matrix} {origin[0] + x - mx:.4f} {origin[1] + y - my:.4f} {-low[2]:.4f}"


def _mesh_model(mesh_id, verts, faces):
    vertices = "\n".join(f'     <vertex x="{v[0]:.5f}" y="{v[1]:.5f}" z="{v[2]:.5f}"/>' for v in verts)
    triangles = "\n".join(f'     <triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}"/>' for f in faces)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<model unit="millimeter" xml:lang="en-US" xmlns="{CORE}" '
        f'xmlns:BambuStudio="{BAMBU_NS}" xmlns:p="{PROD}" requiredextensions="p">\n'
        ' <metadata name="BambuStudio:3mfVersion">1</metadata>\n'
        " <resources>\n"
        f'  <object id="{mesh_id}" p:UUID="{uuid.uuid4()}" type="model">\n'
        "   <mesh>\n    <vertices>\n" + vertices + "\n    </vertices>\n"
        "    <triangles>\n" + triangles + "\n    </triangles>\n   </mesh>\n  </object>\n"
        " </resources>\n <build/>\n</model>\n"
    )


def _model_settings(configs, layout, count):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<config>"]
    for object_id, mesh_id, name, faces in configs:
        safe = _escape(name)
        lines += [
            f'  <object id="{object_id}">',
            f'    <metadata key="name" value="{safe}"/>',
            '    <metadata key="extruder" value="1"/>',
            f'    <metadata face_count="{faces}"/>',
            f'    <part id="{mesh_id}" subtype="{SOLID_PART}">',
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
        for (object_id, _, _, _), spot in zip(configs, layout):
            if spot[0] == plate:
                lines += [
                    "    <model_instance>",
                    f'      <metadata key="object_id" value="{object_id}"/>',
                    '      <metadata key="instance_id" value="0"/>',
                    f'      <metadata key="identify_id" value="{object_id}"/>',
                    "    </model_instance>",
                ]
        lines.append("  </plate>")
    lines.append("</config>")
    return "\n".join(lines) + "\n"


def _from_source(source, machine):
    if source is None or Path(source).suffix.lower() != ".3mf":
        return None
    try:
        with zipfile.ZipFile(source) as archive:
            settings = json.loads(archive.read("Metadata/project_settings.config"))
    except (KeyError, OSError, ValueError, zipfile.BadZipFile):
        return None
    if not isinstance(settings, dict) or settings.get("printer_settings_id") != machine:
        return None
    return settings


def _profile_root():
    override = os.environ.get("KRACKUP_BAMBU_PROFILES")
    for raw in ([override] if override else []) + list(PROFILE_DIRS):
        root = Path(os.path.expandvars(os.path.expanduser(raw)))
        if (root / "BBL" / "machine").is_dir():
            return root / "BBL"
    return None


def _from_profiles(root, machine):
    index = {}
    for kind in ("machine", "process", "filament"):
        index[kind] = {}
        for name in glob.glob(str(root / kind / "**" / "*.json"), recursive=True):
            try:
                data = json.loads(Path(name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and data.get("name"):
                index[kind][data["name"]] = data
    printer = _flatten(index["machine"], machine)
    process_name = printer.get("default_print_profile", "")
    filament_names = printer.get("default_filament_profile") or []
    filament_name = filament_names[0] if filament_names else ""
    process = _flatten(index["process"], process_name) if process_name in index["process"] else {}
    filament = _flatten(index["filament"], filament_name) if filament_name in index["filament"] else {}
    settings = {}
    for part in (printer, process, filament):
        settings.update({key: value for key, value in part.items() if key not in PROFILE_KEYS})
    settings.update(
        {
            "name": "project_settings",
            "from": "project",
            "printer_settings_id": machine,
            "print_settings_id": process_name,
            "filament_settings_id": [filament_name] if filament_name else [],
            "version": _version(root),
        }
    )
    return settings


def _flatten(index, name, seen=()):
    if name in seen or name not in index:
        return {}
    data = index[name]
    merged = _flatten(index, data["inherits"], seen + (name,)) if data.get("inherits") else {}
    for extra in data.get("include", []):
        merged.update(_flatten(index, extra, seen + (name,)))
    merged.update(data)
    return merged


def _version(root):
    try:
        return json.loads((root.parent / "BBL.json").read_text(encoding="utf-8")).get("version", "")
    except (OSError, ValueError):
        return ""
