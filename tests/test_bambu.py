"""Bambu Studio project output."""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

import manifold3d as mf

from krackup import bambu
from krackup.meshio import load_models
from krackup.plates import pack, plate_origin

BED = (256.0, 256.0)
SETTINGS = {"printer_settings_id": "Bambu Lab P1S 0.4 nozzle", "version": "02.08.02.61"}


def _cube(size):
    mesh = mf.Manifold.cube(size, center=True).translate((0, 0, size[2] / 2)).to_mesh()
    return (
        np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3],
        np.asarray(mesh.tri_verts, dtype=np.int64),
    )


class SettingsTest(unittest.TestCase):
    def test_settings_come_from_a_matching_input_project(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "in.3mf"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("Metadata/project_settings.config", json.dumps(SETTINGS))
            settings, where = bambu.project_settings("p1s", source)
            self.assertEqual(settings, SETTINGS)
            self.assertIn("in.3mf", where)
            # Another printer's project is not reused.
            found = bambu.project_settings("a1mini", source)
            self.assertTrue(found is None or "profiles" in found[1])

    def test_other_printers_get_no_bambu_project(self):
        self.assertIsNone(bambu.project_settings("mk4"))

    def test_profiles_are_flattened_with_includes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "BBL"
            for kind in ("machine", "process", "filament"):
                (root / kind).mkdir(parents=True)
            files = {
                "machine/base.json": {"name": "base", "printable_area": ["0x0", "200x200"], "nozzle_diameter": ["0.4"]},
                "machine/gcode.json": {"name": "gcode", "machine_start_gcode": "G28"},
                "machine/p1s.json": {
                    "name": "Bambu Lab P1S 0.4 nozzle", "inherits": "base", "include": ["gcode"],
                    "printable_area": ["0x0", "256x256"], "default_print_profile": "std",
                    "default_filament_profile": ["pla"], "type": "machine",
                },
                "process/std.json": {"name": "std", "layer_height": "0.2"},
                "filament/pla.json": {"name": "pla", "filament_type": ["PLA"]},
            }
            for name, body in files.items():
                (root / name).write_text(json.dumps(body))
            (Path(folder) / "BBL.json").write_text(json.dumps({"version": "02.08.00.01"}))
            settings = bambu._from_profiles(root, "Bambu Lab P1S 0.4 nozzle")
        self.assertEqual(settings["printable_area"], ["0x0", "256x256"])
        self.assertEqual(settings["machine_start_gcode"], "G28")
        self.assertEqual(settings["layer_height"], "0.2")
        self.assertEqual(settings["filament_settings_id"], ["pla"])
        self.assertEqual(settings["print_settings_id"], "std")
        self.assertEqual(settings["version"], "02.08.00.01")
        self.assertNotIn("inherits", settings)
        self.assertNotIn("type", settings)

    def test_brims_and_supports_widen_the_gap(self):
        self.assertEqual(bambu.spacing(None, 6.0), 6.0)
        self.assertEqual(bambu.spacing({"brim_type": "no_brim"}, 6.0), 6.0)
        self.assertEqual(bambu.spacing({"brim_type": "auto_brim", "brim_width": "5", "brim_object_gap": "0"}, 6.0), 12.0)
        self.assertEqual(
            bambu.spacing({"brim_type": "no_brim", "enable_support": "1"}, 6.0), 10.0
        )


class ProjectTest(unittest.TestCase):
    def test_project_has_plates_settings_and_loads_back(self):
        objects = [(f"part_{n}", *_cube((200, 200, 20 + n))) for n in range(3)]
        layout = pack([(200, 200)] * 3, BED, 8.0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "project.3mf"
            count = bambu.write_project(path, objects, layout, BED, SETTINGS, title="t")
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                root = archive.read("3D/3dmodel.model").decode()
                config = archive.read("Metadata/model_settings.config").decode()
                settings = json.loads(archive.read("Metadata/project_settings.config"))
            loaded = load_models(path)
        self.assertEqual(count, 3)
        self.assertIn("BambuStudio-02.08.02.61", root)
        self.assertEqual(settings, SETTINGS)
        self.assertEqual(config.count("<plate>"), 3)
        self.assertTrue({f"3D/Objects/object_{k}.model" for k in (1, 2, 3)} <= names)
        self.assertEqual(len(loaded), 3)
        for (plate, x, y, _), (_, verts, _) in zip(layout, loaded):
            origin = plate_origin(plate, 3, BED)
            middle = (verts.min(axis=0) + verts.max(axis=0)) / 2
            np.testing.assert_allclose(middle[:2], (origin[0] + x, origin[1] + y), atol=1e-3)
            self.assertAlmostEqual(verts[:, 2].min(), 0.0, places=3)
