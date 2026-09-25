"""Fits a bar on a P1S bed and puts a pentagon dowel on the cut."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import manifold3d as mf

from krackup.meshio import write_stl
from krackup.pins import dowel_solid
from krackup.run import Stopped, krack


class KrackUpTest(unittest.TestCase):
    def test_pentagon_dowel_matches_jimmy_size(self):
        solid = dowel_solid(10.0, 10.0)
        mesh = solid.to_mesh()
        verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
        center = verts.mean(axis=0)
        radial = np.linalg.norm((verts - center)[:, :2], axis=1)
        self.assertAlmostEqual(float(radial.max()), 10.0, delta=0.05)
        self.assertAlmostEqual(float(verts[:, 2].max() - verts[:, 2].min()), 10.0, delta=0.05)
        self.assertGreater(solid.volume(), 200.0)

    def test_bar_is_cut_to_the_bed_with_a_dowel(self):
        bar = mf.Manifold.cube((300, 50, 40), center=True)
        mesh = bar.to_mesh()
        verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
        faces = np.asarray(mesh.tri_verts, dtype=np.int64)
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / "bar.stl"
            out = Path(folder) / "out"
            write_stl(src, verts, faces)
            manifest = krack(src, out, "p1s", 10.0, 0.1, 50.0, write_project=False, log=lambda *_: None)
            self.assertGreaterEqual(len(manifest["parts"]), 2)
            for part in manifest["parts"]:
                x, y, z = part["print_size_mm"]
                self.assertLessEqual(x, 241)
                self.assertLessEqual(y, 241)
                self.assertLessEqual(z, 245)
            dowels = sum(item["count"] for item in manifest["dowels"])
            self.assertGreaterEqual(dowels, 2)
            self.assertEqual(manifest["dowel"]["pitch_mm"], 50.0)
            self.assertEqual(manifest["dowel"]["min_pins_per_plane"], 2)
            self.assertTrue((out / "ASSEMBLY.txt").exists())
            self.assertTrue(list((out / "dowels").glob("pentagon_*.stl")))

    def test_scale_shrinks_the_part_and_stop_aborts(self):
        bar = mf.Manifold.cube((200, 40, 30), center=True)
        mesh = bar.to_mesh()
        verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
        faces = np.asarray(mesh.tri_verts, dtype=np.int64)
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / "bar.stl"
            out = Path(folder) / "out"
            write_stl(src, verts, faces)
            with self.assertRaises(Stopped):
                krack(
                    src, out, "p1s", 10.0, 0.1, 100.0, False,
                    should_stop=lambda: True, log=lambda *_: None,
                )
            manifest = krack(
                src, out, "p1s", 10.0, 0.1, 100.0, False,
                scale=0.5, log=lambda *_: None,
            )
        longest = max(max(part["print_size_mm"]) for part in manifest["parts"])
        self.assertLess(longest, 120)
        self.assertEqual(manifest["dowel"]["scale"], 0.5)


class DimensionsTest(unittest.TestCase):
    def test_original_size_comes_from_the_file(self):
        from krackup.ui import model_dimensions

        text = """solid t
facet normal 0 0 1
 outer loop
  vertex 0 0 0
  vertex 10 0 0
  vertex 0 20 5
 endloop
endfacet
endsolid t
"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tiny.stl"
            path.write_text(text)
            size = model_dimensions(path)["parts"][0]["size"]
        self.assertEqual(size, [10.0, 20.0, 5.0])


class SettingsUiTest(unittest.TestCase):
    def test_file_browser_lists_models_and_saves_settings(self):
        import json
        import threading
        from http.server import ThreadingHTTPServer
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen

        from krackup import ui
        from krackup.ui import Handler

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model.stl").write_bytes(b"solid x\nendsolid x\n")
            (root / "notes.txt").write_text("skip")
            ui.CONFIG_PATH = root / "settings.json"
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_port
            try:
                listing = json.load(urlopen(f"http://127.0.0.1:{port}/api/browse?path={root}"))
                names = [item["name"] for item in listing["entries"]]
                self.assertIn("model.stl", names)
                self.assertTrue(next(item for item in listing["entries"] if item["name"] == "model.stl")["model"])
                body = json.dumps({"pitch": 80, "min_pins": 3, "browse": str(root)}).encode()
                request = Request(
                    f"http://127.0.0.1:{port}/api/config",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                saved = json.load(urlopen(request))
                self.assertEqual(saved["settings"]["pitch"], 80)
                self.assertEqual(saved["settings"]["min_pins"], 3)
                opened = []
                ui_open = ui.open_folder
                ui.open_folder = lambda raw: opened.append(raw) or str(root)
                opened_body = json.dumps({"output": str(root)}).encode()
                opened_request = Request(
                    f"http://127.0.0.1:{port}/api/open-output",
                    data=opened_body,
                    headers={"Content-Type": "application/json"},
                )
                opened_res = json.load(urlopen(opened_request))
                self.assertEqual(opened, [str(root)])
                self.assertEqual(opened_res["path"], str(root))
                missing = Request(
                    f"http://127.0.0.1:{port}/api/open-output",
                    data=json.dumps({"output": str(root / "missing")}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                ui.open_folder = ui_open
                with self.assertRaises(HTTPError) as missing_open:
                    urlopen(missing)
                missing_open.exception.close()
                self.assertEqual(missing_open.exception.code, 400)
            finally:
                ui.open_folder = ui_open
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
