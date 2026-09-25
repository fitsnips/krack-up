"""Fits a bar on a P1S bed and puts a pentagon dowel on the cut."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import manifold3d as mf

from krackup.meshio import write_stl
from krackup.pins import dowel_solid
from krackup.run import krack


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


class SettingsUiTest(unittest.TestCase):
    def test_file_browser_lists_models_and_saves_settings(self):
        import json
        import threading
        from http.server import ThreadingHTTPServer
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
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
