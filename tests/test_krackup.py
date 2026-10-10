"""Fits a bar on a P1S bed and puts a pentagon dowel on the cut."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import manifold3d as mf

from krackup.meshio import read_stl, write_stl
from krackup.pins import dowel_solid
from krackup.run import Stopped, krack
from krackup.solid import manifold_from


def _stl_volume(path):
    verts, faces = read_stl(path)
    return manifold_from(verts, faces).volume()


def _write_solid(path, solid):
    mesh = solid.to_mesh()
    verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    write_stl(path, verts, faces)


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
            manifest = krack(
                src, out, "p1s", 10.0, 0.1, 50.0, write_project=False, labels=False, log=lambda *_: None,
            )
            self.assertEqual(len(manifest["parts"]), 2)
            for part in manifest["parts"]:
                x, y, z = part["print_size_mm"]
                self.assertLessEqual(x, 241)
                self.assertLessEqual(y, 241)
                self.assertLessEqual(z, 245)
            # One cut, one joint. The outside walls are not joints.
            dowels = sum(item["count"] for item in manifest["dowels"])
            self.assertEqual(len(manifest["joints"]), 1)
            self.assertEqual(dowels, manifest["joints"][0]["dowels"])
            self.assertEqual(dowels, 2)
            # Only the two sockets per dowel were removed from the bar.
            radius = manifest["dowels"][0]["radius_mm"]
            socket = dowel_solid(radius + 0.1, 5.1).volume()
            total = sum(_stl_volume(out / part["file"]) for part in manifest["parts"])
            self.assertAlmostEqual(total, bar.volume() - 2 * dowels * socket, delta=50.0)
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


    def test_flat_base_model_only_gets_dowels_on_its_cut(self):
        # A tall post with a big flat base, the usual statue shape.
        post = mf.Manifold.cube((60, 60, 420)) + mf.Manifold.cube((160, 160, 12)).translate((-50, -50, 0))
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / "post.stl"
            _write_solid(src, post)
            manifest = krack(src, Path(folder) / "out", "p1s", 10.0, 0.1, 100.0, False, log=lambda *_: None)
        dowels = sum(item["count"] for item in manifest["dowels"])
        self.assertEqual(dowels, sum(joint["dowels"] for joint in manifest["joints"]))
        for joint in manifest["joints"]:
            self.assertNotEqual(joint["reason"], "cut face in the file")

    def test_small_island_is_kept(self):
        bar = mf.Manifold.cube((300, 50, 40), center=True)
        crumb = mf.Manifold.cube((4, 4, 4)).translate((200, 0, 0))
        messages = []
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / "bar.stl"
            _write_solid(src, bar + crumb)
            manifest = krack(src, Path(folder) / "out", "p1s", 10.0, 0.1, 100.0, False, log=messages.append)
        self.assertEqual(len(manifest["parts"]), 3)
        self.assertTrue(any("small piece" in line for line in messages))

    def test_assembly_text_uses_the_settings(self):
        bar = mf.Manifold.cube((300, 50, 40), center=True)
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / "bar.stl"
            out = Path(folder) / "out"
            _write_solid(src, bar)
            krack(src, out, "p1s", 12.0, 0.2, 100.0, False, log=lambda *_: None)
            text = (out / "ASSEMBLY.txt").read_text()
            names = [path.name for path in (out / "dowels").glob("*.stl")]
        self.assertIn("12 mm long", text)
        self.assertIn("0.2 mm larger in radius", text)
        self.assertNotIn("10 mm unless", text)
        self.assertTrue(all(name.endswith("_L12.stl") for name in names))

    def test_failed_socket_drops_its_dowel(self):
        from krackup import pins

        bar = mf.Manifold.cube((300, 50, 40), center=True)
        messages = []
        real_cut = pins._cut
        pins._cut = lambda solid, tool: None
        self.addCleanup(setattr, pins, "_cut", real_cut)
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / "bar.stl"
            _write_solid(src, bar)
            manifest = krack(src, Path(folder) / "out", "p1s", 10.0, 0.1, 100.0, False, log=messages.append)
        self.assertEqual(manifest["dowels"], [])
        self.assertTrue(any("could not be cut" in line for line in messages))


class BedTest(unittest.TestCase):
    def test_bed_text_is_parsed(self):
        from krackup.printers import parse_bed

        self.assertEqual(parse_bed("300x320x250"), (300.0, 320.0, 250.0))
        self.assertEqual(parse_bed("300 × 300 × 300"), (300.0, 300.0, 300.0))
        for bad in ("300x300", "axbxc", "0x300x300"):
            with self.assertRaises(ValueError):
                parse_bed(bad)

    def test_custom_bed_and_margins_set_the_usable_box(self):
        slab = mf.Manifold.cube((290, 290, 40), center=True)
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / "slab.stl"
            _write_solid(src, slab)
            manifest = krack(
                src, Path(folder) / "out", "p1s", 10.0, 0.1, 100.0, False,
                bed=(320.0, 320.0, 300.0), margin_xy=10.0, margin_z=0.0, log=lambda *_: None,
            )
        self.assertEqual(manifest["printer"], "Custom")
        self.assertEqual(manifest["usable_mm"], [310.0, 310.0, 300.0])
        self.assertEqual(manifest["margin_xy_mm"], 10.0)
        self.assertEqual(len(manifest["parts"]), 1)

    def test_margins_that_leave_no_room_are_an_error(self):
        from krackup.solid import KrackError

        bar = mf.Manifold.cube((300, 50, 40), center=True)
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / "bar.stl"
            _write_solid(src, bar)
            with self.assertRaises(KrackError):
                krack(src, Path(folder) / "out", "mini", 10.0, 0.1, 100.0, False,
                      margin_xy=120.0, log=lambda *_: None)

    def test_long_part_turns_to_the_long_side_of_the_bed(self):
        # MK4 usable is 234 x 194. The slab is 230 along Y, so it has to turn.
        slab = mf.Manifold.cube((180, 230, 20), center=True)
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / "slab.stl"
            _write_solid(src, slab)
            manifest = krack(src, Path(folder) / "out", "mk4", 10.0, 0.1, 100.0, False, log=lambda *_: None)
        self.assertEqual(len(manifest["parts"]), 1)
        x, y, _ = manifest["parts"][0]["print_size_mm"]
        self.assertLessEqual(x, 234.4)
        self.assertLessEqual(y, 194.4)


class ProjectPlatesTest(unittest.TestCase):
    def test_project_holds_parts_and_dowels_on_plates(self):
        import zipfile

        bar = mf.Manifold.cube((300, 50, 40), center=True)
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / "bar.stl"
            out = Path(folder) / "out"
            _write_solid(src, bar)
            manifest = krack(src, out, "p1s", 10.0, 0.1, 100.0, True, log=lambda *_: None)
            names = zipfile.ZipFile(out / "project.3mf").read("3D/3dmodel.model").decode()
            text = (out / "ASSEMBLY.txt").read_text()
        dowels = sum(item["count"] for item in manifest["dowels"])
        self.assertEqual(names.count("<object "), len(manifest["parts"]) + dowels)
        self.assertEqual(manifest["plates"], 1)
        self.assertIn("on 1 plate.", text)


class LabelTest(unittest.TestCase):
    def test_each_joint_face_gets_own_and_mate_number(self):
        from krackup.labels import DEPTH, HEIGHTS, text_outline

        bar = mf.Manifold.cube((300, 50, 40), center=True)
        volumes = {}
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / "bar.stl"
            _write_solid(src, bar)
            for labels in (False, True):
                out = Path(folder) / f"out-{labels}"
                manifest = krack(src, out, "p1s", 10.0, 0.1, 100.0, False, labels=labels, log=lambda *_: None)
                volumes[labels] = [_stl_volume(out / part["file"]) for part in manifest["parts"]]
            text = (out / "ASSEMBLY.txt").read_text()
        report = manifest["labels"]
        self.assertEqual(report["labeled_faces"], 2)
        self.assertEqual(report["unlabeled_faces"], [])
        self.assertEqual(report["parts_without_joints"], [])
        self.assertIn("3-5 is on part 3", text)
        # Part 1 reads 1-2 and part 2 reads 2-1, DEPTH deep, at one of the
        # label heights.
        for part, (plain, marked) in enumerate(zip(volumes[False], volumes[True]), start=1):
            outline = f"{part}-{3 - part}"
            smallest = text_outline(outline, min(HEIGHTS)).area * DEPTH
            largest = text_outline(outline, max(HEIGHTS)).area * DEPTH
            self.assertGreater(plain - marked, 0.75 * smallest)
            self.assertLess(plain - marked, 1.25 * largest)

    def test_label_reads_from_outside_the_face(self):
        from shapely.geometry import Point

        from krackup.labels import cut_faces, label_joints
        from krackup.pins import _section_shape
        from krackup.split import Piece

        bar = mf.Manifold.cube((200, 60, 40), center=True)
        pos, neg = bar.split_by_plane([1.0, 0.0, 0.0], 0.0)
        pieces = [Piece(pos, [0]), Piece(neg, [0])]
        cut = {"id": 0, "normal": [1.0, 0.0, 0.0], "offset": 0.0}
        ids = {id(pieces[0]): 1, id(pieces[1]): 7}
        faces = cut_faces(pieces, [cut])
        self.assertEqual(len(faces), 2)
        labeled, short, missing = label_joints(faces, ids, log=lambda *_: None)
        self.assertEqual((labeled, short, missing), (2, [], []))
        frame = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        for piece, side in zip(pieces, (1.0, -1.0)):
            label = _section_shape(piece.solid, frame, side * 2.0).difference(
                _section_shape(piece.solid, frame, side * 0.3)
            )
            minx, miny, maxx, maxy = label.bounds
            # "1-7" and "7-1": the left edge has no upper stroke (1 is its
            # right strokes, 7 has no f), the right edge has one. Seen from
            # outside, "right" is -Y on the +X half and +Y on the -X half.
            near_min = minx + 0.02 * (maxx - minx)
            near_max = maxx - 0.02 * (maxx - minx)
            right, left = (near_min, near_max) if side > 0 else (near_max, near_min)
            middle = miny + 0.7 * (maxy - miny)
            self.assertTrue(label.contains(Point(right, middle)))
            self.assertFalse(label.contains(Point(left, middle)))

    def test_face_meeting_two_parts_gets_a_label_for_each(self):
        from krackup.labels import cut_faces, label_joints
        from krackup.split import Piece

        slab = mf.Manifold.cube((200, 80, 30)).translate((-100, -40, 0))
        left = mf.Manifold.cube((80, 80, 30)).translate((-100, -40, -30))
        right = mf.Manifold.cube((80, 80, 30)).translate((20, -40, -30))
        pieces = [Piece(slab, [0]), Piece(left, [0]), Piece(right, [0])]
        cut = {"id": 0, "normal": [0.0, 0.0, 1.0], "offset": 0.0}
        ids = {id(piece): number for number, piece in enumerate(pieces, start=1)}
        faces = cut_faces(pieces, [cut])
        pairs = sorted((ids[id(face["piece"])], ids[id(face["mate"])]) for face in faces)
        self.assertEqual(pairs, [(1, 2), (1, 3), (2, 1), (3, 1)])
        labeled, short, missing = label_joints(faces, ids, log=lambda *_: None)
        self.assertEqual((labeled, short, missing), (4, [], []))

    def test_small_face_falls_back_to_own_number(self):
        from krackup.labels import cut_faces, label_joints
        from krackup.split import Piece

        # 20 x 20 has room for "12" at 5 mm but not "12-34".
        post = mf.Manifold.cube((20, 20, 60), center=True)
        top, bottom = post.split_by_plane([0.0, 0.0, 1.0], 0.0)
        pieces = [Piece(top, [0]), Piece(bottom, [0])]
        cut = {"id": 0, "normal": [0.0, 0.0, 1.0], "offset": 0.0}
        ids = {id(pieces[0]): 12, id(pieces[1]): 34}
        labeled, short, missing = label_joints(cut_faces(pieces, [cut]), ids, log=lambda *_: None)
        self.assertEqual(labeled, 0)
        self.assertEqual(sorted(short), [[12, 34], [34, 12]])
        self.assertEqual(missing, [])


class OrientTest(unittest.TestCase):
    def test_forty_degree_overhang_needs_support(self):
        from krackup.orient import _support

        # One triangle facing down, tilted 40 degrees from straight down.
        tilt = np.radians(40.0)
        verts = np.array(
            [[0, 0, 10], [0, 10, 10], [10 * np.cos(tilt), 0, 10 + 10 * np.sin(tilt)]],
            dtype=np.float64,
        )
        overhang, _ = _support(verts, np.array([[0, 1, 2]]))
        self.assertGreater(overhang, 0.0)


class CliTest(unittest.TestCase):
    def test_bad_input_is_an_error_message_not_a_traceback(self):
        import contextlib
        import io

        from krackup.cli import main

        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / "model.obj"
            src.write_text("nope")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                code = main([str(src), "-o", str(Path(folder) / "out")])
        self.assertEqual(code, 1)
        self.assertIn("unsupported file type", err.getvalue())

    def test_default_output_keeps_dotted_folders(self):
        from unittest import mock

        from krackup import cli

        import contextlib
        import io

        with mock.patch.object(cli, "krack") as fake, contextlib.redirect_stdout(io.StringIO()):
            cli.main(["/tmp/v1.2/model.stl"])
        self.assertEqual(fake.call_args.args[1], "/tmp/v1.2/model-krackup")


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
            saved_config = ui.CONFIG_PATH
            ui.CONFIG_PATH = root / "settings.json"
            self.addCleanup(setattr, ui, "CONFIG_PATH", saved_config)
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
