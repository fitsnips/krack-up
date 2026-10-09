"""Plate packing and the Bambu plate grid."""

import random
import re
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

import manifold3d as mf

from krackup.meshio import load_models, write_3mf
from krackup.plates import GAP, pack, plate_origin

BED = (256.0, 256.0)
MARGIN = 8.0
CORNER = (0.0, 0.0, 18.0, 28.0)


def _rects(sizes, layout):
    for (w, d), (plate, x, y, turned) in zip(sizes, layout):
        if turned:
            w, d = d, w
        yield plate, (x - w / 2, y - d / 2, x + w / 2, y + d / 2)


class PackTest(unittest.TestCase):
    def test_parts_stay_on_the_bed_apart_and_out_of_the_corner(self):
        rng = random.Random(1)
        sizes = [(rng.uniform(10, 235), rng.uniform(10, 235)) for _ in range(40)]
        layout = pack(sizes, BED, MARGIN, exclude=[CORNER])
        by_plate = {}
        for plate, rect in _rects(sizes, layout):
            x0, y0, x1, y1 = rect
            self.assertGreaterEqual(x0, MARGIN - 1e-6)
            self.assertGreaterEqual(y0, MARGIN - 1e-6)
            self.assertLessEqual(x1, BED[0] - MARGIN + 1e-6)
            self.assertLessEqual(y1, BED[1] - MARGIN + 1e-6)
            self.assertFalse(x0 < CORNER[2] and y0 < CORNER[3], rect)
            for other in by_plate.get(plate, []):
                apart = (
                    x1 + GAP <= other[0] + 1e-6 or other[2] + GAP <= x0 + 1e-6
                    or y1 + GAP <= other[1] + 1e-6 or other[3] + GAP <= y0 + 1e-6
                )
                self.assertTrue(apart, (rect, other))
            by_plate.setdefault(plate, []).append(rect)

    def test_small_parts_share_a_plate(self):
        layout = pack([(50, 50)] * 9, BED, MARGIN)
        self.assertEqual({spot[0] for spot in layout}, {0})

    def test_long_part_is_turned_to_fit_beside_another(self):
        layout = pack([(230, 100), (100, 120)], BED, MARGIN)
        self.assertEqual({spot[0] for spot in layout}, {0})

    def test_part_too_big_for_the_corner_gets_its_own_plate(self):
        layout = pack([(239, 237), (30, 30)], BED, MARGIN, exclude=[CORNER])
        self.assertNotEqual(layout[0][0], layout[1][0])
        self.assertAlmostEqual(layout[0][1], 128.0)


class GridTest(unittest.TestCase):
    def test_grid_matches_the_jimmy_project(self):
        # Twenty P1S plates sit five to a row, 307.2 mm apart, rows going -Y.
        self.assertEqual(plate_origin(0, 20, BED), (0.0, 0.0))
        self.assertEqual(plate_origin(4, 20, BED), (4 * 307.2, 0.0))
        np.testing.assert_allclose(plate_origin(6, 20, BED), (307.2, -307.2))
        np.testing.assert_allclose(plate_origin(4, 15, BED), (0.0, -307.2))


class ProjectTest(unittest.TestCase):
    def test_objects_land_on_their_plates(self):
        cube = mf.Manifold.cube((200, 200, 20), center=True).translate((0, 0, 10)).to_mesh()
        verts = np.asarray(cube.vert_properties, dtype=np.float64)[:, :3]
        faces = np.asarray(cube.tri_verts, dtype=np.int64)
        objects = [(f"part_{n}", verts, faces) for n in range(3)]
        layout = pack([(200, 200)] * 3, BED, MARGIN)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "project.3mf"
            write_3mf(path, objects, layout=layout, bed=BED)
            config = zipfile.ZipFile(path).read("Metadata/model_settings.config").decode()
            loaded = load_models(path)
        self.assertEqual(config.count("<plate>"), 3)
        for (plate, x, y, _), (_, placed, _) in zip(layout, loaded):
            origin = plate_origin(plate, 3, BED)
            middle = (placed.min(axis=0) + placed.max(axis=0)) / 2
            np.testing.assert_allclose(middle[:2], (origin[0] + x, origin[1] + y), atol=1e-3)
            self.assertAlmostEqual(placed[:, 2].min(), 0.0, places=3)
        plates = re.findall(r"<plate>(.*?)</plate>", config, re.S)
        for index, block in enumerate(plates):
            self.assertIn(f'value="{index + 1}"', block)
