"""Joints between objects that were already cut apart in the file."""

import unittest

import manifold3d as mf
import numpy as np

from krackup.faces import pin_existing_faces
from krackup.geom import rot_from_to
from krackup.pins import dowel_solid
from krackup.split import Piece


def _pose():
    """A rigid move, like an object laid out somewhere else on the plate."""
    rotation = rot_from_to([0.2, -0.5, 0.84], [0.0, 0.0, 1.0]) @ rot_from_to([1, 0, 0], [0.6, 0.8, 0])
    return rotation, np.array([310.0, -140.0, 25.0])


def _move(solid, rotation, shift):
    return solid.transform(np.column_stack([rotation, shift]))


def _halves(solid, height, holes=()):
    """Cut `solid` at z = height, like Bambu's cut tool, then lay B out elsewhere.

    `holes` are (x, y) spots where both halves already have a socket.
    """
    above, below = solid.split_by_plane([0.0, 0.0, 1.0], height)
    for x, y in holes:
        tool = dowel_solid(7.6, 10.4).translate((x, y, height))
        above = above - tool
        below = below - tool
    rotation, shift = _pose()
    a = Piece(below, [])
    a.origin, a.source = 0, "A"
    b = Piece(_move(above, rotation, shift), [])
    b.origin, b.source = 1, "B"
    return a, b, rotation, shift


def _hole_centers(pins, index, rotation=None, shift=None):
    centers = []
    for pin in pins:
        hole = pin["holes"][index]
        point = hole["frame"].T @ np.array([hole["x"], hole["y"], hole["offset"]])
        if rotation is not None:
            point = rotation.T @ (point - shift)
        centers.append(point)
    return np.array(centers)


def _prism(outline, height):
    """A block that narrows toward the top, like most statue parts.

    A straight prism has the same outline top and bottom, so its end walls
    would look like a joint too.
    """
    outline = np.asarray(outline, dtype=np.float64)
    center = outline.mean(axis=0)
    solid = mf.CrossSection([outline - center]).extrude(height, 0, 0.0, (0.55, 0.55))
    return solid.translate((center[0], center[1], 0.0))


class FileJointTest(unittest.TestCase):
    def _run(self, a, b, min_pins=2, pitch=100.0):
        log = []
        pins, _ = pin_existing_faces([a, b], 10.0, 0.1, pitch, min_pins, log=log.append)
        return pins, log

    def assertSocketsLineUp(self, pins, rotation, shift):
        on_a = _hole_centers(pins, 0)
        on_b = _hole_centers(pins, 1, rotation, shift)
        # Every hole on A has a hole on B at the same spot once B is put back.
        for point in on_a:
            self.assertLess(float(np.min(np.linalg.norm(on_b - point, axis=1))), 0.5)
        for point in on_b:
            self.assertLess(float(np.min(np.linalg.norm(on_a - point, axis=1))), 0.5)

    def test_asymmetric_face_holes_line_up(self):
        outline = [(0, 0), (140, 0), (140, 45), (50, 45), (50, 110), (0, 110)]
        a, b, rotation, shift = _halves(_prism(outline, 90), 40.0)
        pins, _ = self._run(a, b)
        self.assertGreaterEqual(len(pins), 2)
        self.assertSocketsLineUp(pins, rotation, shift)
        for pin in pins:
            self.assertIs(pin["positive"], a)
            self.assertIs(pin["negative"], b)

    def test_rectangle_face_holes_line_up_either_way_round(self):
        a, b, rotation, shift = _halves(_prism([(0, 0), (130, 0), (130, 70), (0, 70)], 90), 45.0)
        pins, _ = self._run(a, b)
        self.assertGreaterEqual(len(pins), 2)
        self.assertSocketsLineUp(pins, rotation, shift)

    def test_square_face_holes_line_up_any_quarter_turn(self):
        a, b, rotation, shift = _halves(_prism([(0, 0), (90, 0), (90, 90), (0, 90)], 90), 45.0)
        pins, _ = self._run(a, b)
        self.assertGreaterEqual(len(pins), 2)
        self.assertSocketsLineUp(pins, rotation, shift)
        # The symmetric layout still leaves room between neighbouring sockets.
        centers = _hole_centers(pins, 0)
        for i, one in enumerate(centers):
            for two in centers[i + 1 :]:
                self.assertGreater(float(np.linalg.norm(one - two)), 2 * 10.1)

    def test_round_face_gets_one_centered_dowel(self):
        a, b, rotation, shift = _halves(mf.Manifold.cylinder(90, 50, 28, 96), 45.0)
        pins, log = self._run(a, b)
        self.assertEqual(len(pins), 1)
        self.assertSocketsLineUp(pins, rotation, shift)
        self.assertTrue(any("round face" in line for line in log))

    def test_pieces_of_one_object_are_not_paired(self):
        a, b, _, _ = _halves(_prism([(0, 0), (130, 0), (130, 70), (0, 70)], 90), 45.0)
        b.origin = a.origin
        pins, _ = self._run(a, b)
        self.assertEqual(pins, [])

    def test_face_that_already_has_dowels_is_left_alone(self):
        outline = [(0, 0), (140, 0), (140, 70), (0, 70)]
        a, b, _, _ = _halves(_prism(outline, 90), 45.0, holes=[(35, 35), (105, 35)])
        before = a.solid.volume(), b.solid.volume()
        pins, _ = self._run(a, b)
        self.assertEqual(pins, [])
        self.assertEqual((a.solid.volume(), b.solid.volume()), before)

    def test_face_with_one_dowel_is_topped_up(self):
        outline = [(0, 0), (140, 0), (140, 70), (0, 70)]
        a, b, rotation, shift = _halves(_prism(outline, 90), 45.0, holes=[(35, 35)])
        pins, _ = self._run(a, b)
        self.assertGreaterEqual(len(pins), 1)
        self.assertSocketsLineUp(pins, rotation, shift)
        # New holes keep clear of the one that was there.
        for x, y, _ in _hole_centers(pins, 0):
            self.assertGreater(np.hypot(x - 35, y - 35), 15)

    def test_krack_up_cut_faces_are_skipped(self):
        a, b, _, _ = _halves(_prism([(0, 0), (130, 0), (130, 70), (0, 70)], 90), 45.0)
        a.planes = [((0.0, 0.0, 1.0), 45.0)]
        pins, _ = self._run(a, b)
        self.assertEqual(pins, [])


if __name__ == "__main__":
    unittest.main()


class SameCutTest(unittest.TestCase):
    def test_halves_of_one_bambu_cut_pair_first(self):
        from krackup import faces

        class Fake:
            def __init__(self, origin, cut_id):
                self.origin = origin
                self.cut_id = cut_id

        a, b, c = Fake(0, 7), Fake(1, 7), Fake(2, 9)
        face = {name: {"piece": piece, "area": 1000.0} for name, piece in zip("abc", (a, b, c))}
        scores = {frozenset("bc"): 0.95, frozenset("ab"): 0.85, frozenset("ac"): 0.85}
        name_of = {id(f): n for n, f in face.items()}
        real_shape, real_align = faces._shape, faces._align
        faces._shape = lambda f: {"name": name_of[id(f)]}
        faces._align = lambda sa, sb: {"score": scores[frozenset((sa["name"], sb["name"]))]}
        self.addCleanup(setattr, faces, "_shape", real_shape)
        self.addCleanup(setattr, faces, "_align", real_align)
        pairs = faces._match([face["a"], face["b"], face["c"]])
        self.assertEqual(len(pairs), 1)
        self.assertEqual({name_of[id(pairs[0][0])], name_of[id(pairs[0][1])]}, {"a", "b"})


class ScaledConnectorTest(unittest.TestCase):
    def test_connector_found_after_scaling_the_model(self):
        from krackup.faces import _cut_faces

        # A 15 mm deep socket: a 10 mm Bambu connector after scaling by 1.5.
        block = mf.Manifold.cube((120, 120, 60), center=True)
        socket = mf.Manifold.cylinder(16, 11.0, 11.0, 48).translate((0, 0, 15))
        piece = Piece(block - socket, [])
        top = lambda faces: max(faces, key=lambda f: f["normal"][2])
        self.assertEqual(top(_cut_faces(piece))["existing"], [])
        self.assertEqual(len(top(_cut_faces(piece, scale=1.5))["existing"]), 1)
