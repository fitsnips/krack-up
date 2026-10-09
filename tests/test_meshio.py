"""STL and 3MF reading and writing."""

import tempfile
import unittest
import zipfile
from pathlib import Path

import manifold3d as mf
import numpy as np

from krackup.meshio import load_models, read_stl, write_3mf, write_stl
from krackup.solid import manifold_from

CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
PROD = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"


def _arrays(solid):
    mesh = solid.to_mesh()
    return (
        np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3],
        np.asarray(mesh.tri_verts, dtype=np.int64),
    )


def _mesh_xml(object_id, solid, extra=""):
    verts, faces = _arrays(solid)
    vs = "".join(f'<vertex x="{v[0]}" y="{v[1]}" z="{v[2]}"/>' for v in verts)
    ts = "".join(f'<triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}"/>' for f in faces)
    return (
        f'<object id="{object_id}" type="model"{extra}><mesh><vertices>{vs}</vertices>'
        f"<triangles>{ts}</triangles></mesh></object>"
    )


def _write_3mf(path, model, extra_files=None):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>',
        )
        archive.writestr("3D/3dmodel.model", model)
        for name, text in (extra_files or {}).items():
            archive.writestr(name, text)


class ThreeMfTest(unittest.TestCase):
    def test_a_plain_cube_is_loaded(self):
        # Eight vertices. The old loader skipped any mesh under 50.
        model = (
            f'<model unit="millimeter" xmlns="{CORE}"><resources>'
            + _mesh_xml(1, mf.Manifold.cube((20, 30, 40)))
            + '</resources><build><item objectid="1"/></build></model>'
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cube.3mf"
            _write_3mf(path, model)
            loaded = load_models(path)
        self.assertEqual(len(loaded), 1)
        _, verts, _ = loaded[0]
        np.testing.assert_allclose(verts.max(axis=0) - verts.min(axis=0), [20, 30, 40])

    def test_units_are_converted_to_millimeters(self):
        model = (
            f'<model unit="inch" xmlns="{CORE}"><resources>'
            + _mesh_xml(1, mf.Manifold.cube((1, 2, 3)))
            + '</resources><build><item objectid="1"/></build></model>'
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "inch.3mf"
            _write_3mf(path, model)
            _, verts, _ = load_models(path)[0]
        np.testing.assert_allclose(verts.max(axis=0) - verts.min(axis=0), [25.4, 50.8, 76.2])

    def test_support_objects_are_skipped(self):
        model = (
            f'<model unit="millimeter" xmlns="{CORE}"><resources>'
            + _mesh_xml(1, mf.Manifold.cube((20, 20, 20)))
            + _mesh_xml(2, mf.Manifold.cube((5, 5, 5)), ' type="support"').replace(' type="model"', "", 1)
            + '</resources><build><item objectid="1"/><item objectid="2"/></build></model>'
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "support.3mf"
            _write_3mf(path, model)
            loaded = load_models(path)
        self.assertEqual(len(loaded), 1)

    def test_bambu_negative_part_is_cut_and_modifier_is_ignored(self):
        body = mf.Manifold.cube((60, 60, 60))
        # A connector socket, the way Bambu's cut tool leaves it.
        socket = mf.Manifold.cylinder(10, 5, 5, 32).translate((30, 30, 50))
        modifier = mf.Manifold.cube((100, 100, 5)).translate((-20, -20, -2))
        model = (
            f'<model unit="millimeter" xmlns="{CORE}" xmlns:p="{PROD}"><resources>'
            + _mesh_xml(1, body)
            # Bambu writes every part but the body as type="other".
            + _mesh_xml(2, socket).replace('type="model"', 'type="other"', 1)
            + _mesh_xml(3, modifier).replace('type="model"', 'type="other"', 1)
            + '<object id="4" type="model"><components>'
            '<component objectid="1"/><component objectid="2"/><component objectid="3"/>'
            "</components></object>"
            '</resources><build><item objectid="4"/></build></model>'
        )
        settings = (
            '<?xml version="1.0"?><config><object id="4">'
            '<metadata key="name" value="Body"/>'
            '<part id="1" subtype="normal_part"/>'
            '<part id="2" subtype="negative_part"/>'
            '<part id="3" subtype="modifier_part"/>'
            "</object></config>"
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bambu.3mf"
            _write_3mf(path, model, {"Metadata/model_settings.config": settings})
            loaded = load_models(path)
        self.assertEqual(len(loaded), 1)
        name, verts, faces = loaded[0]
        self.assertEqual(name, "Body")
        solid = manifold_from(verts, faces)
        self.assertAlmostEqual(solid.volume(), body.volume() - (body ^ socket).volume(), delta=1.0)
        np.testing.assert_allclose(verts.min(axis=0), [0, 0, 0], atol=1e-6)

    def test_rotated_component_follows_the_3mf_convention(self):
        # 3MF applies transforms to row vectors. This one turns +X into +Y,
        # then moves 100 mm along X.
        bar = mf.Manifold.cube((40, 2, 2))
        model = (
            f'<model unit="millimeter" xmlns="{CORE}"><resources>'
            + _mesh_xml(1, bar)
            + '<object id="2" type="model"><components>'
            '<component objectid="1" transform="0 1 0 -1 0 0 0 0 1 100 0 0"/>'
            "</components></object>"
            '</resources><build><item objectid="2"/></build></model>'
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "turned.3mf"
            _write_3mf(path, model)
            _, verts, _ = load_models(path)[0]
        np.testing.assert_allclose(verts.min(axis=0), [98, 0, 0], atol=1e-6)
        np.testing.assert_allclose(verts.max(axis=0), [100, 40, 2], atol=1e-6)

    def test_written_project_parts_do_not_overlap(self):
        small = _arrays(mf.Manifold.cube((20, 20, 20), center=True))
        large = _arrays(mf.Manifold.cube((120, 20, 20), center=True))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "project.3mf"
            write_3mf(path, [("small", *small), ("large", *large), ("small2", *small)])
            loaded = load_models(path)
        spans = sorted((v[:, 0].min(), v[:, 0].max()) for _, v, _ in loaded)
        for (_, left_end), (right_start, _) in zip(spans, spans[1:]):
            self.assertGreaterEqual(right_start - left_end, 9.99)


class StlTest(unittest.TestCase):
    def test_binary_stl_with_solid_header(self):
        verts, faces = _arrays(mf.Manifold.cube((10, 10, 10)))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cube.stl"
            write_stl(path, verts, faces)
            data = bytearray(path.read_bytes())
            # SolidWorks and others start binary headers with "solid".
            data[:80] = b"solid exported by some cad tool".ljust(80, b" ")
            path.write_bytes(bytes(data))
            read_verts, read_faces = read_stl(path)
        self.assertEqual(len(read_faces), len(faces))
        np.testing.assert_allclose(read_verts.max(axis=0), [10, 10, 10])

    def test_ascii_stl(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tri.stl"
            path.write_text(
                "solid t\nfacet normal 0 0 1\n outer loop\n  vertex 0 0 0\n"
                "  vertex 1 0 0\n  vertex 0 1 0\n endloop\nendfacet\nendsolid t\n"
            )
            verts, faces = read_stl(path)
        self.assertEqual(faces.shape, (1, 3))
        np.testing.assert_allclose(verts.max(axis=0), [1, 1, 0])


if __name__ == "__main__":
    unittest.main()
