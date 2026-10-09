"""The settings server only answers its own page."""

import json
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from krackup import ui


class LocalOnlyTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        saved = ui.CONFIG_PATH
        ui.CONFIG_PATH = self.root / "settings.json"
        self.addCleanup(setattr, ui, "CONFIG_PATH", saved)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def status(self, path, body=None, headers=None):
        data = None if body is None else body.encode()
        request = Request(self.base + path, data=data, headers=headers or {})
        try:
            with urlopen(request) as response:
                return response.status
        except HTTPError as error:
            error.close()
            return error.code

    def test_own_page_works(self):
        self.assertEqual(self.status("/api/status"), 200)
        body = json.dumps({"pitch": 90})
        self.assertEqual(self.status("/api/config", body, {"Content-Type": "application/json"}), 200)

    def test_other_host_name_is_refused(self):
        # DNS rebinding: evil.example resolves to 127.0.0.1.
        headers = {"Host": f"evil.example:{self.server.server_port}"}
        self.assertEqual(self.status(f"/api/browse?path={self.root}", headers=headers), 403)

    def test_post_from_another_site_is_refused(self):
        body = json.dumps({"input": "/nowhere.stl"})
        headers = {"Content-Type": "application/json", "Origin": "http://evil.example"}
        self.assertEqual(self.status("/api/run", body, headers), 403)
        headers = {"Content-Type": "application/json", "Sec-Fetch-Site": "cross-site"}
        self.assertEqual(self.status("/api/run", body, headers), 403)

    def test_simple_cross_site_post_is_refused(self):
        # A form or a no-cors fetch can only send text/plain without a preflight.
        body = json.dumps({"pitch": 1})
        self.assertEqual(self.status("/api/config", body, {"Content-Type": "text/plain"}), 415)
        self.assertFalse(ui.CONFIG_PATH.exists())

    def test_json_that_is_not_an_object_is_refused(self):
        self.assertEqual(self.status("/api/config", "[1, 2]", {"Content-Type": "application/json"}), 400)


class JobErrorTest(unittest.TestCase):
    def test_bad_mesh_is_reported_to_the_page(self):
        saved = ui.CONFIG_PATH
        with tempfile.TemporaryDirectory() as folder:
            ui.CONFIG_PATH = Path(folder) / "settings.json"
            self.addCleanup(setattr, ui, "CONFIG_PATH", saved)
            src = Path(folder) / "open.stl"
            # One loose triangle is not a solid.
            src.write_text(
                "solid t\nfacet normal 0 0 1\n outer loop\n  vertex 0 0 0\n"
                "  vertex 1 0 0\n  vertex 0 1 0\n endloop\nendfacet\nendsolid t\n"
            )
            ui.start_job({"input": str(src), "output": str(Path(folder) / "out")})
            deadline = time.time() + 30
            while ui.JOB.snapshot()["running"] and time.time() < deadline:
                time.sleep(0.05)
        snapshot = ui.JOB.snapshot()
        self.assertFalse(snapshot["running"])
        self.assertIn("not a solid", snapshot["error"])


if __name__ == "__main__":
    unittest.main()
