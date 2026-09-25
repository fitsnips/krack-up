"""Local settings window: edit the cut options, pick a file, and run."""

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from krackup.printers import DEFAULT_PRINTER, PRINTERS
from krackup.run import krack

PAGE = Path(__file__).with_name("ui_page.html").read_text(encoding="utf-8")
CONFIG_PATH = Path.home() / ".config" / "krack-up" / "settings.json"
MODEL_SUFFIXES = {".stl", ".3mf"}

DEFAULTS = {
    "input": "",
    "output": "",
    "printer": DEFAULT_PRINTER,
    "pitch": 100.0,
    "min_pins": 2,
    "length": 10.0,
    "tolerance": 0.1,
    "write_3mf": False,
    "browse": str(Path.home() / "Downloads"),
}


class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.log = []
        self.error = ""
        self.output = ""

    def snapshot(self):
        with self.lock:
            return {
                "running": self.running,
                "log": "\n".join(self.log[-400:]),
                "error": self.error,
                "output": self.output,
            }


JOB = Job()


def load_settings():
    settings = dict(DEFAULTS)
    if CONFIG_PATH.is_file():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            saved = {}
        if isinstance(saved, dict):
            settings.update({key: saved[key] for key in DEFAULTS if key in saved})
    browse_path = Path(str(settings["browse"])).expanduser()
    if not browse_path.is_dir():
        downloads = Path.home() / "Downloads"
        settings["browse"] = str(downloads if downloads.is_dir() else Path.home())
    return settings


def save_settings(settings):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stored = {key: settings.get(key, DEFAULTS[key]) for key in DEFAULTS}
    CONFIG_PATH.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    return stored


def browse(raw_path):
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.home() / path
    path = path.resolve()
    if path.is_file():
        path = path.parent
    if not path.is_dir():
        raise FileNotFoundError(f"No folder at {path}")
    entries = []
    for child in path.iterdir():
        if child.name.startswith("."):
            continue
        try:
            is_dir = child.is_dir()
        except OSError:
            continue
        suffix = child.suffix.lower()
        entries.append(
            {
                "name": child.name,
                "path": str(child),
                "dir": is_dir,
                "model": suffix in MODEL_SUFFIXES and not is_dir,
                "size": "" if is_dir else _size(child),
            }
        )
    entries.sort(key=lambda item: (not item["dir"], item["name"].lower()))
    return {"path": str(path), "entries": entries}


def _size(path):
    try:
        count = path.stat().st_size
    except OSError:
        return ""
    units = ("B", "KB", "MB", "GB")
    size = float(count)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return ""


def _numbers(settings):
    pitch = float(settings.get("pitch", 100))
    min_pins = int(settings.get("min_pins", 2))
    length = float(settings.get("length", 10))
    tolerance = float(settings.get("tolerance", 0.1))
    printer = str(settings.get("printer", DEFAULT_PRINTER))
    if printer not in PRINTERS:
        raise ValueError("Unknown printer")
    if pitch <= 0:
        raise ValueError("Dowel spacing must be greater than 0")
    if min_pins < 1:
        raise ValueError("Minimum dowels must be at least 1")
    if length <= 0:
        raise ValueError("Dowel length must be greater than 0")
    if tolerance < 0:
        raise ValueError("Clearance cannot be negative")
    browse_path = str(settings.get("browse") or DEFAULTS["browse"])
    return {
        "input": str(settings.get("input", "")).strip(),
        "output": str(settings.get("output", "")).strip(),
        "printer": printer,
        "pitch": pitch,
        "min_pins": min_pins,
        "length": length,
        "tolerance": tolerance,
        "write_3mf": bool(settings.get("write_3mf")),
        "browse": browse_path,
    }


def _check(settings):
    cleaned = _numbers(settings)
    source = Path(cleaned["input"]).expanduser()
    if not source.is_file():
        raise ValueError("Pick an STL or 3MF file")
    if source.suffix.lower() not in MODEL_SUFFIXES:
        raise ValueError("Input must be an .stl or .3mf file")
    if not cleaned["output"]:
        cleaned["output"] = str(source.with_suffix("")) + "-krackup"
    cleaned["input"] = str(source)
    cleaned["browse"] = str(source.parent)
    return cleaned


def start_job(settings):
    checked = _check(settings)
    save_settings(checked)
    with JOB.lock:
        if JOB.running:
            raise RuntimeError("A cut is already running")
        JOB.running = True
        JOB.log = [f"Cutting {checked['input']}"]
        JOB.error = ""
        JOB.output = ""

    def worker():
        try:
            def log(message):
                with JOB.lock:
                    JOB.log.append(str(message))

            krack(
                checked["input"],
                checked["output"],
                printer=checked["printer"],
                length=checked["length"],
                tolerance=checked["tolerance"],
                pitch=checked["pitch"],
                min_pins=checked["min_pins"],
                write_project=checked["write_3mf"],
                log=log,
            )
            with JOB.lock:
                JOB.output = checked["output"]
        except Exception as exc:
            with JOB.lock:
                JOB.error = str(exc)
        finally:
            with JOB.lock:
                JOB.running = False

    threading.Thread(target=worker, daemon=True).start()
    return checked


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/config":
            body = {
                "settings": load_settings(),
                "home": str(Path.home() / "Downloads"),
                "printers": [
                    {"id": key, "name": value["name"], "bed": list(value["bed"])}
                    for key, value in PRINTERS.items()
                ],
            }
            self._json(200, body)
            return
        if parsed.path == "/api/browse":
            query = parse_qs(parsed.query)
            raw = query.get("path", [str(Path.home() / "Downloads")])[0]
            try:
                self._json(200, browse(raw))
            except (FileNotFoundError, OSError) as exc:
                self._json(404, {"error": str(exc)})
            return
        if parsed.path == "/api/status":
            self._json(200, JOB.snapshot())
            return
        self._json(404, {"error": "Not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            payload = json.loads(self._read_body() or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "Settings were not valid JSON"})
            return
        if parsed.path == "/api/config":
            try:
                current = load_settings()
                current.update(payload)
                self._json(200, {"settings": save_settings(_numbers(current))})
            except ValueError as exc:
                self._json(400, {"error": str(exc)})
            return
        if parsed.path == "/api/run":
            try:
                self._json(200, {"started": True, "settings": start_job(payload)})
            except (ValueError, RuntimeError) as exc:
                self._json(400, {"error": str(exc)})
            return
        self._json(404, {"error": "Not found"})

    def log_message(self, fmt, *args):
        return

    def _read_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length).decode("utf-8") if length else ""

    def _json(self, status, payload):
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _send(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(host="127.0.0.1", port=8765, open_browser=True):
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{server.server_port}/"
    print(f"krack-up settings  {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
