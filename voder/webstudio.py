"""Local browser studio. No cloud service, build step, or additional dependency."""
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import threading
import uuid
import webbrowser

import numpy as np

from .engine import BAND_CENTERS, BAND_WIDTHS, Voice, synthesize
from .operator import DEMOS, EXAMPLES, audition_phone
from .phonemes import PHONES


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address=("127.0.0.1", 0)):
        super().__init__(address, StudioHandler)
        self.render_lock = threading.Lock()
        self.cache_lock = threading.Lock()
        self.audio = OrderedDict()

    def remember(self, data):
        key = uuid.uuid4().hex
        with self.cache_lock:
            self.audio[key] = data
            while len(self.audio) > 6 or (sum(map(len, self.audio.values())) > 80_000_000 and len(self.audio) > 1):
                self.audio.popitem(last=False)
        return key


class StudioHandler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, status, data, content_type="application/json; charset=utf-8"):
        if not isinstance(data, bytes):
            data = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def local_request(self):
        port = self.server.server_port
        hosts = (f"127.0.0.1:{port}", f"localhost:{port}")
        return self.headers.get("Host") in hosts and self.headers.get("Origin") in (None, *("http://" + h for h in hosts))

    def do_GET(self):
        if not self.local_request():
            self.send(403, {"error": "This studio accepts local requests only."})
            return
        path = self.path.split("?", 1)[0]
        assets = {"/": ("index.html", "text/html; charset=utf-8"), "/studio.css": ("studio.css", "text/css; charset=utf-8"), "/studio.js": ("studio.js", "text/javascript; charset=utf-8")}
        if path in assets:
            name, mime = assets[path]
            self.send(200, files("voder").joinpath("web", name).read_bytes(), mime)
        elif path == "/api/config":
            self.send(200, {"bands": BAND_CENTERS.tolist(), "widths": BAND_WIDTHS.tolist(), "demos": DEMOS,
                            "phones": [{"id": k, "example": EXAMPLES[k], "kind": p.kind} for k, p in PHONES.items() if k in EXAMPLES]})
        elif path.startswith("/api/audio/"):
            with self.server.cache_lock:
                data = self.server.audio.get(path.rsplit("/", 1)[-1])
            self.send(200 if data else 404, data or {"error": "Audio expired. Render again."}, "audio/wav" if data else "application/json")
        else:
            self.send(404, {"error": "Not found"})

    def do_POST(self):
        # Consume bounded bodies before replying, including rejected requests.
        # Closing a Windows socket with unread request bytes can discard a 403
        # response and surface as a connection reset in the client instead.
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 100_000:
                self.send(413, {"error": "Request is empty or too large."})
                return
            self.connection.settimeout(15)
            payload = self.rfile.read(size)
        except (ValueError, TimeoutError):
            self.send(400, {"error": "Incomplete request."})
            return
        if not self.local_request() or self.headers.get("X-Voder-Request") != "1":
            self.send(403, {"error": "This studio accepts local requests only."})
            return
        if self.path not in ("/api/render", "/api/audition"):
            self.send(404, {"error": "Not found"})
            return
        if not self.server.render_lock.acquire(blocking=False):
            self.send(409, {"error": "A render is already running. Please wait."})
            return
        try:
            request = json.loads(payload)
            if not isinstance(request, dict):
                raise ValueError("Expected a JSON object.")
            voice = Voice(**request.get("voice", {}))
            if self.path == "/api/audition":
                phone = request.get("phone")
                if not isinstance(phone, str):
                    raise ValueError("Choose a phoneme to preview.")
                result = audition_phone(phone, voice)
            else:
                text = request.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Enter some text first.")
                result = synthesize(text, voice, phonemes=bool(request.get("phonemes", False)))
            key = self.server.remember(result.wav_bytes())
            peaks = [float(np.max(np.abs(x), initial=0)) for x in np.array_split(result.samples, min(800, len(result.samples)))]
            self.send(200, {"id": key, "audio": "/api/audio/" + key, "peaks": peaks,
                            "phonemes": result.phonemes, "score": result.score()})
        except (ValueError, TypeError, AttributeError) as exc:
            self.send(400, {"error": str(exc)})
        except Exception:
            self.send(500, {"error": "Synthesis failed. Try a shorter passage or different settings."})
        finally:
            self.server.render_lock.release()


def main(*, open_browser=True, port=0):
    server = StudioServer(("127.0.0.1", port))
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Voder studio: {url}\nKeep this process running while using the studio. Ctrl+C closes it.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    main(open_browser=not args.no_browser, port=args.port)
