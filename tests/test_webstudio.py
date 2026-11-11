import http.client
import io
import json
import threading
import unittest
import wave

from voder.webstudio import StudioServer


class WebStudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = StudioServer()
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def request(self, method, path, data=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=15)
        body = json.dumps(data) if data is not None else None
        connection.request(method, path, body, headers or {"Content-Type": "application/json", "X-Voder-Request": "1"})
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def test_assets_and_inventory(self):
        for path, mime in (("/", "text/html"), ("/studio.css", "text/css"), ("/studio.js", "text/javascript")):
            status, headers, content = self.request("GET", path)
            self.assertEqual(status, 200)
            self.assertIn(mime, headers["Content-Type"])
            self.assertGreater(len(content), 100)
        status, _, body = self.request("GET", "/api/config")
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(len(data["phones"]), 39)
        self.assertEqual(len(data["bands"]), 10)

    def test_text_render_returns_real_audio_and_score(self):
        status, _, body = self.request("POST", "/api/render", {"text": "She saw me.", "voice": {"mode": "clear"}})
        data = json.loads(body)
        self.assertEqual(status, 200, data)
        self.assertEqual(data["score"]["voice"]["mode"], "clear")
        self.assertLessEqual(len(data["peaks"]), 800)
        self.assertGreater(max(data["peaks"]), .1)
        status, _, wav = self.request("GET", data["audio"])
        self.assertEqual(status, 200)
        with wave.open(io.BytesIO(wav), "rb") as audio:
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getsampwidth(), 2)
            self.assertGreater(audio.getnframes(), 1000)

    def test_phoneme_mode_and_audition(self):
        for route, data in (("/api/render", {"text": "HH AH0 L OW1", "phonemes": True}),
                            ("/api/audition", {"phone": "IY", "voice": {"mode": "classic"}})):
            status, _, body = self.request("POST", route, data)
            self.assertEqual(status, 200, body)
            self.assertTrue(json.loads(body)["score"]["gestures"])

    def test_invalid_inputs_are_actionable_errors(self):
        for data in ({"text": ""}, {"text": "hi", "voice": {"pitch": 900}}, {"text": "hi", "voice": []},
                     {"text": "a" * 4001}, {"text": "NOTAPHONE", "phonemes": True}, ["hello"]):
            with self.subTest(data=str(data)[:60]):
                status, _, body = self.request("POST", "/api/render", data)
                self.assertEqual(status, 400, body)
                self.assertTrue(json.loads(body)["error"])

    def test_busy_render_rejects_concurrent_work(self):
        with self.server.render_lock:
            status, _, _ = self.request("POST", "/api/render", {"text": "hello"})
        self.assertEqual(status, 409)

    def test_foreign_origin_and_rebinding_hosts_are_rejected(self):
        for headers in ({"Origin": "https://example.com", "X-Voder-Request": "1"},
                        {"Host": "evil.example", "X-Voder-Request": "1"}, {}):
            # An empty explicit header set must omit the custom request header.
            actual = headers or {"Content-Type": "application/json"}
            status, _, _ = self.request("POST", "/api/render", {"text": "hello"}, actual)
            self.assertEqual(status, 403)

    def test_missing_routes_and_expired_audio(self):
        for path in ("/api/audio/missing", "/../pyproject.toml", "/unknown"):
            status, _, _ = self.request("GET", path)
            self.assertEqual(status, 404)

    def test_audio_cache_is_bounded(self):
        ids = [self.server.remember(b"test") for _ in range(7)]
        self.assertNotIn(ids[0], self.server.audio)
        self.assertIn(ids[-1], self.server.audio)
        self.assertLessEqual(len(self.server.audio), 6)
