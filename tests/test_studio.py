"""Native workbench integration checks; no speaker output or visible window."""
import time
import unittest
from unittest.mock import patch

try:
    import tkinter as tk
    from voder.studio import Studio
except ImportError:
    tk = None


@unittest.skipIf(tk is None, "Tk is not installed")
class StudioTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError:
            self.skipTest("Tk display unavailable")
        self.root.withdraw()
        self.app = Studio(self.root)
        self.app.text.delete("1.0", "end")
        self.app.text.insert("1.0", "Hello.")
        self.root.update()

    def tearDown(self):
        if hasattr(self, "app"):
            self.app.close()

    def finish(self):
        deadline = time.monotonic() + 15
        while self.app.future is not None and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.01)
        self.assertIsNone(self.app.future, "Worker did not complete")

    def test_mode_comparison_and_stale_audio_notice(self):
        self.app.start_render()
        self.finish()
        current = self.app.utterance
        self.assertEqual(current.voice.mode, "clear")
        self.app.mode.set("classic")
        self.assertIn("Settings changed", self.app.status.get())
        self.app.start_render()
        self.finish()
        self.assertIs(self.app.previous, current)
        self.assertEqual(self.app.utterance.voice.mode, "classic")
        self.assertTrue(self.app.previous_path.is_file())

    def test_render_failure_keeps_last_successful_audio(self):
        self.app.start_render()
        self.finish()
        prior = self.app.utterance
        wav = self.app.audio_path.read_bytes()
        self.app.raw.set(True)
        self.app.text.delete("1.0", "end")
        self.app.text.insert("1.0", "INVALIDPHONE")
        self.app.start_render()
        self.finish()
        self.assertIs(self.app.utterance, prior)
        self.assertEqual(self.app.audio_path.read_bytes(), wav)
        self.assertIn("Synthesis failed", self.app.status.get())
        self.assertIn("Unknown phoneme", self.app.status.get())

    def test_operator_preview_preserves_editor_and_uses_normal_history(self):
        self.app.start_render()
        self.finish()
        prior = self.app.utterance
        original_text = self.app.text.get("1.0", "end-1c")
        with patch.object(self.app, "play") as play:
            self.app.preview_phone("IY")
            self.finish()
            play.assert_called_once()
        self.assertEqual(self.app.text.get("1.0", "end-1c"), original_text)
        self.assertIs(self.app.previous, prior)
        self.assertEqual(self.app.utterance.gestures[0].phone, "IY")
        self.assertIn("Preview IY", self.app.status.get())

    def test_stop_cancels_pending_autoplay(self):
        self.app.autoplay.set(True)
        with patch.object(self.app, "play") as play:
            self.app.start_render()
            self.app.stop()
            self.finish()
            play.assert_not_called()
            self.app.preview_phone("AA")
            self.app.stop()
            self.finish()
            play.assert_not_called()

    def test_operator_card_demos_and_measured_meter(self):
        self.app.open_operator()
        self.app.operator_window.withdraw()
        self.root.update()
        self.assertTrue(self.app.operator_window.winfo_exists())
        self.app.raw.set(True)
        self.app.load_demo("How are you?")
        self.assertFalse(self.app.raw.get())
        self.assertEqual(self.app.text.get("1.0", "end-1c"), "How are you?")
        self.app.start_render()
        self.finish()
        self.app.draw_meters(.3)
        labels = [self.app.meters.itemcget(i, "text") for i in self.app.meters.find_all() if self.app.meters.type(i) == "text"]
        self.assertIn("dBFS", labels[0])
        self.app.meter_view.set("Control targets")
        self.app.draw_meters(.3)
        labels = [self.app.meters.itemcget(i, "text") for i in self.app.meters.find_all() if self.app.meters.type(i) == "text"]
        self.assertIn("score targets", labels[0])
