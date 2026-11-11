import unittest

import numpy as np

from voder import Voice, audition_phone, synthesize
from voder.analysis import measure_audio
from voder.operator import DEMOS, EXAMPLES
from voder.phonemes import PHONES


class OperatorTests(unittest.TestCase):
    def test_card_covers_inventory_and_both_modes(self):
        self.assertEqual(set(EXAMPLES), set(PHONES) - {"SIL"})
        for mode in ("clear", "classic"):
            for symbol in EXAMPLES:
                with self.subTest(mode=mode, phone=symbol):
                    audio = audition_phone(symbol, Voice(mode=mode), duration=.3)
                    self.assertTrue(np.all(np.isfinite(audio.samples)))
                    self.assertGreater(np.max(np.abs(audio.samples)), .1)
                    self.assertAlmostEqual(audio.duration, .42, delta=2/audio.sample_rate)

    def test_stop_preview_retains_short_release(self):
        audio = audition_phone("T", duration=1)
        self.assertLess(audio.gestures[0].duration, .12)
        tail = audio.samples[round(.25 * audio.sample_rate):]
        self.assertLess(np.max(np.abs(tail)), 1e-5)
        sustained = audition_phone("S", duration=1)
        self.assertGreater(np.sqrt(np.mean(sustained.samples[round(.3 * audio.sample_rate):round(.9 * audio.sample_rate)]**2)), .05)

    def test_preview_validation_and_schwa(self):
        for duration in (0, 5, float("nan")):
            with self.assertRaises(ValueError):
                audition_phone("AA", duration=duration)
        with self.assertRaises(ValueError):
            audition_phone("INVALID")
        self.assertEqual(audition_phone("AH").gestures[0].stress, 1)
        self.assertEqual(audition_phone("AH0").gestures[0].stress, 0)

    def test_demo_phrases_use_supported_pronunciations(self):
        for text in DEMOS.values():
            audio = synthesize(text)
            self.assertEqual(audio.unknown_words, [])
            self.assertTrue(all(g.phone in PHONES for g in audio.gestures))


class AudioMeterTests(unittest.TestCase):
    def test_sine_calibrates_rms_peak_and_frequency_band(self):
        sr = 24000
        x = .5 * np.sin(2 * np.pi * 1000 * np.arange(sr) / sr)
        levels = measure_audio(x, sr, .5)
        self.assertAlmostEqual(levels.rms_dbfs, -9.03, delta=.05)
        self.assertAlmostEqual(levels.peak_dbfs, -6.02, delta=.05)
        self.assertEqual(int(np.argmax(levels.bands_dbfs)), 4)
        self.assertAlmostEqual(levels.bands_dbfs[4], -9.03, delta=.1)
        quieter = measure_audio(x * .5, sr, .5)
        self.assertAlmostEqual(levels.bands_dbfs[4] - quieter.bands_dbfs[4], 6.02, delta=.05)

    def test_zero_and_start_window_do_not_read_future_audio(self):
        x = np.concatenate([np.zeros(2400), np.ones(2400)])
        for at in (0, .05, .1):
            levels = measure_audio(x, 24000, at)
            self.assertEqual(levels.rms_dbfs, -100)
            self.assertEqual(levels.bands_dbfs, (-100.,) * 10)
        self.assertEqual(measure_audio(np.array([]), 24000, 0).peak_dbfs, -100)

    def test_meter_uses_sample_rate_for_bin_mapping(self):
        for sr in (16000, 22050, 48000):
            x = np.sin(2 * np.pi * 2800 * np.arange(sr) / sr)
            levels = measure_audio(x, sr, .5)
            self.assertEqual(int(np.argmax(levels.bands_dbfs)), 7)

    def test_invalid_meter_arguments(self):
        for sr, at in ((0, 0), (22050, -1), (22050, float("nan"))):
            with self.assertRaises(ValueError):
                measure_audio(np.zeros(20), sr, at)
