import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np
from scipy.signal import correlate, welch

from voder import Voice, synthesize
from voder.text import lexicon, normalize, phonemize


class FrontEndTests(unittest.TestCase):
    def test_dictionary_and_stress(self):
        tokens, unknown = phonemize("Hello world!")
        self.assertEqual([t.phone for t in tokens], ["HH", "AH", "L", "OW", "W", "ER", "L", "D", "SIL"])
        self.assertEqual(tokens[3].stress, 1)
        self.assertEqual(unknown, [])
        self.assertGreater(len(lexicon()), 100000)

    def test_all_dictionary_phones_are_supported(self):
        from voder.text import parse_phone
        for symbol in {p for word in lexicon().values() for p in word}:
            parse_phone(symbol)

    def test_numbers(self):
        text = normalize("Dr. Lee has 1,234.50 & 007.")
        self.assertIn("doctor lee", text)
        self.assertIn("one thousand two hundred thirty four point five zero", text)
        self.assertIn("zero zero seven", text)

    def test_fallback_is_reported(self):
        tokens, unknown = phonemize("blorptastic blorptastic")
        self.assertEqual(unknown, ["blorptastic"])
        self.assertTrue(tokens)

    def test_unicode_word_boundaries(self):
        tokens, unknown = phonemize("Hello—world…")
        self.assertEqual(unknown, [])
        self.assertIn("world", [t.word for t in tokens])
        self.assertEqual(tokens[-1].phone, "SIL")

    def test_raw_validation(self):
        tokens, _ = phonemize("HH AH0 L OW1 SIL", raw=True)
        self.assertEqual(tokens[-1].phone, "SIL")
        for text in ("NOPE", "T1", "AA9"):
            with self.assertRaises(ValueError):
                phonemize(text, raw=True)

    def test_long_input_rejected(self):
        with self.assertRaises(ValueError):
            synthesize("a" * 4001)

    def test_custom_pronunciation_takes_precedence(self):
        tokens, unknown = phonemize("Moe the Voder", pronunciations={"Moe": "M OW1", "the": "DH IY1"})
        self.assertEqual([(t.phone, t.stress) for t in tokens[:4]], [("M", 0), ("OW", 1), ("DH", 0), ("IY", 1)])
        self.assertEqual(unknown, [])
        self.assertEqual(phonemize("the cat")[0][1].phone, "AH")
        self.assertEqual(phonemize("the apple")[0][1].phone, "IY")

    def test_invalid_custom_pronunciations(self):
        for custom in ([], {"bad name": "AA1"}, {"name": ""}, {"name": "SIL"}, {"name": "T1"}, {"name": ["N"]}):
            with self.subTest(custom=custom), self.assertRaises(ValueError):
                phonemize("name", pronunciations=custom)
        with self.assertRaises(ValueError):
            synthesize("AA1", phonemes=True, pronunciations={"name": "N EY1 M"})


class SignalTests(unittest.TestCase):
    def test_finite_bounded_pcm_and_wav(self):
        audio = synthesize("Hello. I am the Voder!")
        self.assertTrue(np.all(np.isfinite(audio.samples)))
        self.assertGreater(np.sqrt(np.mean(audio.samples**2)), .01)
        self.assertLessEqual(np.max(np.abs(audio.samples)), .921)
        with wave.open(io.BytesIO(audio.wav_bytes()), "rb") as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, 22050))
            self.assertEqual(wav.getnframes(), len(audio.samples))

    def test_pitch_changes_actual_waveform_period(self):
        for pitch in (100, 180):
            audio = synthesize("AA1", Voice(pitch=pitch, expression=0, breath=0), phonemes=True)
            x = audio.samples[int(.055 * audio.sample_rate):int(.135 * audio.sample_rate)]
            corr = correlate(x, x, mode="full")[len(x)-1:]
            low, high = int(audio.sample_rate / (pitch * 1.2)), int(audio.sample_rate / (pitch * .8))
            period = low + np.argmax(corr[low:high])
            measured = audio.sample_rate / period
            self.assertAlmostEqual(measured, pitch, delta=4)

    def test_fricative_is_high_frequency_and_aperiodic(self):
        audio = synthesize("S", Voice(breath=0), phonemes=True)
        x = audio.samples[int(.05 * audio.sample_rate):int(.14 * audio.sample_rate)]
        power = np.abs(np.fft.rfft(x))**2
        freqs = np.fft.rfftfreq(len(x), 1 / audio.sample_rate)
        self.assertGreater(power[freqs > 2500].sum() / power.sum(), .7)
        corr = correlate(x, x, mode="full")[len(x)-1:]
        self.assertLess(np.max(np.abs(corr[100:300])) / corr[0], .3)

    def test_stop_has_closure_and_burst(self):
        audio = synthesize("T", phonemes=True)
        sr = audio.sample_rate
        closure = audio.samples[int(.05 * sr):int(.075 * sr)]
        release = audio.samples[int(.087 * sr):int(.105 * sr)]
        self.assertLess(np.mean(closure**2), np.mean(release**2) * .01)

    def test_rate_and_question_contour(self):
        slow = synthesize("Hello?", Voice(rate=.7))
        fast = synthesize("Hello?", Voice(rate=1.4))
        self.assertGreater(slow.duration, fast.duration * 1.7)
        statement = synthesize("Hello.")
        question = synthesize("Hello?")
        self.assertGreater(question.gestures[-2].pitch_end, statement.gestures[-2].pitch_end)

    def test_seed_and_silence(self):
        np.testing.assert_array_equal(synthesize("hello").samples, synthesize("hello").samples)
        self.assertFalse(np.array_equal(synthesize("S", Voice(seed=1), phonemes=True).samples,
                                        synthesize("S", Voice(seed=2), phonemes=True).samples))
        for text in ("", "...", "   "):
            self.assertEqual(np.max(np.abs(synthesize(text).samples)), 0)

    def test_all_phones_and_sample_rates(self):
        from voder.phonemes import PHONES
        for mode in ("clear", "classic"):
            for sr in (16000, 22050, 24000, 44100, 48000):
                audio = synthesize(" ".join(PHONES), Voice(sample_rate=sr, mode=mode), phonemes=True)
                self.assertTrue(np.all(np.isfinite(audio.samples)))
                self.assertTrue(all(len(g.bands) == 10 for g in audio.gestures))

    def test_connected_voicing_has_no_boundary_dropout(self):
        audio = synthesize("AA1 AA1", Voice(expression=0, breath=0), phonemes=True)
        boundary = round(audio.gestures[1].start * audio.sample_rate)
        tight = audio.samples[boundary-44:boundary+44]
        wide = audio.samples[boundary-440:boundary+440]
        self.assertGreater(np.sqrt(np.mean(tight**2) / np.mean(wide**2)), .6)

    def test_voiced_fricative_preserves_low_frequency_voice(self):
        fractions = []
        for phone in ("S", "Z"):
            audio = synthesize(phone, Voice(expression=0, breath=0, pitch=100), phonemes=True)
            x = audio.samples[round(.07*audio.sample_rate):round(.14*audio.sample_rate)]
            freq, power = welch(x, fs=audio.sample_rate, nperseg=1024)
            fractions.append(power[freq < 500].sum() / power.sum())
        self.assertLess(fractions[0], .02)
        self.assertGreater(fractions[1], .15)

    def test_front_and_back_vowels_have_distinct_second_formants(self):
        fractions = []
        for phone in ("IY1", "UW1"):
            audio = synthesize(phone, Voice(expression=0, breath=0, pitch=100), phonemes=True)
            x = audio.samples[round(.07*audio.sample_rate):round(.14*audio.sample_rate)]
            freq, power = welch(x, fs=audio.sample_rate, nperseg=1024)
            fractions.append(power[(freq > 2000) & (freq < 2600)].sum() / power.sum())
        self.assertGreater(fractions[0], fractions[1] * 2)

    def test_pre_fortis_shortening_and_pause(self):
        beat = synthesize("beat.")
        bead = synthesize("bead.")
        self.assertLess(beat.gestures[1].duration, bead.gestures[1].duration)
        pause = beat.gestures[-1]
        x = beat.samples[round((pause.start + .02) * beat.sample_rate):round((pause.start + pause.duration) * beat.sample_rate)]
        self.assertEqual(np.max(np.abs(x)), 0)

    def test_extreme_settings_remain_finite(self):
        for pitch, rate, expression, breath in ((60, .5, 0, 0), (300, 2, 2, .3)):
            audio = synthesize("Are you the Voder?", Voice(pitch, rate, expression, breath, sample_rate=16000))
            self.assertTrue(np.all(np.isfinite(audio.samples)))
            self.assertLessEqual(np.max(np.abs(audio.samples)), .921)

    def test_invalid_voice_settings(self):
        for setting in ({"pitch": 0}, {"rate": float("nan")}, {"sample_rate": 8000}, {"breath": 1}, {"seed": -1}, {"mode": "neural"}):
            with self.assertRaises(ValueError):
                Voice(**setting)


class CLITests(unittest.TestCase):
    def test_stdin_wav_and_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav, score = Path(tmp) / "speech.wav", Path(tmp) / "score.json"
            result = subprocess.run([sys.executable, "-m", "voder", "-o", str(wav), "--score", str(score)],
                                    input="Hello world.", text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreater(wav.stat().st_size, 1000)
            self.assertEqual(len(json.loads(score.read_text())["band_centers_hz"]), 10)
            self.assertEqual(json.loads(score.read_text())["voice"]["mode"], "clear")

    def test_cli_custom_lexicon_and_classic_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            lex = Path(tmp) / "lexicon.json"
            lex.write_text(json.dumps({"moe": "M OW1"}))
            result = subprocess.run([sys.executable, "-m", "voder", "Moe", "--lexicon", str(lex), "--mode", "classic", "-o", str(Path(tmp) / "voice.wav")], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("M OW1", result.stdout)
            lex.write_text('{"moe": []}')
            result = subprocess.run([sys.executable, "-m", "voder", "Moe", "--lexicon", str(lex)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
