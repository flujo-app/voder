"""Text -> timed control gestures -> buzz/hiss -> resonant filters -> PCM.

No recordings, operating-system voices, neural models, or network calls.
"""
from dataclasses import asdict, dataclass
import io
import math
from pathlib import Path
import wave

import numpy as np
from scipy.signal import butter, lfilter

from .phonemes import PHONES
from .text import Token, phonemize

# A modern approximation, not a reconstruction of the original circuitry.
BAND_CENTERS = np.array([200, 350, 500, 700, 1000, 1400, 2000, 2800, 3900, 5500], dtype=float)
BAND_WIDTHS = np.array([150, 180, 220, 260, 350, 450, 600, 800, 1100, 1500], dtype=float)


@dataclass(frozen=True)
class Voice:
    pitch: float = 120
    rate: float = 1.0
    expression: float = 1.0
    breath: float = .02
    sample_rate: int = 22050
    seed: int = 1939
    mode: str = "clear"

    def __post_init__(self):
        for key, low, high in [("pitch", 60, 300), ("rate", .5, 2), ("expression", 0, 2), ("breath", 0, .3)]:
            value = getattr(self, key)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{key} must be between {low} and {high}")
        if self.sample_rate not in (16000, 22050, 24000, 44100, 48000):
            raise ValueError("sample_rate must be 16000, 22050, 24000, 44100, or 48000")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.mode not in ("clear", "classic"):
            raise ValueError("mode must be 'clear' or 'classic'")


@dataclass(frozen=True)
class Gesture:
    phone: str
    word: str
    stress: int
    start: float
    duration: float
    pitch_start: float
    pitch_end: float
    bands: tuple[float, ...]
    bands_end: tuple[float, ...]
    voiced: float
    noise: float
    kind: str
    formants: tuple[float, float, float]
    formants_end: tuple[float, float, float]


@dataclass
class Utterance:
    samples: np.ndarray
    sample_rate: int
    gestures: list[Gesture]
    unknown_words: list[str]
    voice: Voice

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def phonemes(self) -> str:
        return " ".join(g.phone + (str(g.stress) if g.kind == "vowel" else "") for g in self.gestures)

    def wav_bytes(self) -> bytes:
        stream = io.BytesIO()
        with wave.open(stream, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(np.rint(np.clip(self.samples, -1, 1) * 32767).astype("<i2").tobytes())
        return stream.getvalue()

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.wav_bytes())
        return path

    def score(self) -> dict:
        return {"sample_rate": self.sample_rate, "duration": self.duration,
                "schema_version": 2, "voice": asdict(self.voice),
                "band_centers_hz": BAND_CENTERS.tolist(), "unknown_words": self.unknown_words,
                "gestures": [asdict(g) for g in self.gestures]}


def spectral_gains(formants, kind: str, phone: str) -> np.ndarray:
    """Translate vowel resonances / consonant spectra into ten finger positions."""
    f = BAND_CENTERS
    gains = np.zeros(10)
    for center, width, amplitude in zip(formants, (85, 120, 180), (1.0, .65, .35)):
        # Integrate a resonance over each channel's effective pass band.
        effective = np.sqrt(width**2 + (BAND_WIDTHS * .42)**2)
        gains += amplitude / (1 + ((f - center) / effective)**2)
    if kind == "nasal":
        gains *= np.array([1, .7, .3, .08, .12, .1, .08, .05, .02, .01])
    elif kind in ("fricative", "affricate", "stop"):
        if phone in ("S", "Z", "T", "D"):
            gains = .03 + (f / 5500)**2
        elif phone in ("SH", "ZH", "CH", "JH"):
            gains = .015 + np.exp(-.5 * ((f - 3200) / 1150)**2)
        elif phone in ("F", "V", "TH", "DH"):
            gains = .08 + .3 * (f / 5500)
        elif phone in ("K", "G"):
            gains = .04 + np.exp(-.5 * ((f - 2300) / 850)**2)
        else:
            gains = .3 * np.exp(-f / 2500) + .08
    if kind == "silence":
        return np.zeros(10)
    return gains / max(float(np.linalg.norm(gains)), .001)


def plan(tokens: list[Token], voice: Voice) -> list[Gesture]:
    if not tokens:
        return []
    # Pitch resets at phrase boundaries and rises into a question mark.
    ranges = {}
    phrase_start = 0
    for index, token in enumerate(tokens):
        if token.phone == "SIL" or index == len(tokens) - 1:
            for j in range(phrase_start, index + 1):
                ranges[j] = (phrase_start, index, token.question)
            phrase_start = index + 1
    gestures = []
    cursor = .04
    for index, token in enumerate(tokens):
        p = PHONES[token.phone]
        duration = token.pause or p.duration
        if p.kind == "vowel":
            duration *= 1.22 if token.stress == 1 else 1.08 if token.stress == 2 else .8
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        if voice.mode == "clear" and p.kind == "vowel" and following is not None:
            # Vowels are shorter before unvoiced consonants (e.g. beat vs bead).
            after = PHONES[following.phone]
            if after.kind in ("stop", "fricative", "affricate") and not after.voice:
                duration *= .82
        if following is None or following.phone == "SIL":
            duration *= 1.22 if p.kind in ("vowel", "nasal", "liquid") else 1
        duration /= voice.rate
        first, last, question = ranges[index]
        count = max(1, last - first + (tokens[last].phone != "SIL"))
        pos = min(1, (index - first) / count)
        end_pos = min(1, (index + 1 - first) / count)
        def pitch(t):
            contour = 1.08 - .19 * t + (.35 * max(0, (t - .65) / .35) if question else 0)
            return voice.pitch * (1 + (contour - 1 + (.065 if token.stress == 1 else 0)) * voice.expression)
        formants = p.formants
        # Unstressed AH is schwa; breath follows the next vowel's mouth shape.
        if token.phone == "AH" and not token.stress:
            formants = (480, 1400, 2500)
        if token.phone == "HH" and following is not None:
            formants = PHONES[following.phone].formants
        gains = spectral_gains(formants, p.kind, token.phone)
        end = spectral_gains(p.end or formants, p.kind, token.phone)
        gestures.append(Gesture(token.phone, token.word, token.stress, cursor, duration,
                                pitch(pos), pitch(end_pos), tuple(gains), tuple(end),
                                p.voice, p.noise, p.kind, tuple(formants), tuple(p.end or formants)))
        cursor += duration
    return gestures


def _resonator(center: float, width: float, sample_rate: int):
    omega = 2 * np.pi * center / sample_rate
    alpha = np.sin(omega) / (2 * center / width)
    return np.array([alpha, 0, -alpha]) / (1 + alpha), np.array([1, -2 * np.cos(omega) / (1 + alpha), (1 - alpha) / (1 + alpha)])


def render(gestures: list[Gesture], voice: Voice) -> np.ndarray:
    sr = voice.sample_rate
    length = round(((gestures[-1].start + gestures[-1].duration) if gestures else .04) * sr) + round(.08 * sr)
    if length > sr * 600:
        raise ValueError("Utterance exceeds the ten-minute synthesis limit; split the text into smaller chunks")
    f0 = np.full(length, voice.pitch, dtype=np.float32)
    voicing = np.zeros(length, dtype=np.float32)
    noise = np.zeros(length, dtype=np.float32)
    # Rendering channels one at a time keeps long utterances modest in memory.
    control_segments = []
    previous = None
    for g in gestures:
        start = round(g.start * sr)
        end = round((g.start + g.duration) * sr)
        n = end - start
        t = np.linspace(0, 1, n, dtype=np.float32)
        f0[start:end] = g.pitch_start + (g.pitch_end - g.pitch_start) * t
        envelope = np.ones(n, dtype=np.float32)
        fade = min(round(.006 * sr), n // 2)
        if fade:
            envelope[:fade] = np.linspace(0, 1, fade)
            envelope[-fade:] = np.linspace(1, 0, fade)
        voiced = g.voiced * envelope
        hiss = (g.noise + voice.breath * g.voiced) * envelope
        if g.kind == "stop":
            # Closure, short release impulse, then aspiration before next vowel.
            voiced *= np.where(t < .58, .16, .4)
            burst = np.where(t >= .58, np.exp(-np.maximum(t - .58, 0) * 25), 0)
            hiss = g.noise * (burst + .09 * (t > .63)) * envelope
        elif g.kind == "affricate":
            voiced *= t > .25
            hiss *= np.clip((t - .25) / .08, 0, 1)
        voicing[start:end] = voiced
        noise[start:end] = hiss
        control_segments.append((start, end, g, previous))
        previous = g
    # Continuous sawtooth buzz with polyBLEP alias reduction; no per-phone reset.
    time = np.arange(length) / sr
    f0 *= 1 + .003 * voice.expression * np.sin(2 * np.pi * 5 * time)
    dt = f0 / sr
    phase = np.remainder(np.cumsum(dt, dtype=np.float64), 1)
    buzz = 2 * phase - 1
    left = phase < dt
    x = phase[left] / dt[left]
    buzz[left] -= 2 * x - x * x - 1
    right = phase > 1 - dt
    x = (phase[right] - 1) / dt[right]
    buzz[right] -= x * x + 2 * x + 1
    # Mild pre-emphasis keeps higher resonances audible without recorded speech.
    buzz = lfilter([1, -.92], [1], buzz) * 4
    rng = np.random.default_rng(voice.seed)
    source = buzz * voicing + rng.standard_normal(length) * noise * .24
    output = np.zeros(length, dtype=np.float64)
    for band, (center, width) in enumerate(zip(BAND_CENTERS, BAND_WIDTHS)):
        gains = np.zeros(length, dtype=np.float32)
        for start, end, g, prior in control_segments:
            n = end - start
            t = np.linspace(0, 1, n, dtype=np.float32)
            values = g.bands[band] + (g.bands_end[band] - g.bands[band]) * np.clip((t - .15) / .7, 0, 1)
            # Smooth connected voiced gestures while preserving consonant attacks.
            if prior and prior.kind in ("vowel", "liquid", "nasal") and g.kind in ("vowel", "liquid", "nasal"):
                transition = min(round(.025 * sr), n)
                weight = np.linspace(0, 1, transition)
                values[:transition] = prior.bands_end[band] * (1 - weight) + values[:transition] * weight
            gains[start:end] = values
        b, a = _resonator(center, width, sr)
        output += lfilter(b, a, source) * gains
    # Remove DC and tame peaks. Deterministic normalization leaves output headroom.
    b, a = butter(1, 65, btype="highpass", fs=sr)
    output = lfilter(b, a, output)
    peak = np.max(np.abs(output), initial=0)
    if peak > 1e-9:
        output = np.tanh(output * (1.6 / peak)) * (.92 / np.tanh(1.6))
    fade = min(round(.008 * sr), length // 2)
    output[:fade] *= np.linspace(0, 1, fade)
    output[-fade:] *= np.linspace(1, 0, fade)
    return output.astype(np.float32)


def synthesize(text: str, voice: Voice | None = None, *, phonemes: bool = False,
               pronunciations: dict[str, str] | None = None) -> Utterance:
    """Render English text (or ARPABET) into mono floating-point PCM in memory."""
    voice = voice or Voice()
    tokens, unknown = phonemize(text, raw=phonemes, pronunciations=pronunciations)
    gestures = plan(tokens, voice)
    return _render_utterance(gestures, voice, unknown)


def _render_utterance(gestures: list[Gesture], voice: Voice, unknown: list[str]) -> Utterance:
    """Shared final rendering path for speech and operator previews."""
    if voice.mode == "clear":
        from .clear import render_clear
        samples = render_clear(gestures, voice)
    else:
        samples = render(gestures, voice)
    return Utterance(samples, voice.sample_rate, gestures, unknown, voice)
