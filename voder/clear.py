"""Voder-inspired source/filter refinement with moving formants.

Voicing and aspiration use three moving resonators; consonant hiss uses the
original ten fixed channels. Control envelopes remain connected across phones.
This is a modern extension, not a historical Voder circuit model.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, lfilter

from .engine import BAND_CENTERS, BAND_WIDTHS, Gesture, Voice, _resonator


def _buzz(f0, sr):
    """Continuous-phase polyBLEP pulse excitation with a gentle spectral tilt."""
    dt = f0 / sr
    phase = np.remainder(np.cumsum(dt, dtype=np.float64), 1)
    wave = 2 * phase - 1
    left = phase < dt
    t = phase[left] / dt[left]
    wave[left] -= 2 * t - t * t - 1
    right = phase > 1 - dt
    t = (phase[right] - 1) / dt[right]
    wave[right] -= t * t + 2 * t + 1
    # Differentiation supplies harmonics to excite F2/F3. The lowpass softens
    # the release without relying on a recorded or learned glottal waveform.
    pulse = lfilter([1, -.97], [1], wave)
    b, a = butter(1, 4200, fs=sr)
    return lfilter(b, a, pulse) * np.sqrt(120 / f0) * 5


def _controls(gestures, voice):
    sr = voice.sample_rate
    duration = gestures[-1].start + gestures[-1].duration if gestures else .04
    length = round(duration * sr) + round(.08 * sr)
    if length > sr * 600:
        raise ValueError("Utterance exceeds the ten-minute synthesis limit; split the text into smaller chunks")
    f0 = np.full(length, voice.pitch, dtype=np.float32)
    voiced = np.zeros(length, dtype=np.float32)
    aspiration = np.zeros(length, dtype=np.float32)
    frication = np.zeros(length, dtype=np.float32)
    # Formant and noise weights need only a control-rate grid, not ten large
    # full-resolution arrays. Resonators update every 2.5 ms, retaining state.
    hop = max(1, round(sr * .0025))
    centers = np.arange(0, length, hop) + hop / 2
    frequencies = np.tile([500., 1500., 2500.], (len(centers), 1))
    weights = np.zeros((len(centers), 3))
    bands = np.zeros((len(centers), 10))
    ranges = []
    for g in gestures:
        start, end = round(g.start * sr), round((g.start + g.duration) * sr)
        ranges.append((start, end, g))
        t = np.linspace(0, 1, end - start)
        f0[start:end] = np.linspace(g.pitch_start, g.pitch_end, end - start)
        voiced[start:end] = g.voiced * (1.10 if g.stress == 1 else 1)
        aspiration[start:end] = (voice.breath + .006) * g.voiced
        if g.kind == "breath":
            aspiration[start:end] = g.noise * 1.8
        elif g.kind == "stop":
            # Preserve a low voice bar for b/d/g; unvoiced closures stay quiet.
            voiced[start:end] *= np.where(t < .58, .22, .45)
            burst = np.where(t >= .58, np.exp(-np.maximum(t - .58, 0) * 28), 0)
            frication[start:end] = g.noise * burst
            aspiration[start:end] += .08 * g.noise * (t > .65)
        elif g.kind == "affricate":
            attack = np.clip((t - .25) / .06, 0, 1)
            voiced[start:end] *= attack
            frication[start:end] = g.noise * attack
        elif g.kind == "fricative":
            frication[start:end] = g.noise
        idx = (centers >= start) & (centers < end)
        glide = np.clip(((centers[idx] - start) / (end - start) - .15) / .65, 0, 1)
        frequencies[idx] = np.array(g.formants) + glide[:, None] * (np.array(g.formants_end) - g.formants)
        weights[idx] = (1, .65, .35)
        if g.kind == "nasal":
            weights[idx] = (1, .08, .035)
        elif g.kind in ("fricative", "stop", "affricate"):
            # Voiced consonants need a low-frequency murmur separate from hiss.
            frequencies[idx, 0] = 240
            weights[idx] = (.8, .07, .025)
        bands[idx] = g.bands
    # Interpolate actual mouth positions at boundaries. Smoothing only the
    # controls (not the waveform) removes artificial per-phone silence.
    frequencies = gaussian_filter1d(frequencies, .009 * sr / hop, axis=0, mode="nearest")
    weights = gaussian_filter1d(weights, .005 * sr / hop, axis=0, mode="nearest")
    bands = gaussian_filter1d(bands, .002 * sr / hop, axis=0, mode="nearest")
    f0 = gaussian_filter1d(f0, .004 * sr, mode="nearest")
    voiced = gaussian_filter1d(voiced, .002 * sr, mode="constant")
    aspiration = gaussian_filter1d(aspiration, .0015 * sr, mode="constant")
    # Keep stop attacks fast; apply a short fade only at outer noise boundaries.
    frication = gaussian_filter1d(frication, .0005 * sr, mode="constant")
    return length, hop, centers, frequencies, weights, bands, f0, voiced, aspiration, frication, ranges


def render_clear(gestures: list[Gesture], voice: Voice) -> np.ndarray:
    sr = voice.sample_rate
    (length, hop, centers, frequencies, weights, bands, f0,
     voiced, aspiration, frication, ranges) = _controls(gestures, voice)
    f0 *= 1 + .002 * voice.expression * np.sin(2 * np.pi * 4.7 * np.arange(length) / sr)
    rng = np.random.default_rng(voice.seed)
    noise = rng.standard_normal(length)
    source = _buzz(f0, sr) * voiced + noise * aspiration * .4
    output = np.zeros(length)
    # Parallel formants preserve the source/filter architecture while placing
    # vowel peaks accurately between the original fixed band's center points.
    for band, width in enumerate((75., 105., 160.)):
        filtered = np.zeros(length)
        state = np.zeros(2)
        for frame, start in enumerate(range(0, length, hop)):
            end = min(length, start + hop)
            b, a = _resonator(frequencies[frame, band], width, sr)
            filtered[start:end], state = lfilter(b, a, source[start:end], zi=state)
        gain = np.interp(np.arange(length), centers, weights[:, band])
        # Alternating polarity reduces cancellation between parallel resonances.
        output += filtered * gain * (1 if band != 1 else -1)
    hiss = noise * frication * .19
    for band, (center, width) in enumerate(zip(BAND_CENTERS, BAND_WIDTHS)):
        b, a = _resonator(center, width, sr)
        gain = np.interp(np.arange(length), centers, bands[:, band])
        output += lfilter(b, a, hiss) * gain
    b, a = butter(1, 65, btype="highpass", fs=sr)
    output = lfilter(b, a, output)
    # Leave actual pauses silent after a short resonator decay.
    for start, end, g in ranges:
        if g.kind == "silence":
            release = min(round(.012 * sr), end - start)
            output[start:start + release] *= np.linspace(1, 0, release)
            output[start + release:end] = 0
    # Linear headroom normalization preserves spectral contrast (no tanh
    # harmonics that would fill in valleys between vowel formants).
    peak = np.max(np.abs(output), initial=0)
    if peak > 1e-9:
        output *= .92 / peak
    fade = min(round(.008 * sr), length // 2)
    output[:fade] *= np.linspace(0, 1, fade)
    output[-fade:] *= np.linspace(1, 0, fade)
    return output.astype(np.float32)
