"""Short-time measurements of rendered PCM, independent of synthesis targets."""
from dataclasses import dataclass
import math

import numpy as np

from .engine import BAND_CENTERS, BAND_WIDTHS


@dataclass(frozen=True)
class AudioLevels:
    rms_dbfs: float
    peak_dbfs: float
    bands_dbfs: tuple[float, ...]


def measure_audio(samples: np.ndarray, sample_rate: int, at: float, *, window_size: int = 2048) -> AudioLevels:
    """Measure the trailing PCM window ending at `at` seconds.

    Band levels are integrated FFT power in the ten channel pass bands, in
    dBFS RMS (full-scale sine: approximately -3 dBFS). Silence floors at -100.
    Zero-padding at the start prevents borrowing future samples into a pause.
    This measures the rendered file, not microphone or loudspeaker output.
    """
    if sample_rate <= 0 or not math.isfinite(at) or at < 0:
        raise ValueError("sample_rate must be positive and at must be finite and nonnegative")
    if not isinstance(window_size, int) or window_size < 16:
        raise ValueError("window_size must be an integer of at least 16")
    end = min(len(samples), round(at * sample_rate))
    x = np.zeros(window_size)
    segment = samples[max(0, end - window_size):end]
    if len(segment):
        x[-len(segment):] = segment
    def db(power):
        return max(-100., 10 * math.log10(max(float(power), 1e-10)))
    rms = db(np.mean(x * x))
    peak = db(np.max(x * x))
    window = np.hanning(window_size)
    # Parseval scaling, with one-sided bins doubled except DC and Nyquist.
    power = np.abs(np.fft.rfft(x * window)) ** 2 / (window_size * np.sum(window * window))
    power[1:-1 if window_size % 2 == 0 else None] *= 2
    frequencies = np.fft.rfftfreq(window_size, 1 / sample_rate)
    bands = tuple(db(power[(frequencies >= center - width / 2) &
                           (frequencies < center + width / 2)].sum())
                  for center, width in zip(BAND_CENTERS, BAND_WIDTHS))
    return AudioLevels(rms, peak, bands)
