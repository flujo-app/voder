# Voder

An offline English text-to-speech engine inspired by Homer Dudley's Voder: buzzing and hissing sources shaped by resonant filters and timed control gestures.

Version 0.4 introduces a **local browser studio**: a brass-and-ivory console, a playable ten-key instrument, listening lessons, and a typographic history of the Voder. Its presentation and teaching approach draw on the user's Grokbot prototype. [Comparison and adaptation notes](docs/grokbot-comparison.md) explain the design and engineering choices.

**Clear**, the default mode, uses moving vowel resonances, a separate consonant-noise path, and continuous voicing across phonemes. **Classic** retains the original fixed ten-band rendering approach. Both generate the waveform locally, with no recorded speech, neural model, operating-system TTS, or network service. This is an experimental mechanical voice, not a circuit-accurate recreation.

## Run

Requires Python 3.10 or newer. From this folder:

```powershell
python -m pip install -e .
python -m voder --studio
```

The studio opens in your browser. Keep the terminal running while using it; Ctrl+C closes the local server. All scripts, styles, and synthesis run locally, with no frontend installation or cloud account. The history's source links are optional external reading.

In **Speak**, select **Clear** or **Classic**, enter text, and click **Synthesize voice**. Ctrl+Enter also synthesizes. Four presets and sliders control pitch, speed, expression, and breath. **Play when ready** enables automatic playback.

After a second render, **Play previous** lets you compare it with the first. Try the same phrase in both modes. Playback and exports use the last completed render; a notice appears when you change the text or controls. Export WAV or the JSON control score with the corresponding buttons.

The oscilloscope and **Audio spectrum** respond to both speech playback and the live instrument, measured after the browser's volume control. The ten bars show FFT energy around each filter band, and RMS reports the output signal level in dBFS. They do not measure your speakers or microphone. Select **Control targets** to see planned gains instead; in Clear mode those bars represent consonant-noise gains, while vowels use moving formants.

In **The instrument**, hold the ivory keys with your pointer or keyboard **1–0** to open resonators. Hold **Space** or the wrist bar to add a pitched buzz; releasing it returns to hiss. **Q/W/E** trigger P/T/K stop bursts. The pitch pedal or **↑/↓** changes live pitch. Multiple keys can sound together. Releasing keys, switching pages, or leaving the window releases the live controls. The operator's card below previews all 39 phonemes through the Python engine, using your current voice settings. A preview becomes the current take without changing your script.

**Operator school** offers four interactive experiments on buzz/hiss, vowel resonances, intonation, and stop consonants. **The story** introduces Homer Dudley, Helen Harper, and the 1939 fair demonstrations, with links to Smithsonian and Julius O. Smith's technical account.

**Stop** or **Escape** silences audio and cancels pending automatic playback, even if a render is still finishing. WAV and JSON score exports use the current take. Renders live in memory; download anything you want to keep before closing or refreshing the page. Browser playback requires Web Audio support. The original Tk interface remains available with `python -m voder --native-studio`; its playback is Windows-only and it requires Tkinter.

For a fixed local URL without automatically opening a browser:

```powershell
python -m voder.webstudio --no-browser --port 8765
```

## Command line

```powershell
python -m voder "Hello world. I am the Voder." -o hello.wav --play
python -m voder "Can electricity speak?" --pitch 150 --rate 0.9 -o question.wav
python -m voder "HH AH0 L OW1 W ER1 L D" --phonemes -o phonemes.wav
python -m voder "Hello world." --mode classic --score controls.json -o classic.wav
python -m voder "Hello Moe." --lexicon names.json -o names.wav
Get-Content story.txt -Raw | python -m voder -o story.wav
```

The installation also provides `voder` and `voder-studio` commands. Use `--help` for all options. Inputs are limited to 4,000 characters and ten minutes of audio per request; split longer material into paragraphs. Empty input produces a short silent WAV.

## Listen and compare

- [Clear sample](examples/hello-voder.wav): the new default, with a matching JSON control score.
- [Original version](examples/hello-voder-before.wav): preserves the original version's output.
- [Current Classic mode](examples/hello-voder-classic.wav): the fixed ten-band renderer with the current text front end.
- [Before and after](examples/compare-voder.wav): original followed by Clear, separated by a short silence.

## Python API

```python
from voder import Voice, synthesize

voice = Voice(pitch=125, rate=0.95, expression=1.0, breath=0.02, mode="clear")
speech = synthesize("Hello world. I am the Voder.", voice)
speech.save("hello.wav")

print(speech.phonemes)
print(speech.unknown_words)
pcm = speech.samples       # float32 mono PCM, with peak headroom
sample_rate = speech.sample_rate
wav_data = speech.wav_bytes()
control_score = speech.score()
```

Use `synthesize("AA1 S T", phonemes=True)` for direct ARPABET control. `SIL` inserts a pause. Stress markers `0`, `1`, and `2` are accepted on vowels. The supported inventory is in `voder/phonemes.py`.

Correct unfamiliar names without changing the bundled dictionary:

```python
speech = synthesize("Hello Moe.", pronunciations={"moe": "M OW1"})
```

Individual sounds can also be auditioned from Python:

```python
from voder import audition_phone

audition_phone("IY", voice, duration=0.8).save("ee.wav")
audition_phone("T", voice).save("stop.wav")
```

Preview duration accepts 0.2–2 seconds, independently of speaking rate. Vowels default to primary stress; explicit stress such as `AH0` is preserved.

The CLI accepts the same mapping in a JSON file via `--lexicon names.json`, for example `{"moe": "M OW1"}`. Overrides take precedence over dictionary entries and function-word reductions. Invalid phonemes produce a descriptive error. Raw phoneme input cannot be combined with word overrides.

| Control | Range | Effect |
| --- | --- | --- |
| `pitch` | 60–300 Hz | Base buzz frequency, before intonation |
| `rate` | 0.5–2 | Speaking speed multiplier |
| `expression` | 0–2 | Stress accents, pitch contours, slight vibrato |
| `breath` | 0–0.3 | Additional noise during voiced sounds |
| `sample_rate` | 16000, 22050, 24000, 44100, 48000 | Output samples per second |
| `seed` | Nonnegative integer | Repeatable noise; default 1939 |
| `mode` | `clear`, `classic` | Moving formants or fixed ten-band rendering |

## How it works

1. **Text → phonemes.** Normalize basic numbers, decimals, punctuation and a few abbreviations. Look up American English pronunciations in the bundled CMU dictionary or a custom override. Reduce common unstressed function words inside phrases, including “the” before a vowel. Unknown words use approximate spelling rules and are reported.
2. **Phonemes → gestures.** Plan duration, lexical stress, phrase pitch, final lengthening and question intonation. Clear shortens vowels before unvoiced consonants. Diphthongs move between resonance targets. The score records timings, pitch and formant endpoints, noise-band targets, voice settings and a schema version.
3. **Excitation.** A continuous-phase sawtooth with polyBLEP alias reduction and differentiation supplies a pulse-like buzz. Seeded white noise supplies hiss. Stops have closure, release and aspiration; affricates combine closure and sustained noise. Clear smooths source controls across phones instead of fading every phone to silence.
4. **Resonant filtering.** Clear sends voicing and aspiration through three parallel moving formants, updating every 2.5 ms with preserved filter state. Consonant noise has a separate ten-channel bank, preserving low-frequency buzz independently of hiss. Classic mixes sources into ten fixed channels at 200, 350, 500, 700, 1000, 1400, 2000, 2800, 3900, and 5500 Hz. These are modern design choices, not claimed historical specifications.
5. **PCM output.** Sum channels, remove DC/very low frequencies, normalize peaks with headroom, and export 16-bit mono WAV. Clear uses linear normalization to preserve spectral valleys; Classic retains its soft limiter.

The text engine renders a complete utterance in memory rather than streaming. NumPy handles arrays and SciPy implements filters. A loopback-only Python server serves the browser studio and processes one render at a time. The manual instrument uses Web Audio oscillators, noise, and ten bandpass filters for immediate performance; it is separate from the Python utterance renderer.

## Limitations

The hand-tuned targets and simple timing model still produce robotic speech, particularly on consonant clusters and unfamiliar names. Clear addresses measurable deficiencies in vowel separation, connected voicing and voiced consonants; perceptual intelligibility has not been measured with a listening study.

Natural-language input is English only. Spelling fallbacks are approximate, and the dictionary does not resolve context-dependent words such as “read.” Currencies, dates, acronyms, equations and complex abbreviations lack dedicated normalization. This is an experimental synthesis foundation, not an accessibility-grade voice.

Tune acoustic targets in `voder/phonemes.py`, planning and Classic rendering in `voder/engine.py`, and Clear rendering in `voder/clear.py`.

## Verification

```powershell
python -m unittest discover -s tests -v
```

The 45 tests cover text and pronunciation handling, waveform pitch, fricatives, stop closures, timing, deterministic noise, both modes at all supported sample rates, bounded PCM and WAV/CLI/score export. Acoustic checks cover connected-vowel dropouts, low-frequency voicing in /z/, and second-formant separation between /iy/ and /uw/. Operator tests cover every phoneme, sustained vs transient previews, demo pronunciations, and meter calibration using known tones and silence. Desktop tests cover preview/editor/history behavior, meter selection, cancelled autoplay, and preservation of successful audio after a failed render; they skip when Tk or a display is unavailable. Browser-server tests cover bundled assets, real rendering and audition, WAV/score responses, invalid input, concurrent requests, local-origin checks, and bounded audio retention. These checks do not establish perceptual speech quality.

## Sources and attribution

- Julius O. Smith's [Voder description](https://www.dsprelated.com/freebooks/sasp/Voder.html) documents its manually controlled pitch, ten filter gains, buzz/hiss switching and stop transients.
- Dudley, Riesz and Watkins, [A Synthetic Speaker (1939)](https://www.sciencedirect.com/science/article/abs/pii/S0016003239908161), describes the original instrument.
- Clear follows the general [formant source/filter approach](https://dsprelated.com/freebooks/pasp/Formant_Synthesis_Models.html). Separate voicing/frication paths are also described in [Klatt's 1980 paper](https://www.fon.hum.uva.nl/david/ma_ssp/doc/Klatt-1980-JAS000971.pdf). This is not a full Klatt synthesizer.
- [CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict) supplies pronunciations, not audio. Its license is preserved in `voder/data/CMUDICT-LICENSE`; `voder/data/SOURCE` records the vendored revision.

