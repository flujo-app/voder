# What we took from the Grokbot prototype

Reviewed the local `sD7atSPjfTljBRoU-grok-workspace` source and its supplied desktop screenshot on 2026-09-12. The imported workspace was inspected without modifying or running its server. This was a source/design comparison, not a listening evaluation of its browser synthesis.

## Integrated into version 0.3

| Grokbot idea | Adaptation in this project |
| --- | --- |
| Operator's card with phoneme buttons and example words (`src/components/voder/Desk.tsx`) | Native desktop card covering our full 39-phone inventory, with short previews, editable voice settings, and preservation of the text editor. |
| Four demonstration phrases (`src/lib/voder/data.ts`) | “She saw me,” “How are you?”, “Good afternoon,” and “The Voder speaks” load into the editor and use our dictionary, timing, and synthesis. |
| Operator-school lessons (`Desk.tsx`, `data.ts`) | Four concise experiments explaining voicing, vowel resonances, pitch, and stop consonants, with wording specific to this engine. |
| Actual waveform/RMS/spectrum feedback (`src/components/voder/Meters.tsx`) | Calibrated short-time FFT band power and RMS/peak dBFS from rendered PCM; a selector distinguishes measured audio from planned control targets. |
| Dedicated transient stop behavior (`src/lib/voder/engine.ts`) | Phoneme previews preserve short stop/affricate releases instead of stretching them like vowels. Existing TTS stop synthesis remains intact. |

The operator and demo concepts are adapted; the TypeScript engine and its platform scaffold were not copied. New Python code uses the existing engine and inventory.

## Version 0.4: the instrument and its story

The default studio now runs in a local browser. It adopts the prototype's warm brass, ivory keys, dark instrument cabinet, serif typography, and separation into speaking, performance, lessons, and history. The layout is newly implemented in bundled HTML/CSS/JavaScript, with no Node build or remote assets.

- Ten continuously held resonators, a buzz/hiss wrist bar, a pitch pedal, and dedicated P/T/K bursts use Web Audio. Keyboard and pointer ownership are tracked independently, and blur, page changes, and Escape release the controls.
- Operator school offers audible demonstrations, vowel comparisons, a question-intonation experiment, and stop previews through the existing Python engine.
- The story introduces Dudley, Helen Harper, and the two 1939 fairs. Links to Julius O. Smith and Smithsonian support the account; it distinguishes our modern synthesis choices from the historical machine.
- The monitor responds to real browser output from both speech and the manual instrument. Volume zero is silence. Frequency calculations use the actual analyser bin count. The Python meter calibration tests continue to apply to the legacy Tk meter; browser FFT bars are a separate visualization.
- Arbitrary text still goes through our dictionary, gesture planner, and Clear/Classic renderers. WAV/score export and previous-take comparison remain available.

## Useful ideas not copied literally

The prototype's immediate keyboard/touch performance and instrument-style presentation are now reflected in the new browser studio. Manual performance and arbitrary text rendering have distinct pages and synthesis paths. Phoneme previews remain finite auditions; the ivory keys are continuous controls.

Its engine uses fixed-band envelopes and a single buzz/hiss mix. Our Clear mode already has moving formants and separate voiced/noise paths, plus stress, a full pronunciation dictionary, reproducible noise and WAV/score export. There is no evidence from this source review that replacing those with the browser engine would improve speech quality. Its band centers differ from ours, so its gain presets are not directly transferable.

## Source issues we avoided

- `engine.ts`, `setVolume`: the gain formula adds `0.02`, so a volume value of zero does not produce a true mute. The store's mute action calls that same method.
- `data.ts`, `compile`: pitch interpolation divides the token index by the number of pitch intervals rather than mapping the full token span across the pitch array. Several demo contours therefore never reach their last pitch target.
- `Meters.tsx`: a 2048-element frequency buffer is used with `fftSize = 2048`, whose one-sided frequency-bin count is 1024. The code derives bin width from the larger buffer length, so frequency labeling is off by a factor of two. Our meter derives bins from the actual FFT and sample rate.
- Some demo voiced consonants are substituted with unvoiced stops or fricatives (G→K, D→T, DH→TH). Our demonstrations retain the original ARPABET consonants through dictionary lookup.

Tests cover preview inventory/durations, preserving transient stops, editor/history behavior, cancellation of pending autoplay, and meter calibration against known sine waves and silence. These establish behavior and measurement correctness, not perceptual intelligibility.
