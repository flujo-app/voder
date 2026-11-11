import argparse
import json
from pathlib import Path
import sys

from .engine import Voice, synthesize


def main():
    parser = argparse.ArgumentParser(description="Offline Voder-inspired text-to-speech. English text → buzz + hiss → resonant filters → WAV.")
    parser.add_argument("text", nargs="?", help="Text to speak; omit to read stdin")
    parser.add_argument("-o", "--output", type=Path, default=Path("speech.wav"))
    parser.add_argument("--pitch", type=float, default=120, help="Base pitch in Hz (60–300)")
    parser.add_argument("--rate", type=float, default=1, help="Speaking speed multiplier (0.5–2)")
    parser.add_argument("--expression", type=float, default=1, help="Pitch contour strength (0–2)")
    parser.add_argument("--breath", type=float, default=.02, help="Added hiss (0–0.3)")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--seed", type=int, default=1939, help="Noise random seed")
    parser.add_argument("--mode", choices=("clear", "classic"), default="clear", help="Moving formants (default) or original ten-band voice")
    parser.add_argument("--lexicon", type=Path, help="JSON word-to-ARPABET pronunciation overrides")
    parser.add_argument("--phonemes", action="store_true", help="Interpret input as ARPABET instead of text")
    parser.add_argument("--score", type=Path, help="Also write timed filter/pitch controls as JSON")
    parser.add_argument("--play", action="store_true", help="Play the generated WAV (Windows)")
    parser.add_argument("--studio", action="store_true", help="Open the local browser studio")
    parser.add_argument("--native-studio", action="store_true", help="Open the legacy Tk desktop panel")
    args = parser.parse_args()
    if args.studio:
        from .webstudio import main as studio_main
        studio_main()
        return
    if args.native_studio:
        from .studio import main as studio_main
        studio_main()
        return
    if args.play and sys.platform != "win32":
        parser.error("--play uses Windows audio; export the WAV and open it in your audio player")
    text = args.text if args.text is not None else sys.stdin.read()
    try:
        voice = Voice(args.pitch, args.rate, args.expression, args.breath, args.sample_rate, args.seed, args.mode)
        pronunciations = json.loads(args.lexicon.read_text(encoding="utf-8")) if args.lexicon else None
        utterance = synthesize(text, voice, phonemes=args.phonemes, pronunciations=pronunciations)
        utterance.save(args.output)
        if args.score:
            args.score.parent.mkdir(parents=True, exist_ok=True)
            args.score.write_text(json.dumps(utterance.score(), indent=2), encoding="utf-8")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"voder: {exc}\n")
    print(f"Saved {args.output.resolve()} · {utterance.duration:.2f}s · {utterance.sample_rate} Hz")
    print(utterance.phonemes)
    if utterance.unknown_words:
        print("Approximate pronunciations: " + ", ".join(utterance.unknown_words), file=sys.stderr)
    if args.play:
        import winsound
        winsound.PlaySound(str(args.output.resolve()), winsound.SND_FILENAME)
