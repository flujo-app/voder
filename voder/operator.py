"""Operator-card and demo concepts adapted from the user's Grokbot prototype.

Uses this project's ARPABET inventory and renderers, not the prototype's
hand-authored ten-band phoneme values or browser event scheduler.
"""
from dataclasses import replace
import math

from .engine import Voice, Utterance, plan, _render_utterance
from .phonemes import PHONES, VOWELS
from .text import Token, parse_phone

EXAMPLES = dict(zip(
    "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW B D G P T K CH JH F V TH DH S Z SH ZH HH M N NG L R W Y".split(),
    "father cat cup saw how my bed bird day sit see go boy book boot bee day go pea tea key chin judge fan van thin this sip zip ship vision hat hum no sing low red we yes".split(),
))

DEMOS = {
    "She saw me": "She saw me.",
    "How are you?": "How are you?",
    "Good afternoon": "Good afternoon.",
    "The Voder speaks": "The Voder speaks.",
}

LESSONS = (
    ("Buzz and hiss", "Compare S and Z: both have hiss, but Z also has a low voiced component. The audio meter shows the difference."),
    ("Vowel resonances", "Compare IY (see), AA (father), and UW (boot). Keep pitch fixed: the changing resonances, rather than pitch, identify the vowel."),
    ("Pitch and intonation", "Load 'How are you?' and compare expression 0 with expression 1. Pitch changes the question contour; it does not select a vowel."),
    ("Stop consonants", "P, T, and K have a quiet closure followed by a short burst. Their previews keep that short release instead of stretching it into a held noise."),
)


def audition_phone(symbol: str, voice: Voice | None = None, *, duration: float = .6) -> Utterance:
    """Preview one sound. Sustain vowels/fricatives, but keep stops transient.

    `duration` is the preview span before normal leading/trailing padding;
    it is independent of the text speaking-rate control.
    """
    if not math.isfinite(duration) or not .2 <= duration <= 2:
        raise ValueError("duration must be between 0.2 and 2 seconds")
    token = parse_phone(symbol)
    if token.phone in VOWELS and not any(c.isdigit() for c in symbol):
        token = replace(token, stress=1)
    voice = voice or Voice()
    gestures = plan([token], voice)
    g = gestures[0]
    if PHONES[token.phone].kind in ("stop", "affricate"):
        # Do not turn a 20 ms release into a 200 ms hiss when auditioning.
        g = replace(g, duration=min(g.duration, duration))
        silence = plan([Token("SIL")], voice)[0]
        gestures = [g]
        if duration > g.duration:
            gestures.append(replace(silence, start=g.start + g.duration, duration=duration - g.duration))
    else:
        gestures = [replace(g, duration=duration)]
    return _render_utterance(gestures, voice, [])
