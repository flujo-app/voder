"""Offline English text normalization, dictionary lookup, and fallback rules."""
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import re
import unicodedata

from .phonemes import PHONES, VOWELS

MAX_TEXT = 4000
SMALL = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()
OVERRIDES = {"voder": "V OW1 D ER0", "vocoder": "V OW1 K OW0 D ER0"}
WEAK_FORMS = {"a": "AH0", "an": "AH0 N", "the": "DH AH0", "and": "AH0 N D",
              "of": "AH0 V", "to": "T AH0", "for": "F ER0", "can": "K AH0 N"}


def number_words(n: int) -> str:
    if n < 20:
        return SMALL[n]
    if n < 100:
        return TENS[n // 10] + (" " + SMALL[n % 10] if n % 10 else "")
    if n < 1000:
        return SMALL[n // 100] + " hundred" + (" " + number_words(n % 100) if n % 100 else "")
    for size, name in [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand")]:
        if n >= size:
            return number_words(n // size) + " " + name + (" " + number_words(n % size) if n % size else "")
    raise ValueError("Invalid number")


def normalize(text: str) -> str:
    if len(text) > MAX_TEXT:
        raise ValueError(f"Text must be at most {MAX_TEXT} characters")
    text = text.replace("’", "'").replace("—", " ").replace("–", " ").replace("…", "...")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\b(mr|mrs|dr|st)\.", lambda m: {"mr": "mister", "mrs": "missus", "dr": "doctor", "st": "saint"}[m[1]], text)
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", text)
    def digits(m: re.Match) -> str:
        whole, dot, fraction = m[0].partition(".")
        spoken = number_words(int(whole)) if len(whole) <= 15 and not (len(whole) > 1 and whole.startswith("0")) else " ".join(SMALL[int(c)] for c in whole)
        return " " + spoken + (" point " + " ".join(SMALL[int(c)] for c in fraction) if dot else "") + " "
    text = re.sub(r"\d+(?:\.\d+)?", digits, text)
    return text.replace("&", " and ").replace("%", " percent ").replace("-", " ")


@lru_cache(maxsize=1)
def lexicon() -> dict[str, tuple[str, ...]]:
    result = {}
    for line in files("voder").joinpath("data/cmudict.dict").read_text(encoding="utf-8").splitlines():
        parts = line.split("#", 1)[0].split()
        if len(parts) < 2 or "(" in parts[0]:
            continue
        result.setdefault(parts[0], tuple(parts[1:]))
    result.update({k: tuple(v.split()) for k, v in OVERRIDES.items()})
    return result


def fallback(word: str) -> tuple[str, ...]:
    """Modest grapheme rules for unknown words; callers report these as uncertain."""
    rules = [("tion", "SH AH0 N"), ("sion", "ZH AH0 N"), ("tch", "CH"), ("igh", "AY1"),
             ("ch", "CH"), ("sh", "SH"), ("th", "TH"), ("ph", "F"), ("ng", "NG"),
             ("qu", "K W"), ("ee", "IY1"), ("ea", "IY1"), ("oo", "UW1"), ("ou", "AW1"),
             ("ow", "OW1"), ("ai", "EY1"), ("ay", "EY1"), ("oi", "OY1"), ("oy", "OY1"),
             ("au", "AO1"), ("er", "ER0"), ("ar", "AA1 R"), ("or", "AO1 R")]
    letters = dict(zip("abcdefghijklmnopqrstuvwxyz", ["AE1", "B", "K", "D", "EH1", "F", "G", "HH", "IH1", "JH", "K", "L", "M", "N", "AA1", "P", "K", "R", "S", "T", "AH1", "V", "W", "K S", "Y", "Z"]))
    out = []
    i = 0
    while i < len(word):
        for spelling, phones in rules:
            if word.startswith(spelling, i):
                out.extend(phones.split()); i += len(spelling); break
        else:
            if word[i] == "e" and i == len(word) - 1 and out:
                i += 1; continue
            phone = letters.get(word[i], "")
            if word[i] == "c" and word[i+1:i+2] in ("e", "i", "y"):
                phone = "S"
            out.extend(phone.split()); i += 1
    return tuple(out)


@dataclass(frozen=True)
class Token:
    phone: str
    stress: int = 0
    word: str = ""
    pause: float = 0
    question: bool = False


def parse_phone(symbol: str, word: str = "") -> Token:
    match = re.fullmatch(r"([A-Z]+)([012])?", symbol.upper())
    if not match or match[1] not in PHONES:
        raise ValueError(f"Unknown phoneme: {symbol!r}; use ARPABET (e.g. HH AH0 L OW1)")
    if match[2] and match[1] not in VOWELS:
        raise ValueError(f"Only vowels can have stress: {symbol!r}")
    return Token(match[1], int(match[2] or 0), word)


def phonemize(text: str, *, raw: bool = False,
              pronunciations: dict[str, str] | None = None) -> tuple[list[Token], list[str]]:
    if len(text) > MAX_TEXT:
        raise ValueError(f"Text must be at most {MAX_TEXT} characters")
    custom = {}
    if pronunciations is not None:
        if not isinstance(pronunciations, dict):
            raise ValueError("Pronunciations must be a JSON object mapping words to ARPABET strings")
        for word, phones in pronunciations.items():
            if not isinstance(word, str) or not re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)*", word):
                raise ValueError(f"Invalid pronunciation word: {word!r}")
            if not isinstance(phones, str) or not phones.strip():
                raise ValueError(f"Pronunciation for {word!r} must be a nonempty ARPABET string")
            symbols = phones.upper().split()
            for symbol in symbols:
                if parse_phone(symbol).phone == "SIL":
                    raise ValueError("Word pronunciations cannot contain SIL")
            custom[word.lower()] = tuple(symbols)
    if raw:
        if custom:
            raise ValueError("Pronunciation overrides cannot be combined with raw phoneme input")
        return [parse_phone(p) for p in text.split()], []
    tokens: list[Token] = []
    unknown: list[str] = []
    words = re.findall(r"[a-z]+(?:'[a-z]+)*|[.,!?;:\n]", normalize(text))
    for index, word in enumerate(words):
        if not word[0].isalpha():
            pause = .30 if word in ".!?" else .15
            # Repeated punctuation is a single boundary, not a chain of silences.
            if tokens and tokens[-1].phone == "SIL":
                prior = tokens.pop()
                pause = max(prior.pause, pause)
                question = prior.question or word == "?"
            else:
                question = word == "?"
            tokens.append(Token("SIL", pause=pause, question=question))
            continue
        phones = custom.get(word) or lexicon().get(word)
        # Reduce unstressed function words inside a phrase. Explicit overrides
        # always win; isolated and phrase-final words keep dictionary stress.
        following = words[index + 1] if index + 1 < len(words) else ""
        if word not in custom and word in WEAK_FORMS and following and following[0].isalpha():
            phones = tuple(WEAK_FORMS[word].split())
            if word == "the":
                next_phones = custom.get(following) or lexicon().get(following, ())
                if next_phones and re.sub(r"\d", "", next_phones[0]) in VOWELS:
                    phones = ("DH", "IY0")
        if phones is None:
            phones = fallback(word)
            if word not in unknown:
                unknown.append(word)
        tokens.extend(parse_phone(p, word) for p in phones)
    return tokens, unknown
