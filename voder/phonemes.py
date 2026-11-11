"""Approximate American English acoustic targets, expressed as ARPABET.

These are hand-tuned synthesis parameters, not measurements of the 1939 Voder.
The three formant targets become gains on a fixed ten-channel filter bank.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Phone:
    formants: tuple[float, float, float]
    duration: float = .10
    voice: float = 1.0
    noise: float = .015
    kind: str = "vowel"
    end: tuple[float, float, float] | None = None


PHONES = {
    "AA": Phone((730, 1090, 2440)),
    "AE": Phone((660, 1720, 2410)),
    "AH": Phone((580, 1190, 2390)),
    "AO": Phone((570, 840, 2410)),
    "AW": Phone((730, 1090, 2440), .16, end=(350, 850, 2400)),
    "AY": Phone((730, 1090, 2440), .16, end=(300, 2200, 3000)),
    "EH": Phone((530, 1840, 2480)),
    "ER": Phone((490, 1350, 1690)),
    "EY": Phone((530, 1840, 2480), .14, end=(300, 2200, 3000)),
    "IH": Phone((390, 1990, 2550)),
    "IY": Phone((270, 2290, 3010)),
    "OW": Phone((570, 840, 2410), .15, end=(300, 750, 2300)),
    "OY": Phone((570, 840, 2410), .16, end=(300, 2200, 3000)),
    "UH": Phone((440, 1020, 2240)),
    "UW": Phone((300, 870, 2240)),
    "B": Phone((250, 900, 2200), .085, .4, .7, "stop"),
    "D": Phone((300, 1700, 2800), .075, .4, .9, "stop"),
    "G": Phone((300, 1500, 2400), .085, .4, .8, "stop"),
    "P": Phone((300, 1000, 2400), .09, 0, .8, "stop"),
    "T": Phone((300, 1800, 3500), .08, 0, 1, "stop"),
    "K": Phone((300, 1600, 2700), .09, 0, 1, "stop"),
    "CH": Phone((400, 2000, 3500), .13, 0, .8, "affricate"),
    "JH": Phone((400, 1800, 3200), .12, .45, .65, "affricate"),
    "F": Phone((400, 1800, 4000), .105, 0, .5, "fricative"),
    "V": Phone((400, 1600, 3500), .09, .6, .35, "fricative"),
    "TH": Phone((400, 1600, 4000), .10, 0, .4, "fricative"),
    "DH": Phone((400, 1600, 3500), .075, .65, .25, "fricative"),
    "S": Phone((400, 2500, 5500), .12, 0, .7, "fricative"),
    "Z": Phone((400, 2500, 5000), .11, .55, .5, "fricative"),
    "SH": Phone((400, 1800, 3500), .12, 0, .7, "fricative"),
    "ZH": Phone((400, 1800, 3500), .11, .55, .5, "fricative"),
    "HH": Phone((600, 1400, 2500), .075, 0, .22, "breath"),
    "M": Phone((250, 1000, 2200), .09, .8, 0, "nasal"),
    "N": Phone((250, 1500, 2500), .085, .8, 0, "nasal"),
    "NG": Phone((250, 1500, 2200), .095, .8, 0, "nasal"),
    "L": Phone((350, 1100, 2700), .085, .85, .01, "liquid"),
    "R": Phone((350, 1300, 1650), .085, .85, .01, "liquid"),
    "W": Phone((280, 650, 2200), .075, .85, .01, "liquid"),
    "Y": Phone((270, 2200, 3000), .075, .85, .01, "liquid"),
    "SIL": Phone((500, 1500, 2500), .12, 0, 0, "silence"),
}

VOWELS = frozenset(k for k, v in PHONES.items() if v.kind == "vowel")
