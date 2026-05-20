"""Pure-string encoders, obfuscators, and mutators.

Deterministic, no LLM dependency. The attacker can ask for a payload to be
wrapped in any of these to evade keyword-based filters; the orchestrator can
round-robin them in library-only mode.
"""

from __future__ import annotations

import base64
import codecs
from typing import Callable, Dict


def base64_encode(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def rot13(text: str) -> str:
    return codecs.encode(text, "rot_13")


_LEET_MAP = str.maketrans({
    "a": "4", "A": "4",
    "e": "3", "E": "3",
    "i": "1", "I": "1",
    "o": "0", "O": "0",
    "s": "5", "S": "5",
    "t": "7", "T": "7",
    "g": "9", "G": "9",
    "b": "8", "B": "8",
})


def leetspeak(text: str) -> str:
    return text.translate(_LEET_MAP)


# Cyrillic / Greek lookalikes — ASCII chars replaced with visually identical
# Unicode characters. Bypasses naive keyword filters but renders identically
# to a human reader.
_HOMOGLYPH = {
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у",
    "x": "х", "i": "і", "k": "к", "m": "м", "h": "һ", "t": "т",
    "A": "А", "E": "Е", "O": "О", "P": "Р", "C": "С", "T": "Т",
}


def unicode_confusables(text: str) -> str:
    return "".join(_HOMOGLYPH.get(c, c) for c in text)


_ZERO_WIDTH = "​"  # zero-width space


def zero_width_inject(text: str, every: int = 3) -> str:
    """Drop a zero-width space every N chars — invisible to humans, breaks
    string-match guardrails."""
    out = []
    for i, c in enumerate(text):
        out.append(c)
        if i and i % every == 0:
            out.append(_ZERO_WIDTH)
    return "".join(out)


def reverse_text(text: str) -> str:
    return text[::-1]


_MORSE = {
    "a": ".-", "b": "-...", "c": "-.-.", "d": "-..", "e": ".", "f": "..-.",
    "g": "--.", "h": "....", "i": "..", "j": ".---", "k": "-.-", "l": ".-..",
    "m": "--", "n": "-.", "o": "---", "p": ".--.", "q": "--.-", "r": ".-.",
    "s": "...", "t": "-", "u": "..-", "v": "...-", "w": ".--", "x": "-..-",
    "y": "-.--", "z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
    " ": "/",
}


def morse_encode(text: str) -> str:
    return " ".join(_MORSE.get(c.lower(), c) for c in text)


def pig_latin(text: str) -> str:
    """Trivial obfuscation that fools simple regex filters."""
    out = []
    for word in text.split():
        if not word:
            continue
        if word[0].lower() in "aeiou":
            out.append(word + "way")
        else:
            out.append(word[1:] + word[0] + "ay")
    return " ".join(out)


# Registry the LLM and the round-robin selector can both use.
ALL_ENCODERS: Dict[str, Callable[[str], str]] = {
    "base64": base64_encode,
    "rot13": rot13,
    "leetspeak": leetspeak,
    "unicode_confusables": unicode_confusables,
    "zero_width_inject": zero_width_inject,
    "reverse": reverse_text,
    "morse": morse_encode,
    "pig_latin": pig_latin,
}


def render_encoder_catalogue() -> str:
    """Compact, LLM-readable description of available encoders."""
    return "\n".join(
        f"- {name}: {fn.__doc__.strip().splitlines()[0] if fn.__doc__ else 'transformation'}"
        for name, fn in ALL_ENCODERS.items()
    )
