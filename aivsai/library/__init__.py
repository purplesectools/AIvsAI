"""Attack primitive library — building blocks the attacker LLM composes with.

The library has three layers:

  primitives:  pure text transformations (encoders, mutators) — deterministic
  wrappers:    parameterizable prompt frames ({goal} placeholders) — templates
  probes:      high-level curated attacks combining primitives + wrappers

The attacker LLM never picks a probe verbatim. In hybrid mode it sees the
catalogue as inspiration and composes original payloads adapted to the user's
goal. In library-only mode the orchestrator round-robins through probes
without an LLM (Garak-style fast scan).
"""

from .primitives import (
    base64_encode,
    rot13,
    leetspeak,
    unicode_confusables,
    zero_width_inject,
    reverse_text,
    morse_encode,
    pig_latin,
    ALL_ENCODERS,
)
from .wrappers import (
    WRAPPERS,
    Wrapper,
    render_wrapper_catalogue,
)
from .extraction import (
    SYSTEM_PROMPT_EXTRACTION_PHRASINGS,
    REFUSAL_SUPPRESSION_HEADERS,
    MULTILINGUAL_OVERRIDES,
)
from .probes import (
    Probe,
    PROBES,
    probes_for_goal,
    render_library_catalogue,
)
from .mutator import mutate

__all__ = [
    "base64_encode", "rot13", "leetspeak", "unicode_confusables",
    "zero_width_inject", "reverse_text", "morse_encode", "pig_latin",
    "ALL_ENCODERS",
    "WRAPPERS", "Wrapper", "render_wrapper_catalogue",
    "SYSTEM_PROMPT_EXTRACTION_PHRASINGS", "REFUSAL_SUPPRESSION_HEADERS",
    "MULTILINGUAL_OVERRIDES",
    "Probe", "PROBES", "probes_for_goal", "render_library_catalogue",
    "mutate",
]
