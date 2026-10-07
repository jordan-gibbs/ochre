"""Hands-free control phrases in transcripts: "<phrase> stop|done|send|cancel|scratch that".

The wake model hears the word; the transcript decides what it meant. A control phrase only counts
when it is the **tail** of a phrase transcript (nothing but punctuation after it), so "I need to
transcribe stop-motion footage" or "transcribe send the file to Bob" never end a session.

Matching is fuzzy because ASR spells the phrase many ways: "Transcribe, stop.", "transcribes
stop", "Transcribe Send!", "trans scribe stop", "transcribed done". The phrase tokens just before
the command word are joined (up to 3 tokens, so split spellings work) and compared to the phrase
with a character similarity ratio. Near words ("describe", "subscribe", "transcript") stay below the
default ratio; when the wake model itself fired moments ago (``armed``) the ratio is relaxed and a
bare command ("Stop.") is accepted on its own, since the audio already said the wake word.

Also here: :func:`strip_leading_phrase` removes the wake word the ASR transcribed at the start of a
session ("Transcribe, hey Sarah" -> "Hey Sarah") and :func:`starts_with_phrase` is the false-wake
check (the session's first words must be the wake word, after optional "hey/okay/so").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum

DEFAULT_RATIO = 0.84   # "transcribes"/"trans scribe" ~0.95 pass; "transcript" 0.80, "describe" 0.67 fail
ARMED_RATIO = 0.7      # wake model fired: accept rougher ASR ("transcript stop")
MAX_PHRASE_TOKENS = 3

# Words allowed before the wake word at the start of a session ("Hey transcribe, ...").
LEAD_FILLERS = frozenset({
    "hey", "hi", "hello", "okay", "ok", "so", "um", "uh", "umm", "uhh", "oh", "alright", "right", "well",
    "and", "a", "the", "yo",
})


class Action(StrEnum):
    NONE = "none"          # ordinary dictation: keep going
    FINISH = "finish"      # "<phrase> stop" / "<phrase> done": insert
    SEND = "send"          # "<phrase> send": insert, then press Enter
    CANCEL = "cancel"      # "<phrase> cancel": discard the session
    SCRATCH = "scratch"    # "<phrase> scratch that": drop the last phrase


# command word sequences (lower-case tokens) -> action. ASR inflections included on purpose:
# "transcribe stop" is often heard as "transcribes stop"/"transcribe stopped".
_COMMANDS: dict[tuple[str, ...], Action] = {
    ("stop",): Action.FINISH, ("stops",): Action.FINISH, ("stopped",): Action.FINISH,
    ("done",): Action.FINISH, ("dun",): Action.FINISH,
    ("send",): Action.SEND, ("sends",): Action.SEND, ("sent",): Action.SEND, ("send", "it"): Action.SEND,
    ("cancel",): Action.CANCEL, ("canceled",): Action.CANCEL, ("cancelled",): Action.CANCEL,
    ("cancels",): Action.CANCEL,
    ("scratch", "that"): Action.SCRATCH, ("scratched", "that"): Action.SCRATCH, ("scratch", "this"): Action.SCRATCH,
    ("scratch",): Action.SCRATCH,
}
_MAX_CMD = max(len(k) for k in _COMMANDS)

# A word: letters/digits/apostrophes, with internal hyphens kept ("stop-motion" is one token,
# so it can never be read as the command "stop").
_TOKEN = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
_TRAIL_JUNK = re.compile(r"[\s,;:\-–—…]+$")


@dataclass(frozen=True)
class Control:
    """Result of :func:`parse_control`."""

    action: Action
    text: str              # transcript with the control phrase (and dangling separators) removed
    matched: str = ""      # the original span that was recognised, e.g. "Transcribe, stop."
    score: float = 0.0     # phrase similarity (1.0 = exact; 0.0 for a bare command when armed)

    @property
    def is_command(self) -> bool:
        return self.action is not Action.NONE


@dataclass(frozen=True)
class _Tok:
    norm: str     # lower-case, apostrophes/hyphens removed
    start: int
    end: int


def _tokens(text: str) -> list[_Tok]:
    out = []
    for m in _TOKEN.finditer(text):
        norm = re.sub(r"['’\-]", "", m.group(0).lower())
        out.append(_Tok(norm, m.start(), m.end()))
    return out


def _squash(s: str) -> str:
    return re.sub(r"[^\w]|_", "", s.lower())


def phrase_ratio(candidate: str, phrase: str) -> float:
    """Similarity of a candidate (spaces ignored) to the phrase: 1.0 exact, ~0.95 one extra letter."""
    a, b = _squash(candidate), _squash(phrase)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _piece_of(token: str, phrase: str) -> bool:
    """Is ``token`` plausibly a piece of the phrase ("trans", "scribe")? Most of its letters must
    appear in order in the phrase. Stops junk words being glued onto it ("ing"+"transcribe")."""
    p = _squash(phrase)
    if not token or not p:
        return False
    blocks = SequenceMatcher(None, token, p, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks) / len(token) >= 0.8


def _match_tokens(toks: list[_Tok], i: int, j: int, phrase: str) -> float:
    """Ratio of ``toks[i:j]`` (joined) to the phrase; 0 if a multi-token join has a foreign piece."""
    if j - i > 1 and len(phrase.split()) == 1 and not all(_piece_of(t.norm, phrase) for t in toks[i:j]):
        return 0.0
    return phrase_ratio("".join(t.norm for t in toks[i:j]), phrase)


def _best_phrase_end_match(toks: list[_Tok], end: int, phrases: list[str],
                           min_ratio: float) -> tuple[int, float] | None:
    """Tokens ``toks[i:end]`` that best match a phrase (1..MAX tokens, ending right before ``end``).

    Returns (i, ratio) or None. Prefers the higher ratio; on ties, fewer tokens (keeps user words).
    """
    best: tuple[int, float] | None = None
    for phrase in phrases:
        n_words = len(phrase.split())
        for k in range(1, min(MAX_PHRASE_TOKENS + n_words - 1, end) + 1):
            i = end - k
            r = _match_tokens(toks, i, end, phrase)
            if r >= min_ratio and (best is None or r > best[1] + 1e-9):
                best = (i, r)
    return best


def _clean_prefix(text: str) -> str:
    """Text kept before a control phrase: drop dangling separators ("Hello world," -> "Hello world")."""
    return _TRAIL_JUNK.sub("", text).rstrip()


def parse_control(text: str, phrase: str = "transcribe", *, aliases: tuple[str, ...] | list[str] = (),
                  armed: bool = False, min_ratio: float | None = None) -> Control:
    """Find a trailing control phrase in one phrase transcript.

    ``aliases``: extra spellings that count as the phrase (e.g. a user's ASR habit). ``armed``: the
    wake model fired during this phrase, so be lenient (lower ratio, bare command accepted).
    """
    toks = _tokens(text)
    if not toks:
        return Control(Action.NONE, text)
    phrases = [phrase, *aliases]
    ratio = min_ratio if min_ratio is not None else (ARMED_RATIO if armed else DEFAULT_RATIO)
    # Longest command first: "... scratch that" must be read as ("scratch", "that"), and
    # "... send it" as ("send", "it"), before any shorter tail is considered.
    for n_cmd in range(min(_MAX_CMD, len(toks)), 0, -1):
        action = _COMMANDS.get(tuple(t.norm for t in toks[-n_cmd:]))
        if action is None:
            continue
        cmd_start = len(toks) - n_cmd
        m = _best_phrase_end_match(toks, cmd_start, phrases, ratio)
        if m is not None:
            i, r = m
            span_start = toks[i].start
            return Control(action, _clean_prefix(text[:span_start]), text[span_start:].strip(), r)
        if armed and cmd_start == 0:
            # The wake model heard the phrase but the ASR dropped it: "Stop." on its own.
            return Control(action, "", text.strip(), 0.0)
    return Control(Action.NONE, text)


def ends_with_phrase(text: str, phrase: str = "transcribe", *, aliases: tuple[str, ...] | list[str] = (),
                     min_ratio: float = DEFAULT_RATIO) -> tuple[bool, str]:
    """Does the transcript END with the bare phrase ("... and that's it. Transcribe.")?

    The segmenter may cut between "transcribe" and "stop"; the controller holds such a phrase
    until the next one shows whether a command followed. Returns (yes, text without the phrase).
    """
    toks = _tokens(text)
    if not toks:
        return False, text
    m = _best_phrase_end_match(toks, len(toks), [phrase, *aliases], min_ratio)
    if m is None:
        return False, text
    return True, _clean_prefix(text[:toks[m[0]].start])


def starts_with_phrase(text: str, phrase: str = "transcribe", *, aliases: tuple[str, ...] | list[str] = (),
                       min_ratio: float = DEFAULT_RATIO, max_fragments: int = 0) -> bool:
    """False-wake check: the first words are the phrase, after optional fillers ("Hey, transcribe ...").

    ``max_fragments``: arbitrary tokens allowed before it (a word cut in half at the start of the
    pre-roll). 0 when the pre-roll began in silence.
    """
    return _leading_span(text, [phrase, *aliases], min_ratio, max_fragments) is not None


def strip_leading_phrase(text: str, phrase: str = "transcribe", *, aliases: tuple[str, ...] | list[str] = (),
                         min_ratio: float = DEFAULT_RATIO, max_fragments: int = 0) -> str:
    """Remove the wake word (and fillers before it) from the start: "Transcribe, hey Sarah." -> "Hey Sarah."."""
    end = _leading_span(text, [phrase, *aliases], min_ratio, max_fragments)
    if end is None:
        return text
    rest = text[end:].lstrip(" \t\r\n,.;:!?-–—…")
    if rest and rest[0].islower() and text[:1].isupper():
        rest = rest[0].upper() + rest[1:]  # the transcript started a sentence; keep it started
    return rest


def _leading_span(text: str, phrases: list[str], min_ratio: float, max_fragments: int) -> int | None:
    toks = _tokens(text)
    lead = 0
    frags = 0
    while lead < len(toks):
        n = len(toks) - lead
        best: tuple[int, float] | None = None
        for p in phrases:
            for k in range(1, min(MAX_PHRASE_TOKENS + len(p.split()) - 1, n) + 1):
                r = _match_tokens(toks, lead, lead + k, p)
                if r >= min_ratio and (best is None or r > best[1] + 1e-9):
                    best = (lead + k, r)
        if best is not None:
            return toks[best[0] - 1].end
        if toks[lead].norm in LEAD_FILLERS:
            lead += 1
        elif frags < max_fragments:
            frags += 1
            lead += 1
        else:
            return None
    return None
