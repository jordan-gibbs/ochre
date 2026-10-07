"""Safety guard between the refiner and the text box (SPEC §5.1 hard rules).

A refiner that answers the dictation, adds content, or eats half of it is worse than no
refiner: the user dictated something and something else got typed. So the core runs every
refined string through ``apply_guard`` and falls back to the raw transcript when it looks
wrong. Each check is cheap and deliberately biased toward returning raw, because raw text is
always an acceptable outcome and a wrong rewrite is not.
"""

from __future__ import annotations

import re

# Content-word comparison ignores these: fillers get removed, and function words come and go
# with punctuation and self-corrections.
# Also spoken numbers/symbols, which normalize() turns into digits, "@" and ".".
_STOPWORDS = frozenset("""
a an the and or but so if then than that this these those to of in on at by for with from as is are
was were be been being am do does did have has had i you he she it we they me him her us them my your
his its our their mine yours what which who whom whose when where why how not no yes ok okay um uh er
ah hmm like just really very well oh you know mean sort kind actually basically literally wait
gonna wanna gotta going want got will would can could should shall may might must there here also
zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen
seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand
million billion point percent dollars dollar oh pm am dot slash colon underscore dash hyphen sign
""".split())

# Words that only an assistant would put in front of the text.
_PREAMBLE = re.compile(
    r"^\s*(sure|certainly|of course|absolutely|okay|ok|alright|got it|here(?:'s| is| are)|"
    r"below is|the (?:cleaned|corrected|refined|edited|polished)|cleaned(?:[- ]up)? (?:text|version)|"
    r"i(?:'m| am) (?:sorry|unable|not able)|i can(?:'t|not)|as an ai|unfortunately)\b[^\n]*?(?::|\n|!|,)",
    re.IGNORECASE)
_ANSWERISH = re.compile(r"\b(?:as an ai|i(?:'m| am) an ai|language model|i don't have access|i cannot)\b",
                        re.IGNORECASE)
_QUESTION_START = re.compile(
    r"^\s*(?:hey \w+,?\s+)?(?:what|who|whom|whose|when|where|why|how|which|is|are|was|were|do|does|did|can|"
    r"could|would|will|should|shall|may|might|have|has|had)\b", re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_WRAP_TAGS = re.compile(r"^\s*<dictation>\s*|\s*</dictation>\s*$", re.IGNORECASE)
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")

MAX_GROWTH = 2.0          # SPEC: refined longer than 2x raw -> raw
GROWTH_SLACK_CHARS = 12   # tiny inputs ("ok" -> "OK.") need a little absolute room
MIN_RECALL = {"clean": 0.6, "polish": 0.45}   # share of raw content words that must survive
MAX_NOVEL = {"clean": 0.25, "polish": 0.45}   # share of refined content words that may be new


def tidy(refined: str) -> str:
    """Strip mechanical artifacts a model may emit around the answer (think blocks, our input
    tags, wrapping quotes) without judging the content."""
    out = _THINK.sub("", refined)
    out = _WRAP_TAGS.sub("", out).strip()
    if len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'“" and out.count(out[0]) == 2:
        out = out[1:-1].strip()
    if out.startswith("“") and out.endswith("”"):
        out = out[1:-1].strip()
    return out


def content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower().replace("’", "'")) if w not in _STOPWORDS]


def apply_guard(raw: str, refined: str, *, mode: str = "clean") -> str:
    """Return ``refined`` (tidied) if it passes every check, else ``raw``."""
    reason = check(raw, refined, mode=mode)
    return raw if reason else tidy(refined)


def check(raw: str, refined: str, *, mode: str = "clean") -> str | None:
    """The failing rule's name, or None if the refinement is acceptable. Exposed for logs/bench."""
    out = tidy(refined or "")
    raw_s = raw.strip()
    if not out:
        return "empty"
    if not raw_s:
        return "empty_raw"
    if len(out) > max(MAX_GROWTH * len(raw_s), len(raw_s) + GROWTH_SLACK_CHARS):
        return "too_long"
    if (m := _PREAMBLE.match(out)) and not _said_first(raw_s, m.group(1)):
        return "preamble"
    if _ANSWERISH.search(out) and not _ANSWERISH.search(raw_s):
        return "assistant_voice"

    raw_words, out_words = content_words(raw_s), content_words(out)
    raw_set, out_set = set(raw_words), set(out_words)
    if len(raw_set) >= 4:
        kept = sum(1 for w in raw_set if w in out_set or _near(w, out_set))
        if kept / len(raw_set) < MIN_RECALL.get(mode, MIN_RECALL["clean"]):
            return "dropped_content"
    novel = [w for w in out_words if w not in raw_set and not _near(w, raw_set)]
    if out_words and len(novel) >= 2 and len(novel) / len(out_words) > MAX_NOVEL.get(mode, MAX_NOVEL["clean"]):
        return "added_content"
    # A dictated question must stay a question: new words + no question mark = probably an answer.
    if _QUESTION_START.match(raw_s) and novel and "?" not in out and not out.rstrip().endswith((":", ";")):
        return "answered_question"
    return None


def _said_first(raw: str, opener: str) -> bool:
    """True if the speaker actually opened with these words ("Okay, so..." dictated as "okay so")."""
    norm = " ".join(_WORD.findall(raw.lower().replace("’", "'"))[:6])
    return norm.startswith(" ".join(_WORD.findall(opener.lower())))


def _stem(word: str) -> str:
    word = word.split("'")[0]
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _near(word: str, vocab: set[str]) -> bool:
    """Cheap morphology tolerance: plural/tense/possessive variants count as the same word."""
    stem = _stem(word)
    return len(stem) >= 3 and any(_stem(v) == stem for v in vocab)
