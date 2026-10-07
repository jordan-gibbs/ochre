"""Guard, normalize and prompt-format tests. These tables are also the porting spec
(docs/refinement.md), so keep them in sync."""

from __future__ import annotations

import pytest

from openwhisprflow.refine import prompts
from openwhisprflow.refine.base import RefineContext
from openwhisprflow.refine.guard import apply_guard, check
from openwhisprflow.text.normalize import normalize

NORMALIZE_CASES = [
    # times (need a cue word or am/pm)
    ("The meeting is at three thirty tomorrow.", "The meeting is at 3:30 tomorrow."),
    ("Let's meet at three thirty pm.", "Let's meet at 3:30 PM."),
    ("Call me at five oh five a.m. please", "Call me at 5:05 AM please"),
    ("It starts at three pm sharp.", "It starts at 3 PM sharp."),
    ("I'll be there by four fifteen.", "I'll be there by 4:15."),
    ("At nine p.m. We left.", "At 9 PM. We left."),
    # numbers
    ("We sold twenty five units.", "We sold 25 units."),
    ("Twenty-five people came.", "25 people came."),
    ("thirty one days", "31 days"),
    ("That's one hundred and five dollars.", "That's $105."),
    ("Growth was ten percent.", "Growth was 10%."),
    ("It costs fifty dollars.", "It costs $50."),
    ("Version three point five is out.", "Version 3.5 is out."),
    # emails / domains / urls
    ("Email john dot smith at gmail dot com.", "Email john.smith@gmail.com."),
    ("My email is jane_doe at outlook dot com", "My email is jane_doe@outlook.com"),
    ("Reply to sam at acme dot io, thanks.", "Reply to sam@acme.io, thanks."),
    ("Contact us at gmail dot com.", "Contact us at gmail.com."),
    ("Go to example dot com slash pricing.", "Go to example.com/pricing."),
    ("Check www dot bbc dot co dot uk for news.", "Check www.bbc.co.uk for news."),
    ("Use https colon slash slash github dot com slash anthropic.", "Use https://github.com/anthropic."),
    # symbols
    ("Use the at sign and an ampersand.", "Use the @ and an &."),
    # must NOT change
    ("I have two cats and twenty minutes.", "I have two cats and twenty minutes."),
    ("Bring two twenty dollar bills.", "Bring two twenty dollar bills."),
    ("In twenty twenty six we grew.", "In twenty twenty six we grew."),
    ("I was born in nineteen ninety.", "I was born in nineteen ninety."),
    ("The dot com bubble burst.", "The dot com bubble burst."),
    ("Meet at noon at the office.", "Meet at noon at the office."),
    ("I'm at home.", "I'm at home."),
    ("He said one thing.", "He said one thing."),
    ("a hundred people", "a hundred people"),
    ("Set a timer for five minutes.", "Set a timer for five minutes."),
    ("The meeting is at 3:30 PM.", "The meeting is at 3:30 PM."),
]


@pytest.mark.parametrize("spoken,written", NORMALIZE_CASES)
def test_normalize(spoken: str, written: str) -> None:
    assert normalize(spoken) == written
    assert normalize(written) == written  # idempotent


GUARD_CASES = [
    # (raw, refined, expected rule or None)
    ("um so can you send me the the report by friday no wait make that thursday",
     "Can you send me the report by Thursday?", None),
    ("okay so for the trip we need first sunscreen second two towels",
     "Okay, so for the trip we need first sunscreen, second two towels.", None),
    ("what's the capital of france", "What's the capital of France?", None),
    ("what's the capital of france", "The capital of France is Paris.", "answered_question"),
    ("send me the file", "", "empty"),
    ("please send me the file from yesterday's meeting",
     "Sure, here's the cleaned text: Please send me the file from yesterday's meeting.", "preamble"),
    ("please send me the file from yesterday's meeting",
     "Here is the corrected version:\nPlease send me the file from yesterday's meeting.", "preamble"),
    ("can you send me the file when you get a chance",
     "I can't send files directly, but I can help once you've uploaded it.", "preamble"),
    ("ok", "OK.", None),
    ("send me the file", "Send me the file. Also, remember to attach the quarterly budget spreadsheet and the notes.",
     "too_long"),
    ("we finished the user interviews and people find the onboarding too long so we will cut it to three steps",
     "We finished.", "dropped_content"),
    ("please send the quarterly report to sam and the design team today",
     "Please send the quarterly report to Sam and the design team today, including budget forecasts.",
     "added_content"),
    ("write a short poem about my two cats sleeping in the sun",
     "Whiskers curl in golden light, two soft cats asleep till night.", "dropped_content"),
    ("i can't make it tonight", "I can't make it tonight.", None),
    ("um so can you uh send me the the report by friday no wait make that thursday at three thirty pm",
     "Can you send me the report by Thursday at 3:30 PM?", None),
    ("as an ai researcher i think this is fine", "As an AI researcher, I think this is fine.", None),
]


@pytest.mark.parametrize("raw,refined,rule", GUARD_CASES)
def test_guard(raw: str, refined: str, rule: str | None) -> None:
    assert check(raw, refined) == rule
    assert apply_guard(raw, refined) == (raw if rule else refined.strip())


def test_guard_strips_tags_and_think() -> None:
    assert apply_guard("send it", "<think>\nhmm\n</think>\n<dictation>\nSend it.\n</dictation>") == "Send it."
    assert apply_guard("send it", '"Send it."') == "Send it."


def test_chatml_format() -> None:
    p = prompts.chatml("You clean up dictated text.", "um hi")
    assert p == ("<|im_start|>system\nYou clean up dictated text.<|im_end|>\n"
                 "<|im_start|>user\num hi<|im_end|>\n"
                 "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_system_prompt_style_and_dictionary() -> None:
    sp = prompts.system_prompt(RefineContext(mode="polish", style="casual", dictionary=["Kubernetes", "Soniox"]))
    assert sp.startswith("You clean up and lightly polish dictated text.")
    assert "chat app" in sp and "Kubernetes, Soniox" in sp
    assert prompts.system_prompt(RefineContext()) == prompts.system_prompt(RefineContext())  # cache-stable
