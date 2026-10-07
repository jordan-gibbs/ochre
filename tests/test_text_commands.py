"""text/commands.py: trailing control phrases, wake-word stripping and the false-wake check."""

from __future__ import annotations

import pytest

from openwhisprflow.text.commands import (
    Action,
    ends_with_phrase,
    parse_control,
    phrase_ratio,
    starts_with_phrase,
    strip_leading_phrase,
)

# ---------------------------------------------------------------- parse_control: positives


@pytest.mark.parametrize("text, action, kept", [
    ("transcribe stop", Action.FINISH, ""),
    ("Transcribe stop.", Action.FINISH, ""),
    ("transcribe, stop.", Action.FINISH, ""),
    ("Transcribe. Stop.", Action.FINISH, ""),
    ("transcribes stop", Action.FINISH, ""),
    ("Transcribe stopped.", Action.FINISH, ""),
    ("trans scribe stop", Action.FINISH, ""),
    ("Trans-scribe, stop!", Action.FINISH, ""),
    ("transcribed done", Action.FINISH, ""),
    ("Transcribe done.", Action.FINISH, ""),
    ("TRANSCRIBE STOP", Action.FINISH, ""),
    ("Transcribe Send!", Action.SEND, ""),
    ("transcribe sent", Action.SEND, ""),
    ("Transcribe, send it.", Action.SEND, ""),
    ("Transcribe cancel.", Action.CANCEL, ""),
    ("transcribe cancelled", Action.CANCEL, ""),
    ("Transcribe, scratch that.", Action.SCRATCH, ""),
    ("transcribe scratch", Action.SCRATCH, ""),
    ("Hello world, transcribe send.", Action.SEND, "Hello world"),
    ("See you at five. Transcribe stop.", Action.FINISH, "See you at five."),
    ("Thanks so much! Transcribe, send!", Action.SEND, "Thanks so much!"),
    ("Let me think about it — transcribe stop", Action.FINISH, "Let me think about it"),
    ("I'll call you later transcribe done", Action.FINISH, "I'll call you later"),
    ("Draft one... transcribe cancel", Action.CANCEL, "Draft one..."),
    ("Ship it on Friday, transcribe scratch that.", Action.SCRATCH, "Ship it on Friday"),
    ("transcribe stop   ", Action.FINISH, ""),
    ("transcribe stop…", Action.FINISH, ""),
    ("Transcribe stop?", Action.FINISH, ""),
])
def test_trailing_commands(text: str, action: Action, kept: str) -> None:
    c = parse_control(text)
    assert c.action is action, (text, c)
    assert c.text == kept
    assert c.is_command


# ---------------------------------------------------------------- parse_control: must NOT trigger


@pytest.mark.parametrize("text", [
    "I need to transcribe stop-motion footage",
    "I need to transcribe stop-motion",
    "Can you transcribe stop signs in the video",
    "transcribe send the file to Bob",
    "transcribe stop the recording please",
    "We should transcribe, then send it to legal for review.",
    "Please describe stop.",            # near word
    "Subscribe, stop.",
    "prescribe done",
    "Read the transcript. Send.",       # transcript (0.80) < default ratio
    "Don't stop.",
    "Stop.",                            # bare command, not armed
    "send",
    "Scratch that.",
    "I scratched that car door",
    "The tribe sent a message",
    "",
    "   ",
    "...",
    "stop transcribe",                  # wrong order
    "transcribe",                       # wake word alone is not a command
])
def test_not_commands(text: str) -> None:
    c = parse_control(text)
    assert c.action is Action.NONE, (text, c)
    assert c.text == text
    assert not c.is_command


def test_armed_accepts_bare_command_and_rough_asr() -> None:
    assert parse_control("Stop.", armed=True).action is Action.FINISH
    assert parse_control("Send!", armed=True).action is Action.SEND
    assert parse_control("Scratch that.", armed=True).action is Action.SCRATCH
    c = parse_control("All done here, transcript stop.", armed=True)
    assert c.action is Action.FINISH and c.text == "All done here"
    # armed still never acts mid-sentence or on unrelated words before the command
    assert parse_control("Don't stop.", armed=True).action is Action.NONE
    assert parse_control("we must stop now", armed=True).action is Action.NONE
    assert parse_control("Please describe stop.", armed=True).action is Action.NONE


def test_custom_phrase_and_aliases() -> None:
    assert parse_control("computer stop", phrase="computer").action is Action.FINISH
    assert parse_control("Hey computer, send.", phrase="hey computer").action is Action.SEND
    assert parse_control("transcribe stop", phrase="computer").action is Action.NONE
    c = parse_control("hello trance cry stop", aliases=["trance cry"])
    assert c.action is Action.FINISH and c.text == "hello"


def test_matched_span_and_score() -> None:
    c = parse_control("Hello, Transcribe Send!")
    assert c.matched == "Transcribe Send!"
    assert c.score == pytest.approx(1.0)
    assert 0.9 < parse_control("transcribes stop").score < 1.0


def test_ratio_calibration() -> None:
    # documents why DEFAULT_RATIO (0.84) sits where it does
    assert phrase_ratio("transcribe", "transcribe") == 1.0
    assert phrase_ratio("transcribes", "transcribe") > 0.9
    assert phrase_ratio("trans scribe", "transcribe") > 0.9
    assert phrase_ratio("transcript", "transcribe") == pytest.approx(0.8)
    assert phrase_ratio("describe", "transcribe") < 0.7
    assert phrase_ratio("subscribe", "transcribe") < 0.7


# ---------------------------------------------------------------- ends_with_phrase (split by the segmenter)


def test_ends_with_phrase() -> None:
    assert ends_with_phrase("That's all for today. Transcribe.") == (True, "That's all for today.")
    assert ends_with_phrase("ok transcribe,") == (True, "ok")
    assert ends_with_phrase("transcribe") == (True, "")
    assert ends_with_phrase("I need to transcribe this")[0] is False
    assert ends_with_phrase("read the transcript")[0] is False


# ---------------------------------------------------------------- start of a session


@pytest.mark.parametrize("text, expected", [
    ("Transcribe, hey Sarah, just checking in.", "Hey Sarah, just checking in."),
    ("Transcribe. Meeting notes for Monday.", "Meeting notes for Monday."),
    ("transcribe hello there", "hello there"),
    ("Hey transcribe, dear team,", "Dear team,"),
    ("Okay, transcribe. Thanks for the update.", "Thanks for the update."),
    ("Um, transcribe, so the plan is", "So the plan is"),
    ("Trans scribe, quick question.", "Quick question."),
    ("Transcribe.", ""),
    ("Hello there", "Hello there"),                  # no wake word: unchanged
])
def test_strip_leading_phrase(text: str, expected: str) -> None:
    assert strip_leading_phrase(text) == expected


@pytest.mark.parametrize("text, ok", [
    ("Transcribe, hey Sarah.", True),
    ("Hey, transcribe.", True),
    ("OK so transcribe the meeting notes", True),
    ("transcribes stop", True),
    ("I need to transcribe this video.", False),     # wake word mid-sentence: false wake
    ("Read the transcript.", False),
    ("Describe the problem.", False),
    ("Subscribe to the channel.", False),
    ("", False),
])
def test_starts_with_phrase(text: str, ok: bool) -> None:
    assert starts_with_phrase(text) is ok


def test_starts_with_phrase_fragment_allowance() -> None:
    # pre-roll that started mid-speech can begin with half a word
    assert not starts_with_phrase("ing transcribe, hello")
    assert starts_with_phrase("ing transcribe, hello", max_fragments=1)
    assert strip_leading_phrase("ing transcribe, hello", max_fragments=1) == "hello"
    assert not starts_with_phrase("need to transcribe this", max_fragments=1)
