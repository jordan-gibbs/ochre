//! Table-driven suite: every case from `tests/test_text_commands.py` and `docs/wakeword.md` §8,
//! plus extra ASR variants and mid-sentence traps.

use super::*;

const P: &str = "transcribe";

fn plain(text: &str) -> Parsed {
    parse_control(text, P, &ParseOptions::default())
}

fn armed(text: &str) -> Parsed {
    parse_control(
        text,
        P,
        &ParseOptions {
            armed: true,
            ..Default::default()
        },
    )
}

use Control::*;

#[test]
fn trailing_commands_trigger() {
    let cases: &[(&str, Control, &str)] = &[
        ("transcribe stop", Finish, ""),
        ("Transcribe stop.", Finish, ""),
        ("transcribe, stop.", Finish, ""),
        ("Transcribe. Stop.", Finish, ""),
        ("transcribes stop", Finish, ""),
        ("Transcribe stopped.", Finish, ""),
        ("trans scribe stop", Finish, ""),
        ("Trans-scribe, stop!", Finish, ""),
        ("transcribed done", Finish, ""),
        ("Transcribe done.", Finish, ""),
        ("transcribe done", Finish, ""),
        ("TRANSCRIBE STOP", Finish, ""),
        ("Transcribe Send!", Send, ""),
        ("transcribe sent", Send, ""),
        ("Transcribe, send it.", Send, ""),
        ("Transcribe cancel.", Cancel, ""),
        ("transcribe cancelled", Cancel, ""),
        ("Transcribe, scratch that.", Scratch, ""),
        ("transcribe scratch", Scratch, ""),
        ("Hello world, transcribe send.", Send, "Hello world"),
        (
            "See you at five. Transcribe stop.",
            Finish,
            "See you at five.",
        ),
        ("Thanks so much! Transcribe, send!", Send, "Thanks so much!"),
        (
            "Let me think about it — transcribe stop",
            Finish,
            "Let me think about it",
        ),
        (
            "I'll call you later transcribe done",
            Finish,
            "I'll call you later",
        ),
        ("Draft one... transcribe cancel", Cancel, "Draft one..."),
        (
            "Ship it on Friday, transcribe scratch that.",
            Scratch,
            "Ship it on Friday",
        ),
        ("transcribe stop   ", Finish, ""),
        ("transcribe stop…", Finish, ""),
        ("Transcribe stop?", Finish, ""),
        // extra ASR variants
        ("Transcribe; stop", Finish, ""),
        ("Transcribe - send.", Send, ""),
        ("Transcribe: cancel!", Cancel, ""),
        ("transcribe stops", Finish, ""),
        ("transcribe sends", Send, ""),
        ("Transcribe canceled.", Cancel, ""),
        ("transcribe cancels", Cancel, ""),
        ("Transcribe, scratched that.", Scratch, ""),
        ("Transcribe scratch this.", Scratch, ""),
        ("tran scribe stop", Finish, ""),
        ("Trans Scribe Send", Send, ""),
        ("transcribe dun", Finish, ""),
        (
            "Can you transcribe, send it to Bob? Transcribe send.",
            Send,
            "Can you transcribe, send it to Bob?",
        ),
        ("can you transcribe, send it", Send, "can you"),
        ("OK that's it.\nTranscribe stop.", Finish, "OK that's it."),
        ("«Bonjour», transcribe stop", Finish, "«Bonjour»"),
        ("Café at 5, transcribe send", Send, "Café at 5"),
        ("Numbers 1, 2, 3 transcribe stop", Finish, "Numbers 1, 2, 3"),
    ];
    for (text, want, kept) in cases {
        let p = plain(text);
        assert_eq!(p.control, Some(*want), "{text:?} -> {p:?}");
        assert_eq!(p.text, *kept, "{text:?} -> {p:?}");
        assert_eq!(
            parse_trailing(text, P),
            (kept.to_string(), Some(*want)),
            "{text:?}"
        );
    }
}

#[test]
fn never_mid_sentence_or_near_words() {
    let cases = [
        "I need to transcribe stop-motion footage",
        "I need to transcribe stop-motion",
        "Can you transcribe stop signs in the video",
        "transcribe send the file to Bob",
        "transcribe stop the recording please",
        "We should transcribe, then send it to legal for review.",
        "can you transcribe, send it to Bob",
        "Can you transcribe, send it to Bob, and then call me?",
        "Please describe stop.",
        "Subscribe, stop.",
        "prescribe done",
        "Read the transcript. Send.",
        "Don't stop.",
        "Stop.",
        "send",
        "Scratch that.",
        "I scratched that car door",
        "The tribe sent a message",
        "",
        "   ",
        "...",
        "stop transcribe",
        "transcribe",
        "Transcribe.",
        "transcribe stopwatch",
        "transcribe-stop",
        "transcribe stop_watch",
        "transcribe sender",
        "transcriber stop", // "transcriber" ratio = 0.95 but... see below
        "transcription done",
        "inscribe send",
        "the scribe sent",
        "transcribe send-off",
        "We'll transcribe, stop by, and send.",
    ];
    for text in cases {
        let p = plain(text);
        if text == "transcriber stop" {
            // One extra letter matches like "transcribes": accepted by design (ASR inflection).
            assert_eq!(p.control, Some(Finish));
            continue;
        }
        assert_eq!(p.control, None, "{text:?} -> {p:?}");
        assert_eq!(p.text, text);
        assert_eq!(parse_trailing(text, P), (text.to_string(), None));
    }
}

#[test]
fn transcription_ratio_is_below_default() {
    assert!(phrase_ratio("transcription", P) < DEFAULT_RATIO);
}

#[test]
fn armed_accepts_bare_command_and_rough_asr() {
    assert_eq!(armed("Stop.").control, Some(Finish));
    assert_eq!(armed("Send!").control, Some(Send));
    assert_eq!(armed("Scratch that.").control, Some(Scratch));
    assert_eq!(armed("Cancel").control, Some(Cancel));
    assert_eq!(armed("done.").control, Some(Finish));
    let c = armed("All done here, transcript stop.");
    assert_eq!(
        (c.control, c.text.as_str()),
        (Some(Finish), "All done here")
    );
    for text in [
        "Don't stop.",
        "we must stop now",
        "Please describe stop.",
        "stop the music",
        "I sent it",
    ] {
        assert_eq!(armed(text).control, None, "{text:?}");
    }
}

#[test]
fn custom_phrase_and_aliases() {
    let o = ParseOptions::default();
    assert_eq!(
        parse_control("computer stop", "computer", &o).control,
        Some(Finish)
    );
    assert_eq!(
        parse_control("Hey computer, send.", "hey computer", &o).control,
        Some(Send)
    );
    assert_eq!(
        parse_control("Hey, computer send", "hey computer", &o).control,
        Some(Send)
    );
    assert_eq!(
        parse_control("transcribe stop", "computer", &o).control,
        None
    );
    assert_eq!(
        parse_control("Juniper, send it.", "juniper", &o).control,
        Some(Send)
    );
    let aliases = vec!["trance cry".to_string()];
    let c = parse_control(
        "hello trance cry stop",
        P,
        &ParseOptions {
            aliases: &aliases,
            ..Default::default()
        },
    );
    assert_eq!((c.control, c.text.as_str()), (Some(Finish), "hello"));
}

#[test]
fn matched_span_and_score() {
    let c = plain("Hello, Transcribe Send!");
    assert_eq!(c.matched, "Transcribe Send!");
    assert!((c.score - 1.0).abs() < 1e-12);
    let s = plain("transcribes stop").score;
    assert!(0.9 < s && s < 1.0);
}

#[test]
fn ratio_calibration() {
    assert_eq!(phrase_ratio("transcribe", P), 1.0);
    assert!(phrase_ratio("transcribes", P) > 0.9);
    assert!(phrase_ratio("trans scribe", P) > 0.9);
    assert!((phrase_ratio("transcript", P) - 0.8).abs() < 1e-12);
    assert!(phrase_ratio("describe", P) < 0.7);
    assert!(phrase_ratio("subscribe", P) < 0.7);
    assert_eq!(phrase_ratio("", P), 0.0);
}

#[test]
fn difflib_parity() {
    // values from Python difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    let cases: &[(&str, &str, f64)] = &[
        ("transcribe", "transcribe", 1.0),
        ("transcribes", "transcribe", 20.0 / 21.0),
        ("transcript", "transcribe", 0.8),
        ("describe", "transcribe", 12.0 / 18.0),
        ("subscribe", "transcribe", 12.0 / 19.0),
        ("abcabc", "cbacba", 0.5),
        ("tribe", "transcribe", 10.0 / 15.0),
    ];
    for (a, b, want) in cases {
        assert!(
            (phrase_ratio(a, b) - want).abs() < 1e-12,
            "{a} {b} {}",
            phrase_ratio(a, b)
        );
    }
}

#[test]
fn ends_with_phrase_cases() {
    let n: &[String] = &[];
    assert_eq!(
        ends_with_phrase("That's all for today. Transcribe.", P, n).as_deref(),
        Some("That's all for today.")
    );
    assert_eq!(
        ends_with_phrase("ok transcribe,", P, n).as_deref(),
        Some("ok")
    );
    assert_eq!(ends_with_phrase("transcribe", P, n).as_deref(), Some(""));
    assert_eq!(ends_with_phrase("I need to transcribe this", P, n), None);
    assert_eq!(ends_with_phrase("read the transcript", P, n), None);
    assert_eq!(ends_with_phrase("", P, n), None);
}

#[test]
fn strip_leading_phrase_cases() {
    let n: &[String] = &[];
    let cases = [
        (
            "Transcribe, hey Sarah, just checking in.",
            "Hey Sarah, just checking in.",
        ),
        (
            "Transcribe. Meeting notes for Monday.",
            "Meeting notes for Monday.",
        ),
        ("transcribe hello there", "hello there"),
        ("Hey transcribe, dear team,", "Dear team,"),
        (
            "Okay, transcribe. Thanks for the update.",
            "Thanks for the update.",
        ),
        ("Um, transcribe, so the plan is", "So the plan is"),
        ("Trans scribe, quick question.", "Quick question."),
        ("Transcribe.", ""),
        ("Hello there", "Hello there"),
        ("Transcribe — ölçü birimi", "Ölçü birimi"),
    ];
    for (text, want) in cases {
        assert_eq!(strip_leading_phrase(text, P, n, 0), want, "{text:?}");
    }
}

#[test]
fn starts_with_phrase_cases() {
    let n: &[String] = &[];
    let cases = [
        ("Transcribe, hey Sarah.", true),
        ("Hey, transcribe.", true),
        ("OK so transcribe the meeting notes", true),
        ("transcribes stop", true),
        ("I need to transcribe this video.", false),
        ("Read the transcript.", false),
        ("Describe the problem.", false),
        ("Subscribe to the channel.", false),
        ("", false),
    ];
    for (text, ok) in cases {
        assert_eq!(starts_with_phrase(text, P, n, 0), ok, "{text:?}");
    }
}

#[test]
fn starts_with_phrase_fragment_allowance() {
    let n: &[String] = &[];
    assert!(!starts_with_phrase("ing transcribe, hello", P, n, 0));
    assert!(starts_with_phrase("ing transcribe, hello", P, n, 1));
    assert_eq!(
        strip_leading_phrase("ing transcribe, hello", P, n, 1),
        "hello"
    );
    assert!(!starts_with_phrase("need to transcribe this", P, n, 1));
}
