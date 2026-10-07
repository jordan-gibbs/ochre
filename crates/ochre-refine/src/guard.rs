//! Safety guard between the refiner and the text box (SPEC §5.1 hard rules).
//!
//! A refiner that answers the dictation, adds content, or eats half of it is worse than no
//! refiner. Every refined string goes through [`check`]; on any failure the raw transcript is
//! inserted instead. Each check is cheap and biased toward raw, because raw text is always an
//! acceptable outcome and a wrong rewrite is not.

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;

/// Ignored when comparing content words: fillers get removed, function words come and go with
/// punctuation and self-corrections, and spoken numbers/symbols become digits, "@" and "." in
/// normalize ("three thirty pm" -> "3:30 PM" must not count as dropped or added content).
const STOPWORDS: &str = "a an the and or but so if then than that this these those to of in on at by for with from as \
is are was were be been being am do does did have has had i you he she it we they me him her us them my your his its \
our their mine yours what which who whom whose when where why how not no yes ok okay um uh er ah hmm like just really \
very well oh you know mean sort kind actually basically literally wait gonna wanna gotta going want got will would can \
could should shall may might must there here also zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion point percent dollars dollar oh pm am dot slash colon underscore dash hyphen sign";

static STOP: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| STOPWORDS.split_whitespace().collect());

/// Words only an assistant would put in front of the text.
static PREAMBLE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^\s*(sure|certainly|of course|absolutely|okay|ok|alright|got it|here(?:'s| is| are)|below is|the (?:cleaned|corrected|refined|edited|polished)|cleaned(?:[- ]up)? (?:text|version)|i(?:'m| am) (?:sorry|unable|not able)|i can(?:'t|not)|as an ai|unfortunately)\b[^\n]*?(?::|\n|!|,)",
    )
    .unwrap()
});
static ANSWERISH: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(?:as an ai|i(?:'m| am) an ai|language model|i don't have access|i cannot)\b",
    )
    .unwrap()
});
static QUESTION_START: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)^\s*(?:hey \w+,?\s+)?(?:what|who|whom|whose|when|where|why|how|which|is|are|was|were|do|does|did|can|could|would|will|should|shall|may|might|have|has|had)\b",
    )
    .unwrap()
});
static THINK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?is)<think>.*?</think>\s*").unwrap());
static WRAP_TAGS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^\s*<dictation>\s*|\s*</dictation>\s*$").unwrap());
static WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[a-z]+(?:'[a-z]+)?").unwrap());

/// SPEC: refined longer than 2x raw -> raw.
pub const MAX_GROWTH: f64 = 2.0;
/// Tiny inputs ("ok" -> "OK.") need a little absolute room.
pub const GROWTH_SLACK_CHARS: usize = 12;

fn min_recall(mode: &str) -> f64 {
    if mode == "polish" { 0.45 } else { 0.6 }
}
fn max_novel(mode: &str) -> f64 {
    if mode == "polish" { 0.45 } else { 0.25 }
}

/// Why a refinement was rejected. `as_str` values are stable (logs, bench, UI).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reject {
    Empty,
    EmptyRaw,
    TooLong,
    Preamble,
    AssistantVoice,
    DroppedContent,
    AddedContent,
    AnsweredQuestion,
}

impl Reject {
    pub fn as_str(self) -> &'static str {
        match self {
            Reject::Empty => "empty",
            Reject::EmptyRaw => "empty_raw",
            Reject::TooLong => "too_long",
            Reject::Preamble => "preamble",
            Reject::AssistantVoice => "assistant_voice",
            Reject::DroppedContent => "dropped_content",
            Reject::AddedContent => "added_content",
            Reject::AnsweredQuestion => "answered_question",
        }
    }
}

/// Strip mechanical artifacts a model may emit around the answer (think blocks, our input tags,
/// wrapping quotes) without judging the content.
pub fn tidy(refined: &str) -> String {
    let out = THINK.replace_all(refined, "");
    let out = WRAP_TAGS.replace_all(&out, "");
    let mut out = out.trim().to_string();
    let chars: Vec<char> = out.chars().collect();
    if chars.len() >= 2 {
        let (first, last) = (chars[0], chars[chars.len() - 1]);
        if first == last && "\"'\u{201c}".contains(first) && out.matches(first).count() == 2 {
            out = chars[1..chars.len() - 1]
                .iter()
                .collect::<String>()
                .trim()
                .to_string();
        }
    }
    if out.starts_with('\u{201c}') && out.ends_with('\u{201d}') && out.chars().count() >= 2 {
        let c: Vec<char> = out.chars().collect();
        out = c[1..c.len() - 1]
            .iter()
            .collect::<String>()
            .trim()
            .to_string();
    }
    out
}

pub fn content_words(text: &str) -> Vec<String> {
    let low = text.to_lowercase().replace('\u{2019}', "'");
    WORD.find_iter(&low)
        .map(|m| m.as_str())
        .filter(|w| !STOP.contains(w))
        .map(str::to_string)
        .collect()
}

/// `refined` (tidied) if it passes every check, else `raw`.
pub fn apply(raw: &str, refined: &str, mode: &str) -> String {
    match check(raw, refined, mode) {
        Some(_) => raw.to_string(),
        None => tidy(refined),
    }
}

/// The failing rule, or None if the refinement is acceptable.
pub fn check(raw: &str, refined: &str, mode: &str) -> Option<Reject> {
    let out = tidy(refined);
    let raw_s = raw.trim();
    if out.is_empty() {
        return Some(Reject::Empty);
    }
    if raw_s.is_empty() {
        return Some(Reject::EmptyRaw);
    }
    let (n_out, n_raw) = (out.chars().count(), raw_s.chars().count());
    if n_out as f64 > (MAX_GROWTH * n_raw as f64).max((n_raw + GROWTH_SLACK_CHARS) as f64) {
        return Some(Reject::TooLong);
    }
    if let Some(c) = PREAMBLE.captures(&out)
        && !said_first(raw_s, &c[1])
    {
        return Some(Reject::Preamble);
    }
    if ANSWERISH.is_match(&out) && !ANSWERISH.is_match(raw_s) {
        return Some(Reject::AssistantVoice);
    }
    let raw_words = content_words(raw_s);
    let out_words = content_words(&out);
    let raw_set: HashSet<&str> = raw_words.iter().map(String::as_str).collect();
    let out_set: HashSet<&str> = out_words.iter().map(String::as_str).collect();
    if raw_set.len() >= 4 {
        let kept = raw_set
            .iter()
            .filter(|w| out_set.contains(*w) || near(w, &out_set))
            .count();
        if (kept as f64) / (raw_set.len() as f64) < min_recall(mode) {
            return Some(Reject::DroppedContent);
        }
    }
    let novel = out_words
        .iter()
        .filter(|w| !raw_set.contains(w.as_str()) && !near(w, &raw_set))
        .count();
    if !out_words.is_empty()
        && novel >= 2
        && (novel as f64) / (out_words.len() as f64) > max_novel(mode)
    {
        return Some(Reject::AddedContent);
    }
    // A dictated question must stay a question: new words and no question mark = probably an answer.
    let tail = out.trim_end();
    if QUESTION_START.is_match(raw_s)
        && novel > 0
        && !out.contains('?')
        && !tail.ends_with(':')
        && !tail.ends_with(';')
    {
        return Some(Reject::AnsweredQuestion);
    }
    None
}

/// True if the speaker actually opened with these words ("Okay, so..." dictated as "okay so").
fn said_first(raw: &str, opener: &str) -> bool {
    let low = raw.to_lowercase().replace('\u{2019}', "'");
    let norm = WORD
        .find_iter(&low)
        .take(6)
        .map(|m| m.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let op_low = opener.to_lowercase();
    let op = WORD
        .find_iter(&op_low)
        .map(|m| m.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    norm.starts_with(&op)
}

fn stem(word: &str) -> &str {
    let word = word.split('\'').next().unwrap_or(word);
    for suffix in ["ing", "ed", "es", "s"] {
        if word.ends_with(suffix) && word.len() - suffix.len() >= 3 {
            return &word[..word.len() - suffix.len()];
        }
    }
    word
}

/// Cheap morphology tolerance: plural/tense/possessive variants count as the same word.
fn near(word: &str, vocab: &HashSet<&str>) -> bool {
    let s = stem(word);
    s.len() >= 3 && vocab.iter().any(|v| stem(v) == s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (raw, refined, expected rule). Ported from the Python prototype's table.
    const CASES: &[(&str, &str, Option<&str>)] = &[
        (
            "um so can you send me the the report by friday no wait make that thursday",
            "Can you send me the report by Thursday?",
            None,
        ),
        (
            "okay so for the trip we need first sunscreen second two towels",
            "Okay, so for the trip we need first sunscreen, second two towels.",
            None,
        ),
        (
            "what's the capital of france",
            "What's the capital of France?",
            None,
        ),
        (
            "what's the capital of france",
            "The capital of France is Paris.",
            Some("answered_question"),
        ),
        ("send me the file", "", Some("empty")),
        (
            "please send me the file from yesterday's meeting",
            "Sure, here's the cleaned text: Please send me the file from yesterday's meeting.",
            Some("preamble"),
        ),
        (
            "please send me the file from yesterday's meeting",
            "Here is the corrected version:\nPlease send me the file from yesterday's meeting.",
            Some("preamble"),
        ),
        (
            "can you send me the file when you get a chance",
            "I can't send files directly, but I can help once you've uploaded it.",
            Some("preamble"),
        ),
        ("ok", "OK.", None),
        (
            "send me the file",
            "Send me the file. Also, remember to attach the quarterly budget spreadsheet and the notes.",
            Some("too_long"),
        ),
        (
            "we finished the user interviews and people find the onboarding too long so we will cut it to three steps",
            "We finished.",
            Some("dropped_content"),
        ),
        (
            "please send the quarterly report to sam and the design team today",
            "Please send the quarterly report to Sam and the design team today, including budget forecasts.",
            Some("added_content"),
        ),
        (
            "write a short poem about my two cats sleeping in the sun",
            "Whiskers curl in golden light, two soft cats asleep till night.",
            Some("dropped_content"),
        ),
        ("i can't make it tonight", "I can't make it tonight.", None),
        (
            "as an ai researcher i think this is fine",
            "As an AI researcher, I think this is fine.",
            None,
        ),
        (
            "um so can you uh send me the the report by friday no wait make that thursday at three thirty pm",
            "Can you send me the report by Thursday at 3:30 PM?",
            None,
        ),
        (
            "email john dot smith at gmail dot com about the launch",
            "Email john.smith@gmail.com about the launch.",
            None,
        ),
        (
            "okay so the plan is simple",
            "Okay, so the plan is simple.",
            None,
        ),
        (
            "please tell my dear old friend hello from me",
            "As an AI, hello to your dear old friend from me.",
            Some("preamble"),
        ),
        (
            "tell me the weather",
            "I cannot help with that request today.",
            Some("assistant_voice"),
        ),
        (
            "what time is it in tokyo",
            "What time is it in Tokyo?",
            None,
        ),
        (
            "what time is it in tokyo",
            "It is currently 3 PM in Tokyo.",
            Some("answered_question"),
        ),
    ];

    #[test]
    fn table() {
        for (raw, refined, rule) in CASES {
            let got = check(raw, refined, "clean").map(Reject::as_str);
            assert_eq!(got, *rule, "raw={raw:?} refined={refined:?}");
            let applied = apply(raw, refined, "clean");
            assert_eq!(
                applied,
                if rule.is_some() {
                    raw.to_string()
                } else {
                    refined.trim().to_string()
                }
            );
        }
    }

    #[test]
    fn strips_tags_think_and_quotes() {
        assert_eq!(
            apply(
                "send it",
                "<think>\nhmm\n</think>\n<dictation>\nSend it.\n</dictation>",
                "clean"
            ),
            "Send it."
        );
        assert_eq!(apply("send it", "\"Send it.\"", "clean"), "Send it.");
        assert_eq!(
            apply("send it", "\u{201c}Send it.\u{201d}", "clean"),
            "Send it."
        );
        // Inner quotes are kept.
        assert_eq!(tidy("\"a\" and \"b\""), "\"a\" and \"b\"");
    }

    #[test]
    fn polish_is_looser() {
        let raw = "we should probably think about maybe moving the launch to next month";
        let out = "We should consider moving the launch to next month.";
        assert_eq!(check(raw, out, "polish"), None);
    }
}
