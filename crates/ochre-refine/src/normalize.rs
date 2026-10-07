//! Deterministic spoken-form -> written-form scaffold (SPEC §5.2).
//!
//! Quill 0.8B is verbatim-only by design: it removes fillers and fixes punctuation but is not
//! trusted to guess that "john at gmail dot com" is an email. These rules run after the model:
//!
//! - times:   "at three thirty pm" -> "at 3:30 PM", "five oh five am" -> "5:05 AM"
//! - numbers: "twenty five" -> "25", "three point five" -> "3.5", "ten percent" -> "10%",
//!   "fifty dollars" -> "$50"
//! - emails:  "john dot smith at gmail dot com" -> "john.smith@gmail.com"
//! - domains: "example dot com slash pricing" -> "example.com/pricing"
//! - symbols: "at sign" -> "@", "ampersand" -> "&", ...
//!
//! Deliberately conservative: a wrong rewrite of plain prose is worse than a missed conversion.
//! A single number word stays a word ("I have two cats") unless a unit makes it unambiguous
//! ("five percent"). Clock times need a cue word ("at/by/until...") or am/pm. A run of number
//! words that does not parse as one number ("twenty twenty six") is left untouched. Idempotent.

use std::sync::LazyLock;

use regex::Regex;

static SPLIT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\s*)(\S+)").unwrap());
static EDGE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)^([^\w@#$%&]*)(.*?)([^\w%]*)$").unwrap());
static WORDLIKE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[a-z0-9][a-z0-9-]*$").unwrap());
static LOCALLIKE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-z0-9][a-z0-9._+-]*$").unwrap());
static WRITTEN_DOMAIN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[a-z0-9.-]+$").unwrap());
static SYMBOLS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    [
        (r"(?i)\bat sign\b", "@"),
        (r"(?i)\bampersand\b", "&"),
        (r"(?i)\bpercent sign\b", "%"),
        (r"(?i)\bdollar sign\b", "$"),
        (r"(?i)\b(?:hash|pound) sign\b", "#"),
        (r"(?i)\bplus sign\b", "+"),
        (r"(?i)\bequals sign\b", "="),
    ]
    .into_iter()
    .map(|(p, s)| (Regex::new(p).unwrap(), s))
    .collect()
});

#[derive(Debug, Clone)]
struct Tok {
    pre: String,
    lead: String,
    core: String,
    trail: String,
    low: String,
}

impl Tok {
    fn new(pre: &str, lead: &str, core: &str, trail: &str) -> Self {
        Tok {
            pre: pre.into(),
            lead: lead.into(),
            core: core.into(),
            trail: trail.into(),
            low: core.to_lowercase(),
        }
    }
    fn words(&self) -> Vec<&str> {
        self.low.split('-').collect()
    }
}

fn tokens(text: &str) -> Vec<Tok> {
    SPLIT
        .captures_iter(text)
        .map(|m| {
            let word = &m[2];
            let e = EDGE.captures(word).unwrap();
            if e[2].is_empty() {
                Tok::new(&m[1], "", word, "")
            } else {
                Tok::new(&m[1], &e[1], &e[2], &e[3])
            }
        })
        .collect()
}

/// Tokens i..j-1 can be merged: no punctuation between them.
fn clean_span(t: &[Tok], i: usize, j: usize) -> bool {
    (i..j.saturating_sub(1)).all(|k| t[k].trail.is_empty())
        && (i + 1..j).all(|k| t[k].lead.is_empty())
}

fn replace(t: &mut Vec<Tok>, i: usize, j: usize, core: &str) {
    let tok = Tok::new(&t[i].pre, &t[i].lead, core, &t[j - 1].trail);
    t.splice(i..j, [tok]);
}

// ---------------------------------------------------------------- numbers

const UNIT_WORDS: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS_WORDS: [&str; 8] = [
    "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

fn unit(w: &str) -> Option<u64> {
    UNIT_WORDS.iter().position(|u| *u == w).map(|n| n as u64)
}
fn tens(w: &str) -> Option<u64> {
    TENS_WORDS
        .iter()
        .position(|u| *u == w)
        .map(|n| (n as u64 + 2) * 10)
}
fn scale(w: &str) -> Option<u64> {
    match w {
        "hundred" => Some(100),
        "thousand" => Some(1_000),
        "million" => Some(1_000_000),
        "billion" => Some(1_000_000_000),
        _ => None,
    }
}
fn is_num(w: &str) -> bool {
    unit(w).is_some() || tens(w).is_some() || scale(w).is_some()
}
fn is_numtok(t: &Tok) -> bool {
    !t.low.is_empty() && t.words().iter().all(|w| is_num(w))
}

#[derive(Clone, Copy, PartialEq)]
enum Prev {
    None,
    Unit,
    Teen,
    Tens,
    Hundred,
    Scale,
}

/// Value of a whole run of English number words, or None if the run is not one number.
pub fn parse_number(words: &[&str]) -> Option<u64> {
    if words.is_empty() || scale(words[0]).is_some() {
        return None;
    }
    let (mut total, mut current, mut prev) = (0u64, 0u64, Prev::None);
    for (i, w) in words.iter().enumerate() {
        if *w == "and" {
            if prev != Prev::Hundred || i == words.len() - 1 {
                return None;
            }
            continue;
        }
        if let Some(n) = unit(w) {
            let kind = if n >= 10 { Prev::Teen } else { Prev::Unit };
            if matches!(prev, Prev::Unit | Prev::Teen)
                || (prev == Prev::Tens && (kind == Prev::Teen || n == 0))
            {
                return None;
            }
            current += n;
            prev = kind;
        } else if let Some(n) = tens(w) {
            if matches!(prev, Prev::Unit | Prev::Teen | Prev::Tens) {
                return None;
            }
            current += n;
            prev = Prev::Tens;
        } else if *w == "hundred" {
            if !matches!(prev, Prev::Unit | Prev::Teen | Prev::Tens) || current == 0 {
                return None;
            }
            current *= 100;
            prev = Prev::Hundred;
        } else if let Some(s) = scale(w) {
            if matches!(prev, Prev::None | Prev::Scale) || current == 0 {
                return None;
            }
            total += current * s;
            current = 0;
            prev = Prev::Scale;
        } else {
            return None;
        }
    }
    Some(total + current)
}

/// End index of the maximal run of number words (and inner "and") starting at i.
fn number_run(t: &[Tok], i: usize) -> usize {
    let mut j = i;
    while j < t.len()
        && (is_numtok(&t[j])
            || (t[j].low == "and"
                && j > i
                && j + 1 < t.len()
                && is_numtok(&t[j + 1])
                && t[j - 1].low == "hundred"))
    {
        if j > i && !clean_span(t, j - 1, j + 1) {
            break;
        }
        j += 1;
    }
    j
}

fn run_words(t: &[Tok], i: usize, j: usize) -> Vec<String> {
    (i..j)
        .flat_map(|k| {
            t[k].words()
                .into_iter()
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn numbers(t: &mut Vec<Tok>) {
    let mut i = 0;
    while i < t.len() {
        if !is_numtok(&t[i]) || scale(&t[i].low).is_some() {
            i += 1;
            continue;
        }
        let mut j = number_run(t, i);
        let words = run_words(t, i, j);
        let refs: Vec<&str> = words.iter().map(String::as_str).collect();
        let Some(value) = parse_number(&refs) else {
            i = j;
            continue;
        };
        let mut text = value.to_string();
        // decimal: "three point five", "zero point two five"
        if j + 1 < t.len()
            && t[j].low == "point"
            && t[j - 1].trail.is_empty()
            && unit(&t[j + 1].low).is_some()
        {
            let mut k = j + 1;
            let mut digits = String::new();
            while k < t.len() && unit(&t[k].low).is_some_and(|n| n <= 9) {
                digits.push_str(&unit(&t[k].low).unwrap().to_string());
                if !t[k].trail.is_empty() {
                    k += 1;
                    break;
                }
                k += 1;
            }
            if !digits.is_empty() {
                text = format!("{text}.{digits}");
                j = k;
            }
        }
        let unit_w = if j < t.len() && t[j - 1].trail.is_empty() {
            t[j].low.clone()
        } else {
            String::new()
        };
        let multiword = words.len() > 1 || text.contains('.');
        if unit_w == "percent" {
            replace(t, i, j + 1, &format!("{text}%"));
        } else if (unit_w == "dollars" || unit_w == "dollar")
            && ((unit_w == "dollars") != (text == "1"))
        {
            replace(t, i, j + 1, &format!("${text}"));
        } else if multiword && !(j < t.len() && t[j].low == "o'clock") {
            replace(t, i, j, &text);
        }
        i += 1;
    }
}

// ---------------------------------------------------------------- times

const TIME_CUES: [&str; 9] = [
    "at", "by", "until", "till", "til", "around", "before", "after", "since",
];

/// "am"/"pm"/"a.m."/"a m" at j: (label, tokens consumed).
fn ampm(t: &[Tok], j: usize) -> (Option<&'static str>, usize) {
    if j >= t.len() {
        return (None, 0);
    }
    let low = t[j].low.trim_end_matches('.');
    match low {
        "am" | "a.m" => return (Some("AM"), 1),
        "pm" | "p.m" => return (Some("PM"), 1),
        _ => {}
    }
    if (low == "a" || low == "p")
        && j + 1 < t.len()
        && t[j + 1].low.trim_end_matches('.') == "m"
        && t[j].trail.trim_matches('.').is_empty()
    {
        return (Some(if low == "a" { "AM" } else { "PM" }), 2);
    }
    (None, 0)
}

/// Minute words at j: "oh five", "fifteen", "thirty", "forty five" -> (minutes, tokens used).
fn minutes(t: &[Tok], j: usize) -> (Option<u64>, usize) {
    if j >= t.len() {
        return (None, 0);
    }
    let w = t[j].words();
    let small_unit = |k: usize| -> Option<u64> {
        (k < t.len())
            .then(|| unit(&t[k].low))
            .flatten()
            .filter(|n| (1..=9).contains(n))
    };
    if w[0] == "oh"
        && w.len() == 1
        && let Some(n) = small_unit(j + 1)
    {
        return (Some(n), 2);
    }
    if w.len() == 1
        && let Some(n) = unit(w[0]).filter(|n| (10..=19).contains(n))
    {
        return (Some(n), 1);
    }
    if let Some(tn) = tens(w[0]).filter(|n| *n <= 50) {
        if w.len() == 2
            && let Some(u) = unit(w[1]).filter(|n| (1..=9).contains(n))
        {
            return (Some(tn + u), 1);
        }
        if w.len() == 1
            && t[j].trail.is_empty()
            && let Some(u) = small_unit(j + 1)
        {
            return (Some(tn + u), 2);
        }
        if w.len() == 1 {
            return (Some(tn), 1);
        }
    }
    (None, 0)
}

/// "a.m." becomes "AM": its period belonged to the abbreviation unless it also ends the sentence
/// (it is the last token, or the next word is capitalized).
fn drop_abbrev_period(t: &mut [Tok], k: usize) {
    if t[k].trail.starts_with('.')
        && k + 1 < t.len()
        && !t[k + 1].core.chars().next().is_some_and(char::is_uppercase)
    {
        t[k].trail.remove(0);
    }
}

fn times(t: &mut Vec<Tok>) {
    let mut i = 0;
    while i < t.len() {
        let Some(hour) = unit(&t[i].low).filter(|h| (1..=12).contains(h)) else {
            i += 1;
            continue;
        };
        let cue = i > 0 && TIME_CUES.contains(&t[i - 1].low.as_str()) && t[i - 1].trail.is_empty();
        let (mut mins, used) = if t[i].trail.is_empty() {
            minutes(t, i + 1)
        } else {
            (None, 0)
        };
        let mut end = i + 1 + used;
        if mins.is_some() && !clean_span(t, i, end) {
            mins = None;
            end = i + 1;
        }
        let (label, n) = if t[end - 1].trail.is_empty() {
            ampm(t, end)
        } else {
            (None, 0)
        };
        if label.is_some() {
            drop_abbrev_period(t, end + n - 1);
        }
        match (mins, label) {
            (Some(m), _) if cue || label.is_some() => {
                let core = format!(
                    "{hour}:{m:02}{}",
                    label.map(|l| format!(" {l}")).unwrap_or_default()
                );
                replace(t, i, end + n, &core);
            }
            (None, Some(l)) => replace(t, i, end + n, &format!("{hour} {l}")),
            _ => {}
        }
        i += 1;
    }
}

// ---------------------------------------------------------------- emails, domains, URLs

const TLDS: &[&str] = &[
    "com", "org", "net", "io", "ai", "dev", "app", "co", "edu", "gov", "uk", "ca", "de", "fr",
    "xyz", "info", "tv", "gg", "ly", "fm", "eu", "au", "jp", "nl", "ch", "se", "es", "it", "in",
    "us", "me",
];
/// Never the first label of a spoken domain.
const NOT_LABEL: &[&str] = &[
    "the", "a", "an", "this", "that", "my", "your", "our", "their", "his", "her", "its", "at",
    "and", "or", "to", "of", "in", "on", "is",
];
/// Never the local part of a spoken email ("send it to me at ...").
const NOT_LOCAL: &[&str] = &[
    "me", "us", "you", "him", "her", "them", "it", "out", "back", "home", "work", "school", "or",
    "and", "here", "there", "look", "now", "least", "all", "first", "last",
];

fn joiner(w: &str) -> Option<&'static str> {
    match w {
        "dot" => Some("."),
        "underscore" => Some("_"),
        "dash" | "hyphen" => Some("-"),
        _ => None,
    }
}

/// Spoken domain starting at i ("example dot co dot uk", or "example.com" already written).
fn domain(t: &[Tok], i: usize) -> Option<(String, usize)> {
    if i >= t.len() || !t[i].lead.is_empty() {
        return None;
    }
    let low = &t[i].low;
    if low.contains('.')
        && TLDS.contains(&low.rsplit('.').next().unwrap_or(""))
        && WRITTEN_DOMAIN.is_match(low)
    {
        return Some((low.clone(), i + 1));
    }
    if !WORDLIKE.is_match(low) || NOT_LABEL.contains(&low.as_str()) {
        return None;
    }
    let mut labels = vec![low.clone()];
    let mut j = i + 1;
    while j + 1 < t.len()
        && t[j].low == "dot"
        && WORDLIKE.is_match(&t[j + 1].low)
        && clean_span(t, j - 1, j + 2)
    {
        labels.push(t[j + 1].low.clone());
        j += 2;
    }
    while labels.len() > 1 && !TLDS.contains(&labels.last().unwrap().as_str()) {
        labels.pop();
        j -= 2;
    }
    (labels.len() >= 2).then(|| (labels.join("."), j))
}

fn emails_and_domains(t: &mut Vec<Tok>) {
    let mut i = 0;
    while i < t.len() {
        let mut scheme = String::new();
        let mut start = i;
        if (t[i].low == "https" || t[i].low == "http")
            && i + 3 < t.len()
            && t[i + 1].low == "colon"
            && t[i + 2].low == "slash"
            && t[i + 3].low == "slash"
            && clean_span(t, i, i + 4)
        {
            scheme = format!("{}://", t[i].low);
            start = i + 4;
        }
        let Some((dom, mut j)) = domain(t, start) else {
            i += 1;
            continue;
        };
        // email: <local> at <domain>
        if scheme.is_empty() && i >= 2 && t[i - 1].low == "at" && clean_span(t, i - 2, j) {
            let mut k = i - 2;
            let mut local = t[k].low.clone();
            while k >= 2
                && let Some(jn) = joiner(&t[k - 1].low)
                && LOCALLIKE.is_match(&t[k - 2].low)
                && clean_span(t, k - 2, k + 1)
            {
                local = format!("{}{jn}{local}", t[k - 2].low);
                k -= 2;
            }
            let word = &t[i - 2].low;
            if LOCALLIKE.is_match(word) && !NOT_LOCAL.contains(&word.as_str()) && !is_num(word) {
                replace(t, k, j, &format!("{local}@{dom}"));
                i = k + 1;
                continue;
            }
        }
        // URL path: "slash pricing slash team"
        let mut path = String::new();
        while j + 1 < t.len()
            && t[j].low == "slash"
            && WORDLIKE.is_match(&t[j + 1].low)
            && clean_span(t, j - 1, j + 2)
        {
            path.push('/');
            path.push_str(&t[j + 1].low);
            j += 2;
        }
        if !scheme.is_empty() || !path.is_empty() || dom != t[start].low {
            replace(t, i, j, &format!("{scheme}{dom}{path}"));
        }
        i += 1;
    }
}

// ---------------------------------------------------------------- entry point

/// Apply every rule. Idempotent: running it twice gives the same result.
pub fn normalize(text: &str) -> String {
    if text.trim().is_empty() {
        return text.to_string();
    }
    let body = text.trim_end();
    let tail = &text[body.len()..];
    let mut t = tokens(body);
    emails_and_domains(&mut t);
    times(&mut t);
    numbers(&mut t);
    let mut out: String = t
        .iter()
        .map(|k| format!("{}{}{}{}", k.pre, k.lead, k.core, k.trail))
        .collect();
    out.push_str(tail);
    for (re, sym) in SYMBOLS.iter() {
        out = re.replace_all(&out, *sym).into_owned();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (spoken, written). Ported from the Python prototype's table, plus extra must-not-change prose.
    const CASES: &[(&str, &str)] = &[
        // times (need a cue word or am/pm)
        (
            "The meeting is at three thirty tomorrow.",
            "The meeting is at 3:30 tomorrow.",
        ),
        ("Let's meet at three thirty pm.", "Let's meet at 3:30 PM."),
        (
            "Call me at five oh five a.m. please",
            "Call me at 5:05 AM please",
        ),
        ("It starts at three pm sharp.", "It starts at 3 PM sharp."),
        ("I'll be there by four fifteen.", "I'll be there by 4:15."),
        ("At nine p.m. We left.", "At 9 PM. We left."),
        ("Leave around ten forty five.", "Leave around 10:45."),
        ("See you at eleven a m.", "See you at 11 AM."),
        // numbers
        ("We sold twenty five units.", "We sold 25 units."),
        ("Twenty-five people came.", "25 people came."),
        ("thirty one days", "31 days"),
        ("That's one hundred and five dollars.", "That's $105."),
        ("Growth was ten percent.", "Growth was 10%."),
        ("It costs fifty dollars.", "It costs $50."),
        ("It costs one dollar.", "It costs $1."),
        ("Version three point five is out.", "Version 3.5 is out."),
        (
            "About two thousand three hundred users.",
            "About 2300 users.",
        ),
        // emails / domains / URLs
        (
            "Email john dot smith at gmail dot com.",
            "Email john.smith@gmail.com.",
        ),
        (
            "My email is jane_doe at outlook dot com",
            "My email is jane_doe@outlook.com",
        ),
        (
            "Reply to sam at acme dot io, thanks.",
            "Reply to sam@acme.io, thanks.",
        ),
        ("Contact us at gmail dot com.", "Contact us at gmail.com."),
        (
            "Go to example dot com slash pricing.",
            "Go to example.com/pricing.",
        ),
        (
            "Check www dot bbc dot co dot uk for news.",
            "Check www.bbc.co.uk for news.",
        ),
        (
            "Use https colon slash slash github dot com slash anthropic.",
            "Use https://github.com/anthropic.",
        ),
        // symbols
        ("Use the at sign and an ampersand.", "Use the @ and an &."),
        // must NOT change
        (
            "I have two cats and twenty minutes.",
            "I have two cats and twenty minutes.",
        ),
        (
            "Bring two twenty dollar bills.",
            "Bring two twenty dollar bills.",
        ),
        (
            "In twenty twenty six we grew.",
            "In twenty twenty six we grew.",
        ),
        (
            "I was born in nineteen ninety.",
            "I was born in nineteen ninety.",
        ),
        ("The dot com bubble burst.", "The dot com bubble burst."),
        ("Meet at noon at the office.", "Meet at noon at the office."),
        ("I'm at home.", "I'm at home."),
        ("He said one thing.", "He said one thing."),
        ("a hundred people", "a hundred people"),
        (
            "Set a timer for five minutes.",
            "Set a timer for five minutes.",
        ),
        ("The meeting is at 3:30 PM.", "The meeting is at 3:30 PM."),
        ("Send it to me at work.", "Send it to me at work."),
        (
            "We looked at one option, then two more.",
            "We looked at one option, then two more.",
        ),
        ("I am at a loss for words.", "I am at a loss for words."),
        (
            "Point taken, I'll be there at six.",
            "Point taken, I'll be there at six.",
        ),
        ("The site is down again.", "The site is down again."),
        ("", ""),
        ("   ", "   "),
    ];

    #[test]
    fn table() {
        for (spoken, written) in CASES {
            assert_eq!(normalize(spoken), *written, "spoken={spoken:?}");
            assert_eq!(normalize(written), *written, "not idempotent: {written:?}");
        }
    }

    #[test]
    fn keeps_whitespace_and_trailing_newline() {
        assert_eq!(
            normalize("  We sold twenty five units.\n"),
            "  We sold 25 units.\n"
        );
        assert_eq!(normalize("line one\nline two"), "line one\nline two");
    }

    #[test]
    fn parse_number_rules() {
        assert_eq!(parse_number(&["one", "hundred", "and", "five"]), Some(105));
        assert_eq!(parse_number(&["twenty", "twenty"]), None);
        assert_eq!(parse_number(&["hundred"]), None);
        assert_eq!(
            parse_number(&["two", "million", "five", "thousand"]),
            Some(2_005_000)
        );
    }
}
