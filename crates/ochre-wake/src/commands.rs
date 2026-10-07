//! Hands-free control phrases in transcripts: `"<phrase> stop|done|send|cancel|scratch that"`.
//! Port of `src/openwhisprflow/text/commands.py` (`docs/wakeword.md` §8).
//!
//! The wake model hears the word; the transcript decides what it meant. A control phrase only
//! counts when it is the **tail** of a phrase transcript (nothing but punctuation after it), so
//! "I need to transcribe stop-motion footage" or "transcribe send the file to Bob" never end a
//! session. Matching is fuzzy because ASR spells the phrase many ways ("Transcribe, stop.",
//! "transcribes stop", "trans scribe stop"): up to 3 tokens before the command word are joined and
//! compared with a difflib-compatible similarity ratio. Near words ("describe", "subscribe",
//! "transcript") stay below [`DEFAULT_RATIO`]. When the wake model fired moments ago (`armed`), the
//! ratio drops to [`ARMED_RATIO`] and a bare command ("Stop.") counts on its own.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::LazyLock;

use regex::Regex;

/// "transcribes" / "trans scribe" ~0.95 pass; "transcript" 0.80, "describe" 0.67 fail.
pub const DEFAULT_RATIO: f64 = 0.84;
/// The wake model fired: accept rougher ASR ("transcript stop").
pub const ARMED_RATIO: f64 = 0.70;
/// Split spellings: "trans scribe", "tran scribe".
pub const MAX_PHRASE_TOKENS: usize = 3;

/// Words allowed before the wake word at the start of a session ("Hey transcribe, ...").
pub const LEAD_FILLERS: &[&str] = &[
    "hey", "hi", "hello", "okay", "ok", "so", "um", "uh", "umm", "uhh", "oh", "alright", "right",
    "well", "and", "a", "the", "yo",
];

/// A recognised trailing command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Control {
    /// `"<phrase> stop"` / `"<phrase> done"`: insert.
    Finish,
    /// `"<phrase> send"`: insert, then press Enter.
    Send,
    /// `"<phrase> cancel"`: discard the session.
    Cancel,
    /// `"<phrase> scratch that"`: drop the last phrase.
    Scratch,
}

const COMMANDS: &[(&[&str], Control)] = &[
    (&["stop"], Control::Finish),
    (&["stops"], Control::Finish),
    (&["stopped"], Control::Finish),
    (&["done"], Control::Finish),
    (&["dun"], Control::Finish),
    (&["send"], Control::Send),
    (&["sends"], Control::Send),
    (&["sent"], Control::Send),
    (&["send", "it"], Control::Send),
    (&["cancel"], Control::Cancel),
    (&["canceled"], Control::Cancel),
    (&["cancelled"], Control::Cancel),
    (&["cancels"], Control::Cancel),
    (&["scratch", "that"], Control::Scratch),
    (&["scratched", "that"], Control::Scratch),
    (&["scratch", "this"], Control::Scratch),
    (&["scratch"], Control::Scratch),
];
const MAX_CMD: usize = 2;

static TOKEN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[\w&&[^_]]+(?:['’\-][\w&&[^_]]+)*").unwrap());
static TRAIL_JUNK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\s,;:\-–—…]+$").unwrap());

/// Options for [`parse_control`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ParseOptions<'a> {
    /// Extra spellings that count as the phrase (a user's ASR habit).
    pub aliases: &'a [String],
    /// The wake model fired recently: lower ratio, bare command accepted.
    pub armed: bool,
    /// Override the ratio threshold.
    pub min_ratio: Option<f64>,
}

/// Result of [`parse_control`].
#[derive(Debug, Clone, PartialEq)]
pub struct Parsed {
    pub control: Option<Control>,
    /// The transcript with the control phrase and dangling separators removed (unchanged if none).
    pub text: String,
    /// The original span that was recognised, e.g. "Transcribe, stop.".
    pub matched: String,
    /// Phrase similarity (1.0 exact; 0.0 for a bare command when armed).
    pub score: f64,
}

/// The required free function: `(text without the trailing control phrase, control)`, not armed.
/// With no control phrase the text comes back unchanged.
pub fn parse_trailing(text: &str, phrase: &str) -> (String, Option<Control>) {
    let p = parse_control(text, phrase, &ParseOptions::default());
    (p.text, p.control)
}

#[derive(Debug, Clone)]
struct Tok {
    norm: Vec<char>,
    span: Range<usize>,
}

fn tokens(text: &str) -> Vec<Tok> {
    TOKEN
        .find_iter(text)
        .map(|m| Tok {
            norm: m
                .as_str()
                .to_lowercase()
                .chars()
                .filter(|c| !matches!(c, '\'' | '’' | '-'))
                .collect(),
            span: m.range(),
        })
        .collect()
}

fn squash(s: &str) -> Vec<char> {
    s.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

fn n_words(phrase: &str) -> usize {
    phrase.split_whitespace().count()
}

// ------------------------------------------------------------------ difflib.SequenceMatcher port

/// Total size of `difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()`.
fn matching_chars(a: &[char], b: &[char]) -> usize {
    let mut b2j: HashMap<char, Vec<usize>> = HashMap::new();
    for (j, c) in b.iter().enumerate() {
        b2j.entry(*c).or_default().push(j);
    }
    let mut total = 0;
    let mut queue = vec![(0, a.len(), 0, b.len())];
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, k) = longest_match(a, b, &b2j, alo, ahi, blo, bhi);
        if k > 0 {
            total += k;
            if alo < i && blo < j {
                queue.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                queue.push((i + k, ahi, j + k, bhi));
            }
        }
    }
    total
}

/// `find_longest_match` without junk: the longest block, earliest in `a`, then earliest in `b`.
#[allow(clippy::too_many_arguments)]
fn longest_match(
    a: &[char],
    b: &[char],
    b2j: &HashMap<char, Vec<usize>>,
    alo: usize,
    ahi: usize,
    blo: usize,
    bhi: usize,
) -> (usize, usize, usize) {
    let (mut besti, mut bestj, mut best) = (alo, blo, 0);
    let mut j2len: HashMap<usize, usize> = HashMap::new();
    for (i, ch) in a.iter().enumerate().take(ahi).skip(alo) {
        let mut next: HashMap<usize, usize> = HashMap::new();
        if let Some(js) = b2j.get(ch) {
            for &j in js {
                if j < blo {
                    continue;
                }
                if j >= bhi {
                    break;
                }
                let k = if j > 0 {
                    j2len.get(&(j - 1)).copied().unwrap_or(0)
                } else {
                    0
                } + 1;
                next.insert(j, k);
                if k > best {
                    besti = i + 1 - k;
                    bestj = j + 1 - k;
                    best = k;
                }
            }
        }
        j2len = next;
    }
    while besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1] {
        besti -= 1;
        bestj -= 1;
        best += 1;
    }
    while besti + best < ahi && bestj + best < bhi && a[besti + best] == b[bestj + best] {
        best += 1;
    }
    (besti, bestj, best)
}

fn ratio_chars(a: &[char], b: &[char]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    2.0 * matching_chars(a, b) as f64 / (a.len() + b.len()) as f64
}

/// Similarity of a candidate (spaces and punctuation ignored) to the phrase: 1.0 exact.
pub fn phrase_ratio(candidate: &str, phrase: &str) -> f64 {
    ratio_chars(&squash(candidate), &squash(phrase))
}

// ------------------------------------------------------------------ matching

/// Is `token` plausibly a piece of the phrase ("trans", "scribe")? Stops junk being glued on.
fn piece_of(token: &[char], phrase: &[char]) -> bool {
    !token.is_empty()
        && !phrase.is_empty()
        && matching_chars(token, phrase) as f64 / token.len() as f64 >= 0.8
}

fn match_tokens(toks: &[Tok], i: usize, j: usize, phrase: &str) -> f64 {
    let p = squash(phrase);
    if j - i > 1 && n_words(phrase) == 1 && !toks[i..j].iter().all(|t| piece_of(&t.norm, &p)) {
        return 0.0;
    }
    let joined: String = toks[i..j].iter().flat_map(|t| t.norm.iter()).collect();
    ratio_chars(&squash(&joined), &p)
}

/// Tokens `toks[i..end]` that best match a phrase (1..=MAX tokens). Ties keep fewer tokens.
fn best_phrase_end_match(
    toks: &[Tok],
    end: usize,
    phrases: &[&str],
    min_ratio: f64,
) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    for phrase in phrases {
        let max_k = (MAX_PHRASE_TOKENS + n_words(phrase) - 1).min(end);
        for k in 1..=max_k {
            let i = end - k;
            let r = match_tokens(toks, i, end, phrase);
            if r >= min_ratio && best.is_none_or(|(_, b)| r > b + 1e-9) {
                best = Some((i, r));
            }
        }
    }
    best
}

fn clean_prefix(text: &str) -> String {
    TRAIL_JUNK.replace(text, "").trim_end().to_string()
}

fn phrase_list<'a>(phrase: &'a str, aliases: &'a [String]) -> Vec<&'a str> {
    std::iter::once(phrase)
        .chain(aliases.iter().map(String::as_str))
        .filter(|p| !p.trim().is_empty())
        .collect()
}

fn lookup(norms: &[&[char]]) -> Option<Control> {
    COMMANDS.iter().find_map(|(words, c)| {
        (words.len() == norms.len()
            && words
                .iter()
                .zip(norms)
                .all(|(w, n)| w.chars().eq(n.iter().copied())))
        .then_some(*c)
    })
}

/// Find a trailing control phrase in one phrase transcript.
pub fn parse_control(text: &str, phrase: &str, opts: &ParseOptions<'_>) -> Parsed {
    let none = || Parsed {
        control: None,
        text: text.to_string(),
        matched: String::new(),
        score: 0.0,
    };
    let toks = tokens(text);
    if toks.is_empty() {
        return none();
    }
    let phrases = phrase_list(phrase, opts.aliases);
    let ratio = opts.min_ratio.unwrap_or(if opts.armed {
        ARMED_RATIO
    } else {
        DEFAULT_RATIO
    });
    // Longest command first: "... scratch that" is ("scratch", "that"), "... send it" ("send", "it").
    for n_cmd in (1..=MAX_CMD.min(toks.len())).rev() {
        let tail: Vec<&[char]> = toks[toks.len() - n_cmd..]
            .iter()
            .map(|t| t.norm.as_slice())
            .collect();
        let Some(control) = lookup(&tail) else {
            continue;
        };
        let cmd_start = toks.len() - n_cmd;
        if let Some((i, r)) = best_phrase_end_match(&toks, cmd_start, &phrases, ratio) {
            let start = toks[i].span.start;
            return Parsed {
                control: Some(control),
                text: clean_prefix(&text[..start]),
                matched: text[start..].trim().to_string(),
                score: r,
            };
        }
        if opts.armed && cmd_start == 0 {
            // The wake model heard the phrase but the ASR dropped it: "Stop." on its own.
            return Parsed {
                control: Some(control),
                text: String::new(),
                matched: text.trim().to_string(),
                score: 0.0,
            };
        }
    }
    none()
}

/// Does the transcript END with the bare phrase ("... that's it. Transcribe.")? Returns the text
/// before it (cleaned). The segmenter may cut between "transcribe" and "stop".
pub fn ends_with_phrase(text: &str, phrase: &str, aliases: &[String]) -> Option<String> {
    let toks = tokens(text);
    if toks.is_empty() {
        return None;
    }
    let (i, _) = best_phrase_end_match(
        &toks,
        toks.len(),
        &phrase_list(phrase, aliases),
        DEFAULT_RATIO,
    )?;
    Some(clean_prefix(&text[..toks[i].span.start]))
}

/// False-wake check: the first words are the phrase, after optional fillers ("Hey, transcribe").
/// `max_fragments`: arbitrary tokens allowed before it (a word cut in half by the pre-roll).
pub fn starts_with_phrase(
    text: &str,
    phrase: &str,
    aliases: &[String],
    max_fragments: usize,
) -> bool {
    leading_span(
        text,
        &phrase_list(phrase, aliases),
        DEFAULT_RATIO,
        max_fragments,
    )
    .is_some()
}

/// Remove the wake word (and fillers before it) from the start: "Transcribe, hey Sarah." -> "Hey Sarah.".
pub fn strip_leading_phrase(
    text: &str,
    phrase: &str,
    aliases: &[String],
    max_fragments: usize,
) -> String {
    let Some(end) = leading_span(
        text,
        &phrase_list(phrase, aliases),
        DEFAULT_RATIO,
        max_fragments,
    ) else {
        return text.to_string();
    };
    let rest = text[end..].trim_start_matches(|c: char| {
        matches!(
            c,
            ' ' | '\t' | '\r' | '\n' | ',' | '.' | ';' | ':' | '!' | '?' | '-' | '–' | '—' | '…'
        )
    });
    let starts_upper = text.chars().next().is_some_and(char::is_uppercase);
    let mut chars = rest.chars();
    match chars.next() {
        Some(c) if c.is_lowercase() && starts_upper => c.to_uppercase().chain(chars).collect(),
        _ => rest.to_string(),
    }
}

fn leading_span(
    text: &str,
    phrases: &[&str],
    min_ratio: f64,
    max_fragments: usize,
) -> Option<usize> {
    let toks = tokens(text);
    let (mut lead, mut frags) = (0, 0);
    while lead < toks.len() {
        let n = toks.len() - lead;
        let mut best: Option<(usize, f64)> = None;
        for p in phrases {
            for k in 1..=(MAX_PHRASE_TOKENS + n_words(p) - 1).min(n) {
                let r = match_tokens(&toks, lead, lead + k, p);
                if r >= min_ratio && best.is_none_or(|(_, b)| r > b + 1e-9) {
                    best = Some((lead + k, r));
                }
            }
        }
        if let Some((end, _)) = best {
            return Some(toks[end - 1].span.end);
        }
        let norm: String = toks[lead].norm.iter().collect();
        if LEAD_FILLERS.contains(&norm.as_str()) {
            lead += 1;
        } else if frags < max_fragments {
            frags += 1;
            lead += 1;
        } else {
            return None;
        }
    }
    None
}

#[cfg(test)]
mod tests;
