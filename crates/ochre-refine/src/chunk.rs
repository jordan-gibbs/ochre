//! Chunked refinement for long dictations (docs/refine-chunking.md).
//!
//! The small local models pass 70-80% of short dictations but few long multi-paragraph ones
//! (eval-v4 "long", 120-170 words: 2B 19%, 0.8B 10% judged pass). [`split`] cuts a long transcript
//! into pieces of about [`TARGET_WORDS`] words at sentence and paragraph boundaries; each piece is
//! refined and guarded on its own and the results are joined with the separators returned here.
//!
//! Measured (3 blind judges, 62 split rows): no gain at 2B (+0.0 pts) or 0.8B (-1.6), and a
//! significant loss at 4B (-21.0 [-32.3, -9.7]). The long rows fail on per-sentence errors
//! (misrecognitions, punctuation, missed corrections) that chunking does not remove, and the
//! pieces lose the context that lets the model merge sentences the recognizer broke. So "auto"
//! turns it on for no model today ([`AUTO_MODELS`] is empty); "on" is there to try it.
//!
//! The boundary rules are deliberately conservative: a boundary is used only when nothing the
//! model must see together can straddle it.
//!
//! - Never next to a self-correction cue ("no wait", "actually", "sorry", "scratch that", ...):
//!   neither in the first words after the boundary nor in the last words before it, and never
//!   before a sentence that contains a strong cue anywhere.
//! - Never next to a short sentence (fewer than [`MIN_SENTENCE_WORDS`] words): fragments like
//!   "No sorry." or "Some context." belong to their neighbours.
//! - Never where the recognizer's period looks mid-thought: either sentence starts lowercase, the
//!   next one starts with a continuation word ("because", "which", "for", ...), the previous one
//!   is a dependent clause ("If we don't hear back by noon.") or ends on a function word ("the").
//! - Never inside a dictated list (GUIDE rule 8): from the lead-in sentence through the sentence
//!   after the last item cue ("bullet", "number one", "new line", "first ... second ...").
//! - A "new paragraph" voice command that clearly stands on its own (after a sentence end, or
//!   capitalized after a comma) is always a boundary: the command is removed and the pieces are
//!   joined with a blank line, which is what the model writes for it (GUIDE rule 9).
//!
//! Everything is ASCII-only and byte-for-byte mirrored by `src/openwhisprflow/refine/chunk.py`,
//! the eval harness's port (`tools/eval/chunk_parity.py` checks the two agree).

/// Below this many words a dictation is refined in one piece (the default threshold).
pub const MIN_WORDS: usize = 80;
/// Aim for pieces about this long: where the small models are strongest.
pub const TARGET_WORDS: usize = 40;
/// Sentences shorter than this are never split off from their neighbours.
pub const MIN_SENTENCE_WORDS: usize = 5;

/// What joins a piece to the one before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sep {
    /// The first piece.
    None,
    Space,
    /// A "new paragraph" command stood here.
    Paragraph,
}

impl Sep {
    pub fn as_str(self) -> &'static str {
        match self {
            Sep::None => "",
            Sep::Space => " ",
            Sep::Paragraph => "\n\n",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Piece {
    pub sep: Sep,
    pub text: String,
}

/// Correction cues that matter at the start of the next sentence or the end of the previous one.
const WEAK_CUES: &[&str] = &[
    "no",
    "nope",
    "wait",
    "actually",
    "sorry",
    "oops",
    "rather",
    "correction",
    "scratch",
    "strike",
    "nevermind",
    "pardon",
    "rephrase",
    "instead",
];
/// Cues that block a boundary before the sentence wherever they occur in it.
const STRONG_BIGRAMS: &[(&str, &str)] = &[
    ("i", "mean"),
    ("make", "that"),
    ("scratch", "that"),
    ("strike", "that"),
    ("or", "rather"),
    ("no", "wait"),
    ("never", "mind"),
    ("let", "me"),
];
const STRONG_WORDS: &[&str] = &["sorry", "correction", "rephrase", "nevermind"];

/// A sentence starting with one of these continues the previous one (the recognizer's period
/// was mid-thought).
const CONTINUATION: &[&str] = &[
    "and",
    "but",
    "or",
    "nor",
    "so",
    "because",
    "cause",
    "which",
    "who",
    "whom",
    "whose",
    "that",
    "where",
    "when",
    "while",
    "whereas",
    "although",
    "though",
    "unless",
    "until",
    "till",
    "if",
    "for",
    "to",
    "with",
    "without",
    "by",
    "from",
    "of",
    "in",
    "on",
    "at",
    "into",
    "like",
    "including",
    "especially",
    "than",
    "as",
    "not",
    "plus",
    "then",
    "usually",
    "mostly",
    "except",
    "instead",
    "otherwise",
    "only",
    "even",
    "rather",
    "both",
    "either",
    "neither",
    "after",
    "before",
    "since",
    "about",
    "via",
    "per",
    "or",
    "etc",
];

/// A sentence opening with one of these is a dependent clause ("If we don't hear back by noon.")
/// whose main clause follows.
const SUBORDINATORS: &[&str] = &[
    "if", "when", "whenever", "once", "unless", "although", "though", "because", "since", "while",
    "whereas", "until", "before", "after", "as",
];

/// A sentence ending on one of these was cut mid-phrase.
const FUNCTION_END: &[&str] = &[
    "the", "a", "an", "to", "of", "and", "or", "but", "with", "for", "in", "on", "at", "from",
    "by", "my", "your", "our", "their", "his", "her", "its", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "will", "would", "can", "could", "should", "if",
    "so", "because", "than", "as", "about", "into", "like", "i", "we", "you",
];

/// Tokens ending in "." that do not end a sentence.
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "st", "sr", "jr", "vs", "etc", "e.g", "i.e", "a.m", "p.m",
    "inc", "ltd", "co", "corp", "approx", "dept", "est", "fig", "no", "vol",
];

const LIST_STRONG: &[&str] = &["bullet", "bullets", "newline", "colon"];
const LIST_ORDINALS: &[&str] = &[
    "first", "firstly", "second", "secondly", "third", "thirdly", "fourth", "fifth", "lastly",
    "finally",
];
const SMALL_NUMBERS: &[&str] = &[
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "1", "2", "3",
    "4", "5", "6", "7", "8", "9", "10",
];

/// Lowercased token with leading/trailing non-ASCII-alphanumerics removed ("Paragraph," ->
/// "paragraph", "I'd" -> "i'd", "e.g." -> "e.g").
fn core(token: &str) -> String {
    token
        .trim_matches(|c: char| !c.is_ascii_alphanumeric())
        .to_ascii_lowercase()
}

const CLOSERS: &[char] = &['"', '\'', ')', ']', '\u{201d}', '\u{2019}'];
const OPENERS: &[char] = &['"', '\'', '(', '[', '\u{201c}', '\u{2018}'];

fn ends_sentence(token: &str) -> bool {
    let t = token.trim_end_matches(CLOSERS);
    if t.ends_with("...") || t.ends_with('\u{2026}') {
        return false; // trailing off: the restart belongs with it
    }
    if !t.ends_with(['.', '!', '?']) {
        return false;
    }
    let c = core(token);
    if t.ends_with('.') && (ABBREVIATIONS.contains(&c.as_str()) || c.len() == 1) {
        return false;
    }
    true
}

fn starts_upper(token: &str) -> bool {
    token
        .trim_start_matches(OPENERS)
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_uppercase())
}

/// Text before a standalone "new paragraph" command: ends a sentence, or a clause when the
/// command is capitalized.
fn paragraph_command_at(tokens: &[&str], i: usize) -> bool {
    if i == 0 || i + 2 >= tokens.len() {
        return false; // nothing before or after: not a boundary
    }
    let (new, para) = (tokens[i], tokens[i + 1]);
    if new != "New" && new != "new" {
        return false;
    }
    let p = para.trim_end_matches(['.', ',', '!', '?', ':', ';']);
    if !p.eq_ignore_ascii_case("paragraph") {
        return false;
    }
    let prev = tokens[i - 1].trim_end_matches(CLOSERS);
    prev.ends_with(['.', '!', '?']) || (new == "New" && prev.ends_with([',', ';', ':']))
}

/// Split at standalone "new paragraph" commands (the command is dropped).
fn paragraphs<'a>(tokens: &[&'a str]) -> Vec<Vec<&'a str>> {
    let mut out: Vec<Vec<&str>> = vec![Vec::new()];
    let mut i = 0;
    while i < tokens.len() {
        if paragraph_command_at(tokens, i) {
            out.push(Vec::new());
            i += 2;
            continue;
        }
        out.last_mut().expect("non-empty").push(tokens[i]);
        i += 1;
    }
    out.retain(|p| !p.is_empty());
    out
}

fn sentences<'a>(tokens: &[&'a str]) -> Vec<Vec<&'a str>> {
    let mut out = Vec::new();
    let mut cur = Vec::new();
    for &t in tokens {
        cur.push(t);
        if ends_sentence(t) {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn has_bigram(words: &[String], pairs: &[(&str, &str)]) -> bool {
    words
        .windows(2)
        .any(|w| pairs.iter().any(|(a, b)| w[0] == *a && w[1] == *b))
}

fn list_cue(words: &[String]) -> (bool, bool) {
    let strong = words.iter().any(|w| LIST_STRONG.contains(&w.as_str()))
        || words.windows(2).any(|w| {
            (matches!(w[0].as_str(), "new" | "next") && w[1] == "line")
                || (matches!(w[0].as_str(), "number" | "step" | "item" | "point")
                    && SMALL_NUMBERS.contains(&w[1].as_str()))
        });
    let ordinal = words
        .iter()
        .take(3)
        .any(|w| LIST_ORDINALS.contains(&w.as_str()));
    (strong, ordinal)
}

/// May the text be cut between sentences `s` and `t`?
fn boundary_ok(s: &[&str], t: &[&str]) -> bool {
    if s.len() < MIN_SENTENCE_WORDS || t.len() < MIN_SENTENCE_WORDS {
        return false;
    }
    // A lowercase start on either side is the recognizer cutting a run-on or an enumeration
    // ("covers who receives it. how fast it goes in the fridge. What to do if ...").
    if !starts_upper(t[0]) || !starts_upper(s[0]) {
        return false;
    }
    let sw: Vec<String> = s.iter().map(|x| core(x)).collect();
    let tw: Vec<String> = t.iter().map(|x| core(x)).collect();
    if CONTINUATION.contains(&tw[0].as_str()) {
        return false;
    }
    if FUNCTION_END.contains(&sw[sw.len() - 1].as_str()) || SUBORDINATORS.contains(&sw[0].as_str())
    {
        return false;
    }
    if tw.iter().take(4).any(|w| WEAK_CUES.contains(&w.as_str()))
        || sw
            .iter()
            .skip(sw.len().saturating_sub(4))
            .any(|w| WEAK_CUES.contains(&w.as_str()))
    {
        return false;
    }
    let tail = &sw[sw.len().saturating_sub(4)..];
    if has_bigram(&tw, STRONG_BIGRAMS)
        || tw.iter().any(|w| STRONG_WORDS.contains(&w.as_str()))
        || has_bigram(tail, STRONG_BIGRAMS)
    {
        return false;
    }
    true
}

/// Allowed cut after each sentence but the last.
fn boundaries(sents: &[Vec<&str>]) -> Vec<bool> {
    let n = sents.len();
    let mut ok: Vec<bool> = (0..n.saturating_sub(1))
        .map(|j| boundary_ok(&sents[j], &sents[j + 1]))
        .collect();
    // Lists stay whole, lead-in sentence and the sentence after the last cue included.
    let cues: Vec<(bool, bool)> = sents
        .iter()
        .map(|s| list_cue(&s.iter().map(|x| core(x)).collect::<Vec<_>>()))
        .collect();
    let ordinal_count = cues.iter().filter(|c| c.1).count();
    let flagged: Vec<usize> = (0..n)
        .filter(|&j| cues[j].0 || (cues[j].1 && ordinal_count >= 2))
        .collect();
    if let (Some(&first), Some(&last)) = (flagged.first(), flagged.last()) {
        let lo = first.saturating_sub(1);
        let hi = (last + 1).min(n - 1);
        for (j, b) in ok.iter_mut().enumerate() {
            if j >= lo && j < hi {
                *b = false;
            }
        }
    }
    ok
}

/// Pack one paragraph's sentences into pieces of about `target` words.
fn pack(sents: &[Vec<&str>], target: usize) -> Vec<String> {
    let ok = boundaries(sents);
    // Units: runs of sentences joined by boundaries that may not be cut.
    let mut units: Vec<Vec<&str>> = vec![Vec::new()];
    for (j, s) in sents.iter().enumerate() {
        units.last_mut().expect("non-empty").extend_from_slice(s);
        if j < ok.len() && ok[j] {
            units.push(Vec::new());
        }
    }
    let total: usize = units.iter().map(Vec::len).sum();
    if total <= target + target / 2 || units.len() < 2 {
        return vec![units.concat().join(" ")];
    }
    let k = total.div_ceil(target);
    let goal = total.div_ceil(k);
    let mut pieces: Vec<Vec<&str>> = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    for u in units {
        if !cur.is_empty() && cur.len() + u.len() > goal && cur.len() * 2 >= goal {
            pieces.push(std::mem::take(&mut cur));
        }
        cur.extend(u);
    }
    if !cur.is_empty() {
        pieces.push(cur);
    }
    if pieces.len() >= 2 && pieces[pieces.len() - 1].len() < target / 3 {
        let last = pieces.pop().expect("len >= 2");
        pieces.last_mut().expect("len >= 1").extend(last);
    }
    pieces.into_iter().map(|p| p.join(" ")).collect()
}

/// Split `text` for chunked refinement. A text under `min_words` words, or one with no usable
/// boundary, comes back as a single piece holding `text` unchanged.
pub fn split(text: &str, min_words: usize, target: usize) -> Vec<Piece> {
    let whole = || {
        vec![Piece {
            sep: Sep::None,
            text: text.to_string(),
        }]
    };
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.len() < min_words.max(1) || target == 0 {
        return whole();
    }
    let mut out = Vec::new();
    for (pi, para) in paragraphs(&tokens).into_iter().enumerate() {
        let mut para = para;
        // "today, New paragraph": the clause comma before the command goes with it.
        if let Some(last) = para.last_mut() {
            *last = last.trim_end_matches([',', ';', ':']);
        }
        para.retain(|t| !t.is_empty());
        if para.is_empty() {
            continue;
        }
        let sents = sentences(&para);
        for (k, p) in pack(&sents, target).into_iter().enumerate() {
            let sep = if out.is_empty() {
                Sep::None
            } else if k == 0 && pi > 0 {
                Sep::Paragraph
            } else {
                Sep::Space
            };
            out.push(Piece { sep, text: p });
        }
    }
    if out.len() < 2 {
        return whole();
    }
    out
}

/// Join refined pieces back with their separators.
pub fn join<'a>(pieces: impl IntoIterator<Item = (Sep, &'a str)>) -> String {
    let mut out = String::new();
    for (sep, text) in pieces {
        out.push_str(sep.as_str());
        out.push_str(text);
    }
    out
}

/// Model sizes where chunking won the judged comparison (docs/refine-chunking.md), so
/// `chunk_long = "auto"` turns it on for them. None of the v4 models did.
pub const AUTO_MODELS: &[&str] = &[];

/// `chunk_long` ("auto" | "on" | "off") for a refiner that resolved to `model` (None: a cloud or
/// external refiner, where "auto" means off).
pub fn enabled(setting: &str, model: Option<&str>) -> bool {
    match setting {
        "on" => true,
        "off" => false,
        _ => model.is_some_and(|m| AUTO_MODELS.contains(&m)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(p: &[Piece]) -> Vec<(&'static str, String)> {
        p.iter().map(|x| (x.sep.as_str(), x.text.clone())).collect()
    }

    fn words(n: usize, w: &str) -> String {
        vec![w; n].join(" ")
    }

    /// A plain sentence of `n` words that starts with a capital and ends with a period.
    fn sent(n: usize) -> String {
        format!("We {} done.", words(n - 2, "ship"))
    }

    #[test]
    fn short_text_is_one_untouched_piece() {
        let t = "Um so, the build. Failed  twice.";
        assert_eq!(
            split(t, 80, 40),
            vec![Piece {
                sep: Sep::None,
                text: t.into()
            }]
        );
        assert_eq!(split("", 80, 40).len(), 1);
    }

    #[test]
    fn long_prose_splits_at_sentence_ends_and_rejoins() {
        let t = (0..10).map(|_| sent(10)).collect::<Vec<_>>().join(" ");
        let p = split(&t, 80, 40);
        assert!(p.len() >= 2, "{p:?}");
        assert!(p.iter().all(|x| x.text.split_whitespace().count() <= 60));
        assert_eq!(p[0].sep, Sep::None);
        assert!(p[1..].iter().all(|x| x.sep == Sep::Space));
        let joined = join(p.iter().map(|x| (x.sep, x.text.as_str())));
        assert_eq!(joined, t);
    }

    #[test]
    fn never_splits_before_a_correction_or_a_fragment() {
        let a = sent(40);
        let b = "Sorry the meeting moved to Friday not Thursday at all.";
        let c = sent(40);
        let t = format!("{a} {b} {c}");
        let p = split(&t, 80, 40);
        // The boundary before "Sorry ..." is locked; the one after it is fine.
        assert_eq!(texts(&p), vec![("", format!("{a} {b}")), (" ", c.clone())]);
        let t = format!("{a} No sorry. {c}");
        assert_eq!(split(&t, 80, 40).len(), 1);
        let t = format!("{a} Actually we could also keep it on Monday. {c}");
        assert_eq!(
            split(&t, 80, 40)[0].text,
            format!("{a} Actually we could also keep it on Monday.")
        );
    }

    #[test]
    fn mid_thought_periods_are_not_boundaries() {
        let a = sent(40);
        let c = sent(40);
        for b in [
            "because the rim is also starting to wear down.",
            "Because the rim is also starting to wear down.",
            "For handling the cold chain deliveries every week.",
        ] {
            let t = format!("{a} {b} {c}");
            let p = split(&t, 80, 40);
            assert!(p[0].text.starts_with(&format!("{a} {b}")), "{b}");
        }
        let t = format!("{} the. {}", words(40, "we"), sent(40));
        assert_eq!(split(&t, 80, 40).len(), 1);
    }

    #[test]
    fn new_paragraph_command_becomes_the_separator() {
        let a = sent(45);
        let c = sent(45);
        let t = format!("{} today, New paragraph also {}", &a[..a.len() - 1], c);
        let p = split(&t, 80, 40);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].text, a[..a.len() - 1].to_string() + " today");
        assert_eq!(
            (p[1].sep, p[1].text.as_str()),
            (Sep::Paragraph, format!("also {c}").as_str())
        );
        let t = format!("{a} New paragraph. {c}");
        assert_eq!(
            texts(&split(&t, 80, 40)),
            vec![("", a.clone()), ("\n\n", c.clone())]
        );
        // Not a command: inside a sentence, or lowercase after a comma.
        let t = format!("{a} The new paragraph about pricing is far too long, I think. {c}");
        assert!(split(&t, 80, 40).iter().all(|x| x.sep != Sep::Paragraph));
        let t = format!("{} done, new paragraph styles {c}", words(45, "we"));
        assert!(split(&t, 80, 40).iter().all(|x| x.sep != Sep::Paragraph));
    }

    #[test]
    fn a_dictated_list_stays_in_one_piece() {
        let lead = sent(30);
        let t = format!(
            "{lead} The three things are these ones here. First we ship the new build today. \
             Second we tell the whole support team about it. Third we watch the error rates. {}",
            sent(30)
        );
        let p = split(&t, 80, 40);
        let list = p.iter().find(|x| x.text.contains("First")).unwrap();
        assert!(list.text.contains("The three things") && list.text.contains("Third"));
        // A "colon ... new line" list locks the lead-in and the sentence after it too.
        let t = format!(
            "{lead} Steps colon new line one cool the fridge. {}",
            sent(60)
        );
        assert_eq!(split(&t, 80, 40).len(), 1);
    }

    #[test]
    fn abbreviations_and_trailing_off_do_not_end_sentences() {
        let t = format!(
            "{} ask Dr. Smith about it and so on... {}",
            sent(40),
            sent(40)
        );
        let p = split(&t, 80, 40);
        assert!(
            p.iter()
                .all(|x| !x.text.ends_with("Dr.") && !x.text.ends_with("..."))
        );
    }

    #[test]
    fn auto_only_for_the_winning_sizes() {
        assert!(enabled("on", None));
        assert!(enabled("on", Some("ochre-refine-4b")));
        assert!(!enabled("off", Some("ochre-refine-2b")));
        // No v4 size won the judged comparison.
        for m in [
            "ochre-refine-4b",
            "ochre-refine-2b",
            "ochre-refine-0.8b",
            "quill-2b",
        ] {
            assert!(!enabled("auto", Some(m)), "{m}");
        }
        assert!(!enabled("auto", None));
    }
}
