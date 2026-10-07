//! Post-STT corrections: whole-phrase replacements and preferred
//! spellings from the user's dictionary. Deterministic and cheap, so it always runs, refinement
//! or not.

use ochre_core::config::DictionaryConfig;

/// Apply replacements (longest phrase first) then preferred spellings, matching whole words
/// case-insensitively. Replacements keep the user's written form exactly; preferred spellings
/// keep a leading capital when the recognizer capitalized the word (sentence start).
pub fn apply_corrections(text: &str, dict: &DictionaryConfig) -> String {
    let mut out = text.to_string();
    let mut reps: Vec<(&String, &String)> = dict
        .replacements
        .iter()
        .filter(|(k, _)| !k.trim().is_empty())
        .collect();
    reps.sort_by_key(|(k, _)| std::cmp::Reverse(k.len()));
    for (said, written) in reps {
        out = replace_whole(&out, said, |_| written.clone());
    }
    for word in dict.words.iter().filter(|w| !w.trim().is_empty()) {
        out = replace_whole(&out, word, |found| {
            let starts_upper = found.chars().next().is_some_and(char::is_uppercase);
            let mut fixed = word.clone();
            if starts_upper && !fixed.chars().next().is_some_and(char::is_uppercase) {
                fixed = capitalize(&fixed);
            }
            fixed
        });
    }
    out
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    c.next()
        .map(|f| f.to_uppercase().chain(c).collect())
        .unwrap_or_default()
}

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '\'' || c == '_'
}

/// Replace every case-insensitive, whole-word occurrence of `needle` in `hay`.
fn replace_whole(hay: &str, needle: &str, mut with: impl FnMut(&str) -> String) -> String {
    let needle_lower = needle.to_lowercase();
    let lower = hay.to_lowercase();
    // Lowercasing can change byte lengths for some scripts; only take the fast path when it doesn't.
    if lower.len() != hay.len() || needle_lower.len() != needle.len() {
        return hay.to_string();
    }
    let mut out = String::with_capacity(hay.len());
    let mut i = 0;
    while let Some(pos) = lower[i..].find(&needle_lower) {
        let start = i + pos;
        let end = start + needle_lower.len();
        let before_ok = hay[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !is_word_char(c));
        let after_ok = hay[end..].chars().next().is_none_or(|c| !is_word_char(c));
        out.push_str(&hay[i..start]);
        if before_ok && after_ok {
            out.push_str(&with(&hay[start..end]));
        } else {
            out.push_str(&hay[start..end]);
        }
        i = end;
    }
    out.push_str(&hay[i..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dict(words: &[&str], reps: &[(&str, &str)]) -> DictionaryConfig {
        DictionaryConfig {
            words: words.iter().map(|s| s.to_string()).collect(),
            replacements: reps
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
        }
    }

    #[test]
    fn preferred_spelling_whole_word_and_case() {
        let d = dict(&["Kubernetes", "iPhone"], &[]);
        assert_eq!(
            apply_corrections("deploy to kubernetes now", &d),
            "deploy to Kubernetes now"
        );
        assert_eq!(
            apply_corrections("Iphone and iphones", &d),
            "IPhone and iphones"
        );
        assert_eq!(apply_corrections("my iphone.", &d), "my iPhone.");
    }

    #[test]
    fn replacements_longest_first_and_exact_form() {
        let d = dict(
            &[],
            &[("open whisper", "OpenAI Whisper"), ("whisper", "Whisper")],
        );
        assert_eq!(
            apply_corrections("I love open whisper", &d),
            "I love OpenAI Whisper"
        );
        assert_eq!(apply_corrections("whispering", &d), "whispering");
    }

    #[test]
    fn empty_dictionary_is_identity() {
        assert_eq!(
            apply_corrections("Héllo, wörld!", &DictionaryConfig::default()),
            "Héllo, wörld!"
        );
    }
}

/// Pure hesitation sounds. Never meaningful in written text, so they are removed deterministically
/// before refinement (and when refinement is off); the model never has to learn them. Ambiguous
/// discourse words ("like", "you know", "I mean") are left for the refiner.
const HESITATIONS: &[&str] = &[
    "um", "umm", "ummm", "uh", "uhh", "uhhh", "uhm", "ah", "ahh", "er", "erm", "hm", "hmm", "hmmm",
    "mm", "mmm", "mhm",
];

fn core_lower(token: &str) -> String {
    token
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase()
}

fn is_hesitation(token: &str) -> bool {
    let raw_core = token.trim_matches(|c: char| !c.is_alphanumeric());
    // A two-letter all-caps token is an acronym, not a sound: "the ER", "UM football". Longer
    // all-caps forms ("UMM", "HMM") are recognizers shouting a hesitation.
    if raw_core.chars().count() == 2 && raw_core.chars().all(|c| c.is_uppercase()) {
        return false;
    }
    let core = core_lower(token);
    !core.is_empty() && HESITATIONS.contains(&core.as_str())
}

/// Remove hesitation sounds, repairing the commas and capitals the recognizer put around them:
/// "Um, so the, uh, build failed. Uh." -> "So the build failed."
pub fn strip_hesitations(text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut capitalize_next = false;
    for token in text.split_whitespace() {
        if !is_hesitation(token) {
            let mut t = token.to_string();
            if capitalize_next {
                t = capitalize(&t);
                capitalize_next = false;
            }
            out.push(t);
            continue;
        }
        let at_sentence_start = out.last().is_none_or(|p| p.ends_with(['.', '!', '?']));
        let terminal = token
            .trim_end_matches(['"', '\'', ')'])
            .chars()
            .last()
            .filter(|c| matches!(c, '.' | '!' | '?'));
        if let Some(prev) = out.last_mut() {
            if let Some(t) = terminal {
                // "we're done, uh." -> "we're done."
                let trimmed = prev.trim_end_matches([',', ';']).to_string();
                *prev = if trimmed.ends_with(['.', '!', '?']) {
                    trimmed
                } else {
                    format!("{trimmed}{t}")
                };
            } else if token.ends_with(',') && prev.ends_with(',') {
                // "the, uh, file" -> "the file": the commas only fenced the hesitation.
                prev.pop();
                if prev.is_empty() {
                    // A detached comma ("can we , um, get") leaves nothing behind.
                    out.pop();
                }
            }
        }
        if at_sentence_start && token.chars().next().is_some_and(char::is_uppercase) {
            capitalize_next = true;
        } else if at_sentence_start && out.is_empty() {
            capitalize_next = text
                .chars()
                .find(|c| c.is_alphabetic())
                .is_some_and(char::is_uppercase);
        }
    }
    out.join(" ")
}

#[cfg(test)]
mod hesitation_tests {
    use super::strip_hesitations as s;

    #[test]
    fn removes_and_repairs() {
        assert_eq!(
            s("Um, so the, uh, build failed. Uh."),
            "So the build failed."
        );
        assert_eq!(s("um so the build failed"), "so the build failed");
        assert_eq!(s("we're done, uh."), "we're done.");
        assert_eq!(
            s("I think uh we should hmm ship it"),
            "I think we should ship it"
        );
        assert_eq!(s("Yes. Umm, let's go."), "Yes. Let's go.");
        assert_eq!(s("UMM, ship it. HMM."), "Ship it.");
        assert_eq!(
            s("hey man so, uh, can we , um, get that going?"),
            "hey man so can we get that going?"
        );
    }

    #[test]
    fn leaves_real_words_alone() {
        for keep in [
            "The UM department",
            "Ahmed and Erma",
            "summer humming",
            "I like it, you know.",
            "uhhuh is a word",
        ] {
            assert_eq!(s(keep), keep);
        }
        assert_eq!(s(""), "");
    }
}

/// Shortest phrase [`collapse_repeats`] collapses when it is said three or more times in a row.
/// One- and two-word repeats ("very, very good", "no no no", "thank you, thank you") are usually
/// deliberate and are left to the refiner.
pub const MIN_REPEAT_WORDS: usize = 3;
/// Shortest phrase collapsed when it is said exactly twice. Short doubled phrases are often
/// deliberate ("Wait for it, wait for it", "What a night, what a night", "...the final round. The
/// final round is a half day"), which the training and eval targets keep (docs/refine-chunking.md).
pub const MIN_PAIR_REPEAT_WORDS: usize = 5;
/// Longest phrase it looks for (a restarted sentence).
pub const MAX_REPEAT_WORDS: usize = 40;

/// Comparison key: ASCII punctuation removed, ASCII letters lowercased ("Long." == "long",
/// "2.8B" == "2.8b"). Everything else is kept as is, so two different non-ASCII words never match.
fn repeat_key(token: &str) -> String {
    token
        .chars()
        .filter(|c| !c.is_ascii_punctuation())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

const NUMBER_WORDS: &[&str] = &[
    "zero",
    "oh",
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
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "point",
    "dot",
    "dash",
];

/// Spoken commands: a dictated list repeats them on purpose ("make new line, make new line").
const COMMAND_WORDS: &[&str] = &["bullet", "comma", "period", "colon", "newline"];

/// A phrase that may be collapsed: every token has letters or digits, it is not all one word
/// ("no no no no no no"), not only numbers (a dictated code or count can repeat) and holds no
/// voice command.
fn collapsible(keys: &[String]) -> bool {
    keys.iter().all(|k| !k.is_empty())
        && keys.iter().any(|k| k != &keys[0])
        && !keys
            .iter()
            .all(|k| k.chars().all(|c| c.is_ascii_digit()) || NUMBER_WORDS.contains(&k.as_str()))
        && !keys.iter().any(|k| COMMAND_WORDS.contains(&k.as_str()))
        && !keys
            .windows(2)
            .any(|w| w[0] == "new" && (w[1] == "line" || w[1] == "paragraph"))
}

/// Collapse a phrase said several times in a row (case and ASCII punctuation ignored) into one
/// copy: the speaker restarting, which the recognizer transcribes every time. Three or more copies
/// need [`MIN_REPEAT_WORDS`] words, exactly two need [`MIN_PAIR_REPEAT_WORDS`].
/// "The 2.8B long. The 2.8B long. The 2.8B long." -> "The 2.8B long." The kept copy is the first
/// one, with the last copy's final token (its punctuation ends the thought). The longest repeated
/// phrase at each position wins. Text with no repeat comes back unchanged.
pub fn collapse_repeats(text: &str) -> String {
    let toks: Vec<&str> = text.split_whitespace().collect();
    let keys: Vec<String> = toks.iter().map(|t| repeat_key(t)).collect();
    let n = toks.len();
    let copies_at = |i: usize, len: usize| {
        let mut c = 1;
        while i + (c + 1) * len <= n && keys[i + c * len..i + (c + 1) * len] == keys[i..i + len] {
            c += 1;
        }
        c
    };
    let mut out: Vec<&str> = Vec::with_capacity(n);
    let mut changed = false;
    let mut i = 0;
    while i < n {
        let longest = MAX_REPEAT_WORDS.min((n - i) / 2);
        let hit = (MIN_REPEAT_WORDS..=longest).rev().find_map(|len| {
            let c = copies_at(i, len);
            let enough = c >= 3 || (c == 2 && len >= MIN_PAIR_REPEAT_WORDS);
            (enough && collapsible(&keys[i..i + len])).then_some((len, c))
        });
        let Some((len, copies)) = hit else {
            out.push(toks[i]);
            i += 1;
            continue;
        };
        out.extend_from_slice(&toks[i..i + len - 1]);
        out.push(toks[i + copies * len - 1]);
        i += copies * len;
        changed = true;
    }
    if changed {
        out.join(" ")
    } else {
        text.to_string()
    }
}

/// The app's whole deterministic pre-pass on the joined recognizer text, before refinement:
/// hesitations out, immediate repeats collapsed, then the user's dictionary.
pub fn prepass(joined: &str, dict: &DictionaryConfig) -> String {
    apply_corrections(&collapse_repeats(&strip_hesitations(joined)), dict)
}

#[cfg(test)]
mod repeat_tests {
    use super::collapse_repeats as c;

    #[test]
    fn collapses_a_restarted_phrase() {
        assert_eq!(
            c("The 2.8B long. The 2.8B long. The 2.8B long."),
            "The 2.8B long."
        );
        assert_eq!(
            c("So the plan is the plan is the plan is to ship Friday."),
            "So the plan is to ship Friday."
        );
        assert_eq!(
            c("we should ship it today, We should ship it today. Then test."),
            "we should ship it today. Then test."
        );
        // The longest repeat wins.
        assert_eq!(
            c("Send the deck to Sam. Send the deck to Sam. Thanks."),
            "Send the deck to Sam. Thanks."
        );
    }

    #[test]
    fn leaves_deliberate_and_short_repeats_alone() {
        for keep in [
            "very, very good",
            "no no no",
            "no no no no no no",
            "thank you, thank you",
            "It is what it is.",
            "one two three one two three",
            "The code is 4 5 6 4 5 6.",
            "Über alles über alles über alles",
            "",
            "  spacing   is   kept  ",
            // Doubled short phrases are often deliberate.
            "Wait for it, wait for it. The whole room went silent.",
            "The thing is, the thing is, nobody told us.",
            "moving on to the final round. The final round is a half day",
            "ask about the bill, ask about the bill again",
            // Voice commands repeat in a dictated list.
            "C D build new line, make new line, make new line, make install.",
            "milk comma eggs comma milk comma eggs comma milk comma eggs",
        ] {
            assert_eq!(c(keep), keep, "{keep:?}");
        }
        // Different non-ASCII words never compare equal.
        assert_eq!(c("ça va bien ça va bien ça va bien"), "ça va bien");
        assert_eq!(
            c("ça va bien çé va bien ça va bien"),
            "ça va bien çé va bien ça va bien"
        );
    }
}

/// What two dictations must share to count as the same thing said again: the words, lowercased,
/// with punctuation and spacing ignored ("Ship it Friday." == "ship it, friday").
pub fn same_words_key(text: &str) -> String {
    text.split_whitespace()
        .map(|t| {
            t.chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod same_words_tests {
    use super::same_words_key as k;

    #[test]
    fn ignores_case_punctuation_and_spacing() {
        assert_eq!(k("Ship it Friday."), k("ship   it, friday"));
        assert_eq!(k("Ship it Friday."), "ship it friday");
        assert_ne!(k("Ship it Friday."), k("Ship it Monday."));
        assert_eq!(k(" ... "), "");
    }
}
