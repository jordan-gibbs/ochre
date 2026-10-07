//! The leading-space join rule (SPEC §6.2).
//!
//! Dictating twice in a row into the same field must not glue the sentences together
//! ("Hello there.How are you?"). We do not look up the caret (asking the
//! focused control what precedes the caret is slow and unreliable across apps).
//! Instead the orchestrator keeps one [`JoinMemory`] and runs every insertion through it:
//!
//! * With `trailing_space` (the default) every insertion ends with one space, so the next one
//!   (or the user's own typing) never runs into it.
//! * Otherwise, one space is prepended when the previous insertion went to the same app and
//!   window, ended with a non-space, and less than `join_window` has passed.
//!
//! Both are skipped when the new text starts with whitespace or with punctuation that attaches
//! to the previous word (`, . ; : ! ? ) ] }` ...), and nothing is prepended when the window
//! is unknown (no app and no window id), because guessing wrong is worse than not joining.

use std::time::{Duration, Instant};

use ochre_core::platform::FocusInfo;

#[derive(Debug, Clone, PartialEq, Eq)]
struct LastInsertion {
    app_name: String,
    window_id: String,
    ended_with_space: bool,
    at: Instant,
}

/// Memory of the last insertion, owned by the orchestrator.
#[derive(Debug, Clone, Default)]
pub struct JoinMemory {
    last: Option<LastInsertion>,
}

impl JoinMemory {
    pub fn new() -> Self {
        Self::default()
    }

    /// The text to inject for `text`, recording it as the latest insertion. Call
    /// [`JoinMemory::forget`] if the insertion then fails.
    pub fn join_text(
        &mut self,
        text: &str,
        focus: &FocusInfo,
        trailing_space: bool,
        join_window: Duration,
    ) -> String {
        self.join_text_at(text, focus, trailing_space, join_window, Instant::now())
    }

    /// [`JoinMemory::join_text`] with an explicit clock (for tests and replays).
    pub fn join_text_at(
        &mut self,
        text: &str,
        focus: &FocusInfo,
        trailing_space: bool,
        join_window: Duration,
        now: Instant,
    ) -> String {
        let body = text.trim_matches(|c: char| c == ' ' || c == '\u{a0}');
        if body.is_empty() {
            return String::new();
        }
        let known = !(focus.app_name.is_empty() && focus.window_id.is_empty());
        let joins = known
            && self.last.as_ref().is_some_and(|last| {
                !last.ended_with_space
                    && last.app_name == focus.app_name
                    && last.window_id == focus.window_id
                    && now.saturating_duration_since(last.at) < join_window
            });
        let first = body.chars().next().unwrap_or(' ');
        let attaches = first.is_whitespace() || is_closing_punct(first);
        let mut out = String::with_capacity(body.len() + 2);
        if joins && !attaches {
            out.push(' ');
        }
        out.push_str(body);
        if trailing_space && !body.ends_with(char::is_whitespace) {
            out.push(' ');
        }
        self.last = Some(LastInsertion {
            app_name: focus.app_name.clone(),
            window_id: focus.window_id.clone(),
            ended_with_space: out.ends_with(char::is_whitespace),
            at: now,
        });
        out
    }

    /// Forget the last insertion (it failed, or the user switched context).
    pub fn forget(&mut self) {
        self.last = None;
    }
}

/// `join_text` as a free function over the orchestrator's memory.
pub fn join_text(
    memory: &mut JoinMemory,
    text: &str,
    focus: &FocusInfo,
    trailing_space: bool,
    join_window: Duration,
) -> String {
    memory.join_text(text, focus, trailing_space, join_window)
}

fn is_closing_punct(c: char) -> bool {
    matches!(
        c,
        ',' | '.'
            | ';'
            | ':'
            | '!'
            | '?'
            | ')'
            | ']'
            | '}'
            | '…'
            | '%'
            | '’'
            | '”'
            | '»'
            | '、'
            | '。'
            | '，'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: Duration = Duration::from_secs(20);

    fn focus(app: &str, win: &str) -> FocusInfo {
        FocusInfo {
            app_name: app.into(),
            window_title: String::new(),
            window_id: win.into(),
            elevated: false,
        }
    }

    #[test]
    fn trailing_space_mode() {
        let mut m = JoinMemory::new();
        let t0 = Instant::now();
        let f = focus("slack", "1");
        assert_eq!(
            m.join_text_at("Hello there.", &f, true, W, t0),
            "Hello there. "
        );
        assert_eq!(
            m.join_text_at("How are you?", &f, true, W, t0 + Duration::from_secs(2)),
            "How are you? "
        );
        assert_eq!(m.join_text_at("  padded  ", &f, true, W, t0), "padded ");
        assert_eq!(
            m.join_text_at("line\n", &f, true, W, t0),
            "line\n",
            "ends with whitespace already"
        );
    }

    #[test]
    fn leading_space_mode_same_window_within_window() {
        let mut m = JoinMemory::new();
        let t0 = Instant::now();
        let f = focus("code", "a:b");
        assert_eq!(m.join_text_at("first", &f, false, W, t0), "first");
        assert_eq!(
            m.join_text_at("second", &f, false, W, t0 + Duration::from_secs(5)),
            " second"
        );
        assert_eq!(
            m.join_text_at(", third", &f, false, W, t0 + Duration::from_secs(6)),
            ", third"
        );
        assert_eq!(
            m.join_text_at("fourth", &f, false, W, t0 + Duration::from_secs(7)),
            " fourth"
        );
    }

    #[test]
    fn no_join_across_windows_apps_or_time() {
        let t0 = Instant::now();
        let mut m = JoinMemory::new();
        m.join_text_at("one", &focus("code", "1"), false, W, t0);
        assert_eq!(
            m.join_text_at("two", &focus("code", "2"), false, W, t0),
            "two"
        );
        assert_eq!(
            m.join_text_at("three", &focus("slack", "2"), false, W, t0),
            "three"
        );
        assert_eq!(
            m.join_text_at("four", &focus("slack", "2"), false, W, t0 + W),
            "four",
            "window elapsed"
        );
    }

    #[test]
    fn no_join_after_space_or_into_unknown_window() {
        let t0 = Instant::now();
        let mut m = JoinMemory::new();
        m.join_text_at("ends with a newline\n", &focus("x", "1"), false, W, t0);
        assert_eq!(
            m.join_text_at("next", &focus("x", "1"), false, W, t0),
            "next"
        );
        let unknown = FocusInfo::default();
        m.join_text_at("a", &unknown, false, W, t0);
        assert_eq!(m.join_text_at("b", &unknown, false, W, t0), "b");
    }

    #[test]
    fn empty_text_is_empty_and_not_recorded() {
        let t0 = Instant::now();
        let mut m = JoinMemory::new();
        m.join_text_at("one", &focus("x", "1"), false, W, t0);
        assert_eq!(m.join_text_at("   ", &focus("x", "1"), false, W, t0), "");
        assert_eq!(
            m.join_text_at("two", &focus("x", "1"), false, W, t0),
            " two"
        );
    }

    #[test]
    fn forget_and_free_fn() {
        let mut m = JoinMemory::new();
        let f = focus("x", "1");
        assert_eq!(join_text(&mut m, "one", &f, false, W), "one");
        m.forget();
        assert_eq!(join_text(&mut m, "two", &f, false, W), "two");
        assert_eq!(join_text(&mut m, "three", &f, false, W), " three");
    }
}
