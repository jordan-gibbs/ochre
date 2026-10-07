//! Pure text helpers for the injectors: UTF-16 batching and newline/tab planning.
//!
//! Windows (`KEYEVENTF_UNICODE`) and macOS (`CGEventKeyboardSetUnicodeString`) both take UTF-16
//! code units, and a batch boundary must never split a surrogate pair or the emoji arrives as
//! two replacement characters.

/// One piece of text to inject: a run of characters, or a key press.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment<'a> {
    Text(&'a str),
    Enter,
    Tab,
}

/// Split text into typed runs and key presses: newline is Enter, tab is Tab. CRLF and a lone CR
/// count as one newline, so Windows line endings never press Enter twice.
pub fn plan(text: &str) -> Vec<Segment<'_>> {
    let mut out = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\n' || b == b'\r' || b == b'\t' {
            if i > start {
                out.push(Segment::Text(&text[start..i]));
            }
            if b == b'\t' {
                out.push(Segment::Tab);
            } else {
                out.push(Segment::Enter);
                if b == b'\r' && bytes.get(i + 1) == Some(&b'\n') {
                    i += 1;
                }
            }
            start = i + 1;
        }
        i += 1;
    }
    if start < text.len() {
        out.push(Segment::Text(&text[start..]));
    }
    out
}

pub fn is_high_surrogate(unit: u16) -> bool {
    (0xD800..=0xDBFF).contains(&unit)
}

/// Length of the next batch from `start`: at most `max` UTF-16 units, extended by one rather
/// than splitting a surrogate pair.
pub fn next_batch(units: &[u16], start: usize, max: usize) -> usize {
    if start >= units.len() {
        return 0;
    }
    let mut n = (units.len() - start).min(max.max(1));
    if is_high_surrogate(units[start + n - 1]) && start + n < units.len() {
        n += 1;
    }
    n
}

/// `units` cut into slices of at most `max` units, never splitting a surrogate pair.
pub fn chunks(units: &[u16], max: usize) -> Vec<&[u16]> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < units.len() {
        let n = next_batch(units, start, max);
        out.push(&units[start..start + n]);
        start += n;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_splits_newlines_and_tabs() {
        use Segment::*;
        assert_eq!(plan(""), vec![]);
        assert_eq!(plan("hi"), vec![Text("hi")]);
        assert_eq!(plan("a\nb"), vec![Text("a"), Enter, Text("b")]);
        assert_eq!(plan("a\r\nb"), vec![Text("a"), Enter, Text("b")]);
        assert_eq!(plan("a\rb"), vec![Text("a"), Enter, Text("b")]);
        assert_eq!(plan("a\n\nb"), vec![Text("a"), Enter, Enter, Text("b")]);
        assert_eq!(plan("\ta\t"), vec![Tab, Text("a"), Tab]);
        assert_eq!(plan("é\n😀"), vec![Text("é"), Enter, Text("😀")]);
    }

    #[test]
    fn batches_never_split_surrogates() {
        let units: Vec<u16> = "ab😀cd".encode_utf16().collect(); // a b D83D DE00 c d
        assert_eq!(next_batch(&units, 0, 3), 4, "pair kept whole");
        assert_eq!(next_batch(&units, 0, 2), 2);
        assert_eq!(next_batch(&units, 4, 20), 2);
        assert_eq!(next_batch(&units, 6, 20), 0);
        let c = chunks(&units, 3);
        assert_eq!(c.len(), 2);
        assert_eq!(String::from_utf16(c[0]).unwrap(), "ab😀");
        let all: Vec<u16> = "😀😀😀".encode_utf16().collect();
        for c in chunks(&all, 1) {
            assert_eq!(String::from_utf16(c).unwrap(), "😀");
        }
        for max in 1..25 {
            let text = "naïve café 😀 emoji 👍🏽 and 中文 text";
            let units: Vec<u16> = text.encode_utf16().collect();
            let joined: String = chunks(&units, max)
                .iter()
                .map(|c| String::from_utf16(c).unwrap())
                .collect();
            assert_eq!(joined, text);
        }
    }
}
