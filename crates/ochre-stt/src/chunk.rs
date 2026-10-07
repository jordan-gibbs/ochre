//! Offline splitting of long recordings for engines with a length limit.

const FRAME_S: f32 = 0.02;
const CUT_WINDOW_S: f32 = 5.0;

/// Split `pcm` into chunks of at most `max` samples, each cut at the quietest 20 ms frame of its
/// last 5 s, so no word is split at a loud point. Audio that already fits is returned whole (best
/// for accuracy). Port of the prototype's `split_at_pauses`.
pub fn split_at_pauses(pcm: &[f32], rate: usize, max: usize) -> Vec<&[f32]> {
    let width = ((rate as f32 * FRAME_S).round() as usize).max(1);
    let window = ((CUT_WINDOW_S * rate as f32) as usize).min(max.saturating_sub(width));
    let mut out = Vec::new();
    let mut pos = 0;
    while pcm.len() - pos > max {
        let lo = pos + (max - window).max(width);
        let seg = &pcm[lo..pos + max];
        let n = seg.len() / width;
        let mut best = (f32::INFINITY, 0usize);
        for i in 0..n {
            let e: f32 = seg[i * width..(i + 1) * width].iter().map(|x| x * x).sum();
            if e < best.0 {
                best = (e, i);
            }
        }
        let cut = lo + (best.1 + 1) * width;
        out.push(&pcm[pos..cut]);
        pos = cut;
    }
    out.push(&pcm[pos..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_audio_is_whole_and_long_audio_cuts_at_the_gap() {
        let rate = 16_000;
        assert_eq!(
            split_at_pauses(&vec![0.1; rate * 10], rate, rate * 30).len(),
            1
        );
        // 40 s of tone with a silent gap at 27-27.2 s.
        let mut pcm: Vec<f32> = (0..rate * 40)
            .map(|i| (i as f32 * 0.2).sin() * 0.3)
            .collect();
        pcm[27 * rate..27 * rate + rate / 5]
            .iter_mut()
            .for_each(|x| *x = 0.0);
        let parts = split_at_pauses(&pcm, rate, rate * 30);
        assert_eq!(parts.len(), 2);
        let first = parts[0].len() as f32 / rate as f32;
        assert!((27.0..27.25).contains(&first), "cut at {first}s");
        assert_eq!(parts.iter().map(|p| p.len()).sum::<usize>(), pcm.len());
    }
}
