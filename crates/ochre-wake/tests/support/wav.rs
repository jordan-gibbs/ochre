//! Minimal RIFF/WAVE reader for tests and the example: PCM 16-bit (any channel count, mixed to
//! mono) or 32-bit float. Returns `(sample_rate, mono samples as int16-scale i16)`.

use std::path::Path;

pub fn read_wav(path: &Path) -> std::io::Result<(u32, Vec<i16>)> {
    let bytes = std::fs::read(path)?;
    let bad = |m: &str| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{}: {m}", path.display()),
        )
    };
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad("not a RIFF/WAVE file"));
    }
    let (mut fmt, mut data) = (None, None);
    let mut p = 12;
    while p + 8 <= bytes.len() {
        let id = &bytes[p..p + 4];
        let len = u32::from_le_bytes(bytes[p + 4..p + 8].try_into().unwrap()) as usize;
        let body = &bytes[p + 8..(p + 8 + len).min(bytes.len())];
        match id {
            b"fmt " => fmt = Some(body),
            b"data" => data = Some(body),
            _ => {}
        }
        p += 8 + len + (len & 1);
    }
    let fmt = fmt.ok_or_else(|| bad("no fmt chunk"))?;
    let data = data.ok_or_else(|| bad("no data chunk"))?;
    let tag = u16::from_le_bytes([fmt[0], fmt[1]]);
    let ch = u16::from_le_bytes([fmt[2], fmt[3]]).max(1) as usize;
    let rate = u32::from_le_bytes(fmt[4..8].try_into().unwrap());
    let bits = u16::from_le_bytes([fmt[14], fmt[15]]);
    let frames: Vec<f32> = match (tag, bits) {
        (1, 16) | (0xFFFE, 16) => data
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32)
            .collect(),
        (3, 32) | (0xFFFE, 32) => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * 32767.0)
            .collect(),
        _ => return Err(bad(&format!("unsupported format tag {tag} / {bits} bits"))),
    };
    let mono = frames
        .chunks_exact(ch)
        .map(|c| {
            (c.iter().sum::<f32>() / ch as f32)
                .round()
                .clamp(-32768.0, 32767.0) as i16
        })
        .collect();
    Ok((rate, mono))
}

/// Linear resampler to 16 kHz (good enough for a demo; real capture resamples in ochre-audio).
#[allow(dead_code)]
pub fn to_16k(rate: u32, x: &[i16]) -> Vec<i16> {
    if rate == 16_000 || x.is_empty() {
        return x.to_vec();
    }
    let n = (x.len() as u64 * 16_000 / rate as u64) as usize;
    (0..n)
        .map(|i| {
            let pos = i as f64 * rate as f64 / 16_000.0;
            let j = pos as usize;
            let f = pos - j as f64;
            let a = x[j.min(x.len() - 1)] as f64;
            let b = x[(j + 1).min(x.len() - 1)] as f64;
            (a + (b - a) * f).round() as i16
        })
        .collect()
}
