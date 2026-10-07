//! Windows clipboard snapshot / restore for the paste fallback (raw Win32, memory formats).
//!
//! Every format backed by global memory is saved (text, RTF, HTML, DIB images, file drops,
//! app-private formats...). GDI-handle formats (bitmaps, metafiles, palettes) and owner-display
//! formats cannot be copied as bytes and are skipped; Windows re-synthesizes CF_BITMAP from the
//! DIB we do keep. What we put on the clipboard is marked to stay out of clipboard history and
//! cloud sync.

use std::time::{Duration, Instant};

use windows::Win32::Foundation::{GlobalFree, HANDLE, HGLOBAL};
use windows::Win32::System::DataExchange::{
    CloseClipboard, EmptyClipboard, EnumClipboardFormats, GetClipboardData,
    GetClipboardSequenceNumber, OpenClipboard, RegisterClipboardFormatW, SetClipboardData,
};
use windows::Win32::System::Memory::{
    GMEM_MOVEABLE, GlobalAlloc, GlobalLock, GlobalSize, GlobalUnlock,
};
use windows::core::PCWSTR;

use super::wide;

const CF_TEXT: u32 = 1;
const CF_BITMAP: u32 = 2;
const CF_METAFILEPICT: u32 = 3;
const CF_OEMTEXT: u32 = 7;
const CF_PALETTE: u32 = 9;
const CF_UNICODETEXT: u32 = 13;
const CF_ENHMETAFILE: u32 = 14;
const CF_LOCALE: u32 = 16;
/// Not memory-backed (GDI handles, owner display): cannot be copied as bytes.
const HANDLE_FORMATS: [u32; 8] = [
    CF_BITMAP,
    CF_METAFILEPICT,
    CF_PALETTE,
    CF_ENHMETAFILE,
    0x80,
    0x82,
    0x83,
    0x8E,
];
/// Synthesized by Windows from CF_UNICODETEXT.
const SYNTHESIZED: [u32; 3] = [CF_TEXT, CF_OEMTEXT, CF_LOCALE];
const MAX_TOTAL: usize = 64 * 1024 * 1024;
const NO_HISTORY: [&str; 3] = [
    "ExcludeClipboardContentFromMonitorProcessing",
    "CanIncludeInClipboardHistory",
    "CanUploadToCloudClipboard",
];

/// Every memory-backed format that was on the clipboard.
pub type Snapshot = Vec<(u32, Vec<u8>)>;

/// The clipboard, open for the lifetime of this guard.
struct Open;

impl Open {
    fn new(timeout: Duration) -> Result<Self, String> {
        let deadline = Instant::now() + timeout;
        loop {
            // SAFETY: no owner window; closed in Drop.
            if unsafe { OpenClipboard(None) }.is_ok() {
                return Ok(Open);
            }
            if Instant::now() > deadline {
                return Err("the clipboard is held open by another app".into());
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Open {
    fn drop(&mut self) {
        // SAFETY: we opened it.
        unsafe {
            let _ = CloseClipboard();
        }
    }
}

fn read(handle: HANDLE, budget: usize) -> Option<Vec<u8>> {
    let mem = HGLOBAL(handle.0);
    // SAFETY: the handle comes from GetClipboardData while the clipboard is open.
    unsafe {
        let size = GlobalSize(mem);
        if size == 0 || size > budget {
            return None;
        }
        let p = GlobalLock(mem) as *const u8;
        if p.is_null() {
            return None;
        }
        let data = std::slice::from_raw_parts(p, size).to_vec();
        let _ = GlobalUnlock(mem);
        Some(data)
    }
}

fn put(format: u32, data: &[u8]) -> bool {
    // SAFETY: we allocate, fill and hand over the memory; it is freed only if the handover fails.
    unsafe {
        let Ok(mem) = GlobalAlloc(GMEM_MOVEABLE, data.len().max(1)) else {
            return false;
        };
        let p = GlobalLock(mem) as *mut u8;
        if p.is_null() {
            let _ = GlobalFree(Some(mem));
            return false;
        }
        std::ptr::copy_nonoverlapping(data.as_ptr(), p, data.len());
        let _ = GlobalUnlock(mem);
        if SetClipboardData(format, Some(HANDLE(mem.0))).is_err() {
            let _ = GlobalFree(Some(mem)); // ownership only passes on success
            return false;
        }
        true
    }
}

fn mark_private() {
    for name in NO_HISTORY {
        let w = wide(name);
        // SAFETY: NUL-terminated wide string.
        let format = unsafe { RegisterClipboardFormatW(PCWSTR(w.as_ptr())) };
        if format != 0 {
            put(format, &0u32.to_le_bytes());
        }
    }
}

pub fn sequence() -> u32 {
    // SAFETY: no arguments.
    unsafe { GetClipboardSequenceNumber() }
}

pub fn save() -> Result<Snapshot, String> {
    let _open = Open::new(Duration::from_millis(500))?;
    let mut out = Vec::new();
    let mut budget = MAX_TOTAL;
    let mut format = 0u32;
    loop {
        // SAFETY: the clipboard is open.
        format = unsafe { EnumClipboardFormats(format) };
        if format == 0 {
            break;
        }
        if HANDLE_FORMATS.contains(&format) || SYNTHESIZED.contains(&format) {
            continue;
        }
        // SAFETY: the clipboard is open.
        if let Ok(h) = unsafe { GetClipboardData(format) }
            && let Some(data) = read(h, budget)
        {
            budget -= data.len();
            out.push((format, data));
        }
    }
    Ok(out)
}

pub fn restore(snapshot: &Snapshot) -> Result<(), String> {
    let _open = Open::new(Duration::from_millis(500))?;
    // SAFETY: the clipboard is open.
    unsafe { EmptyClipboard() }.map_err(|e| e.to_string())?;
    for (format, data) in snapshot {
        put(*format, data);
    }
    if !snapshot.is_empty() {
        mark_private(); // the restore itself should not show up as a new history entry
    }
    Ok(())
}

/// Put `text` on the clipboard (kept out of history); returns the new sequence number.
pub fn set_text(text: &str) -> Result<u32, String> {
    {
        let _open = Open::new(Duration::from_millis(500))?;
        // SAFETY: the clipboard is open.
        unsafe { EmptyClipboard() }.map_err(|e| e.to_string())?;
        let bytes: Vec<u8> = wide(text).iter().flat_map(|u| u.to_le_bytes()).collect();
        if !put(CF_UNICODETEXT, &bytes) {
            return Err("SetClipboardData(CF_UNICODETEXT) failed".into());
        }
        mark_private();
    }
    Ok(sequence())
}

pub fn get_text() -> Option<String> {
    let _open = Open::new(Duration::from_millis(500)).ok()?;
    // SAFETY: the clipboard is open.
    let h = unsafe { GetClipboardData(CF_UNICODETEXT) }.ok()?;
    let data = read(h, MAX_TOTAL)?;
    let units: Vec<u16> = data
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let end = units.iter().position(|u| *u == 0).unwrap_or(units.len());
    Some(String::from_utf16_lossy(&units[..end]))
}
