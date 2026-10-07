//! Linux: raise a thread's nice level for the latency-critical decode path.
//!
//! `setpriority` below 0 needs CAP_SYS_NICE or a raised `RLIMIT_NICE`, and desktop sessions have
//! neither (Ubuntu's default `ulimit -e` is 0). RealtimeKit (`rtkit-daemon`, shipped with
//! PipeWire / PulseAudio on every mainstream desktop) hands out negative nice levels to
//! unprivileged threads over the system bus, so it is the fallback. If both fail the thread keeps
//! its normal priority and one warning is logged per process.

use std::sync::{Mutex, Once};

const RTKIT: &str = "org.freedesktop.RealtimeKit1";
const RTKIT_PATH: &str = "/org/freedesktop/RealtimeKit1";

static BUS: Mutex<Option<zbus::blocking::Connection>> = Mutex::new(None);
static WARNED: Once = Once::new();

/// The calling thread's kernel id.
pub fn current_tid() -> u64 {
    // SAFETY: gettid has no failure mode.
    unsafe { libc::syscall(libc::SYS_gettid) as u64 }
}

/// Set thread `tid` (of this process) to `nice` (negative = higher priority). Tries
/// `setpriority` first, then rtkit. Returns whether either took effect.
pub fn raise_thread(tid: u64, nice: i32) -> bool {
    // SAFETY: plain libc call; on Linux a thread id is a valid `who` for PRIO_PROCESS.
    if unsafe { libc::setpriority(libc::PRIO_PROCESS, tid as libc::id_t, nice) } == 0 {
        return true;
    }
    match rtkit(tid, nice) {
        Ok(()) => true,
        Err(e) => {
            WARNED.call_once(|| {
                tracing::warn!(
                    "cannot raise decode thread priority (no CAP_SYS_NICE, rtkit: {e}); \
                     dictation may be slower while the machine is busy"
                )
            });
            false
        }
    }
}

fn rtkit(tid: u64, nice: i32) -> zbus::Result<()> {
    let mut bus = BUS.lock().unwrap_or_else(|e| e.into_inner());
    if bus.is_none() {
        *bus = Some(zbus::blocking::Connection::system()?);
    }
    let conn = bus.as_ref().expect("just set");
    conn.call_method(
        Some(RTKIT),
        RTKIT_PATH,
        Some(RTKIT),
        "MakeThreadHighPriority",
        &(tid, nice),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn raise_runs() {
        // May or may not take effect (CI has no rtkit); it must not panic or hang.
        let ok = std::thread::spawn(|| super::raise_thread(super::current_tid(), -5))
            .join()
            .unwrap();
        println!("raised: {ok}");
    }
}
