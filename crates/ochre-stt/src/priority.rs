//! Thread priority for the latency-critical decode path (docs/latency.md rule 1).
//!
//! Under background load, a CPU decode at Normal priority can take many times longer than at
//! Above Normal (spinning thread pools stall whenever one worker is preempted). Every thread
//! that does decode work (the decode thread and each ONNX Runtime pool worker) raises itself:
//! `THREAD_PRIORITY_ABOVE_NORMAL` on Windows, the user-interactive QoS class on macOS, and
//! nice -5 on Linux (directly with CAP_SYS_NICE or a raised `RLIMIT_NICE`, else through rtkit;
//! see `ochre_core::linux_priority`, which warns once if both fail).

/// Raise the calling thread's scheduling priority. Returns whether it took effect.
pub fn boost_current_thread() -> bool {
    imp::boost()
}

/// Physical cores (not SMT siblings), at least 1.
pub fn physical_cores() -> usize {
    num_cpus::get_physical().max(1)
}

#[cfg(windows)]
mod imp {
    use windows_sys::Win32::System::Threading::{
        GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL,
    };
    pub fn boost() -> bool {
        // SAFETY: the pseudo-handle from GetCurrentThread is always valid for the calling thread.
        unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL) != 0 }
    }
}

#[cfg(target_vendor = "apple")]
mod imp {
    pub fn boost() -> bool {
        // SAFETY: plain libc call affecting only the calling thread.
        unsafe {
            libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0)
                == 0
        }
    }
}

#[cfg(target_os = "linux")]
mod imp {
    pub fn boost() -> bool {
        use ochre_core::linux_priority::{current_tid, raise_thread};
        raise_thread(current_tid(), -5)
    }
}

#[cfg(all(unix, not(target_vendor = "apple"), not(target_os = "linux")))]
mod imp {
    pub fn boost() -> bool {
        // SAFETY: plain libc calls; `who` 0 is the calling process.
        unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, -5) == 0 }
    }
}

#[cfg(not(any(windows, unix)))]
mod imp {
    pub fn boost() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn boost_runs_and_cores_are_sane() {
        let ok = std::thread::spawn(super::boost_current_thread)
            .join()
            .unwrap();
        if cfg!(any(windows, target_vendor = "apple")) {
            assert!(ok);
        }
        assert!(super::physical_cores() >= 1);
        assert!(super::physical_cores() <= std::thread::available_parallelism().unwrap().get());
    }
}
