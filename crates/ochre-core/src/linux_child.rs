//! Linux: child processes that die with us.
//!
//! `PR_SET_PDEATHSIG` fires when the *thread* that forked the child exits, not the process.
//! Spawning from a short-lived thread (an engine loader, a restart helper) therefore kills the
//! child moments after it starts. Every tied child is forked from one spawner thread that lives as
//! long as the process, so the signal comes only when Ochre itself exits or is killed.

use std::io;
use std::process::{Child, Command};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Mutex, OnceLock};

type Job = (Command, Sender<io::Result<Child>>);

fn spawner() -> &'static Mutex<Sender<Job>> {
    static TX: OnceLock<Mutex<Sender<Job>>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = channel::<Job>();
        std::thread::Builder::new()
            .name("ochre-spawner".into())
            .spawn(move || {
                for (mut cmd, reply) in rx {
                    let _ = reply.send(spawn_here(&mut cmd));
                }
            })
            .expect("spawn the child-process spawner");
        Mutex::new(tx)
    })
}

fn spawn_here(cmd: &mut Command) -> io::Result<Child> {
    use std::os::unix::process::CommandExt;
    // SAFETY: prctl is async-signal-safe; it only asks for SIGTERM when our spawner thread dies,
    // which happens only when the process exits.
    unsafe {
        cmd.pre_exec(|| {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
            Ok(())
        });
    }
    cmd.spawn()
}

/// Spawn `cmd` so that it gets SIGTERM when this process exits, however that happens.
pub fn spawn_tied(cmd: Command) -> io::Result<Child> {
    let (reply_tx, reply_rx) = channel();
    spawner()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .send((cmd, reply_tx))
        .map_err(|_| io::Error::other("spawner thread gone"))?;
    reply_rx
        .recv()
        .map_err(|_| io::Error::other("spawner thread gone"))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_outlives_the_thread_that_asked_for_it() {
        let child = std::thread::spawn(|| {
            let mut cmd = Command::new("sleep");
            cmd.arg("30");
            spawn_tied(cmd).unwrap()
        })
        .join()
        .unwrap();
        // The requesting thread is gone; with a plain pre_exec PDEATHSIG the child would be dying.
        std::thread::sleep(std::time::Duration::from_millis(300));
        let mut child = child;
        assert!(child.try_wait().unwrap().is_none(), "child was killed");
        child.kill().unwrap();
        child.wait().unwrap();
    }
}
