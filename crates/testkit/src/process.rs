//! Child daemons of tests and of `cargo xtask run`: they die with their parent.

use std::io;
use std::process::{Child, Command};

/// Makes `command`'s process die with the calling thread: on Linux, a forced
/// exit of the parent, which runs no cleanup, also kills the child. Normal
/// shutdown still belongs to the caller.
pub fn die_with_parent(command: &mut Command) -> &mut Command {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        let parent = std::process::id() as libc::pid_t;
        // SAFETY: the post-fork callback only calls these Linux syscalls and
        // constructs allocation-free OS errors. It takes no inherited locks.
        // The PID check closes the race where the parent dies before prctl.
        #[allow(unsafe_code)]
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) == -1 {
                    return Err(io::Error::last_os_error());
                }
                if libc::getppid() != parent {
                    return Err(io::Error::from_raw_os_error(libc::ECHILD));
                }
                Ok(())
            });
        }
    }
    command
}

/// Spawns a test daemon that dies with the test ([`die_with_parent`]);
/// normal shutdown still belongs to the caller's kill-and-wait guard.
pub fn spawn_daemon(command: &mut Command) -> io::Result<Child> {
    die_with_parent(command).spawn()
}

/// A process-exit watch opened while the process identity is known to be live.
#[cfg(target_os = "linux")]
pub struct ProcessExit(std::os::fd::OwnedFd);

#[cfg(target_os = "linux")]
impl ProcessExit {
    #[allow(unsafe_code)]
    pub fn watch(pid: u32) -> io::Result<Self> {
        use std::os::fd::FromRawFd;
        // SAFETY: pidfd_open takes scalar arguments and returns a new owned fd.
        let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid as libc::pid_t, 0u32) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: this successful syscall returned a fresh, uniquely owned fd.
        Ok(Self(unsafe {
            std::os::fd::OwnedFd::from_raw_fd(fd as i32)
        }))
    }

    /// Wait for the whole process, including its other threads, to exit.
    /// The test runner owns the deadline for a hung test.
    #[allow(unsafe_code)]
    pub fn wait(self) -> io::Result<()> {
        use std::os::fd::AsRawFd;
        let mut event = libc::pollfd {
            fd: self.0.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        loop {
            // SAFETY: event is a writable pollfd and its owned fd remains open.
            let ready = unsafe { libc::poll(&mut event, 1, -1) };
            if ready < 0 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(error);
            }
            if event.revents & (libc::POLLIN | libc::POLLHUP) != 0 {
                return Ok(());
            }
            return Err(io::Error::other(
                "process-exit watch returned unexpected readiness",
            ));
        }
    }
}
