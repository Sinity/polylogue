//! Direct daemon children in tests. Callers retain their kill-and-wait guard.

use std::io;
use std::process::{Child, Command};

/// Spawns a test daemon. On Linux, a forced parent exit also kills the child;
/// ordinary shutdown still belongs to the caller's kill-and-wait guard.
pub fn spawn_daemon(command: &mut Command) -> io::Result<Child> {
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
    command.spawn()
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
    /// The test runner retains responsibility for the hung-test deadline.
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
