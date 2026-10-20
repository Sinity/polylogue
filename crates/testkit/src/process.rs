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
