//! A real child CLI with a pseudoterminal stdout, without a controlling terminal.
use super::*;
use rustix::pty::{OpenptFlags, grantpt, ioctl_tiocgptpeer, openpt, unlockpt};
use std::fs::File;
use std::process::ExitStatus;

pub(super) fn cli(
    env: &[(&str, &Path)],
    args: &[&str],
    harness: bool,
) -> (ExitStatus, Vec<u8>, Vec<u8>) {
    let flags = OpenptFlags::RDWR | OpenptFlags::NOCTTY | OpenptFlags::CLOEXEC;
    let master = openpt(flags).unwrap();
    grantpt(&master).unwrap();
    unlockpt(&master).unwrap();
    let slave = ioctl_tiocgptpeer(&master, flags).unwrap();
    let mut command = Command::new(crate::BIN);
    for (key, value) in env {
        command.env(key, value);
    }
    for key in [
        "CLAUDECODE",
        "CLAUDE_CODE_SESSION_ID",
        "CODEX_THREAD_ID",
        "CODEX_SESSION_ID",
        "GEMINI_CLI",
        "AGENTCTL_JOB_ID",
        "AGENTCTL_RUN_ID",
        "POLYLOGUE_JOB_ID",
    ] {
        command.env_remove(key);
    }
    if harness {
        command.env("CODEX_SESSION_ID", "synthetic-harness");
    }
    let child = command
        .args(args)
        .stdin(Stdio::null())
        .stdout(File::from(slave))
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(command);
    let output = child.wait_with_output().unwrap();
    let mut master = File::from(master);
    let mut stdout = Vec::new();
    if let Err(error) = master.read_to_end(&mut stdout) {
        assert_eq!(error.raw_os_error(), Some(5), "PTY read failed: {error}");
    }
    (output.status, stdout, output.stderr)
}
