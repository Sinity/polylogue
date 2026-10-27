//! End to end: the real `polylogue` binary over a temporary archive and
//! home directory.
//!
//! - `cli`: the binary as a daemon, driven through the CLI;
//! - `browser_host`: the binary as Chrome's native messaging host.

mod browser_host;
mod cli;
mod onboarding;

const BIN: &str = env!("CARGO_BIN_EXE_polylogue");

/// Runtime files belong to the fixture even when a daemon is killed.
fn command(root: &std::path::Path) -> std::process::Command {
    let runtime = root.join("runtime");
    std::fs::create_dir_all(&runtime).unwrap();
    let mut command = std::process::Command::new(BIN);
    command.env("XDG_RUNTIME_DIR", runtime);
    command
}
