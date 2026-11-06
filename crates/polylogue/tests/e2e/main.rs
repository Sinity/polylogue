//! End to end: the real `polylogue` binary over a temporary archive and
//! home directory.
//!
//! - `cli`: the binary as a daemon, driven through the CLI;
//! - `browser_host`: the binary as Chrome's native messaging host;
//! - `completion`: shell callbacks over an offline synthetic catalog.

mod browser_host;
mod cli;
mod completion;
mod onboarding;

const BIN: &str = env!("CARGO_BIN_EXE_polylogue");

mod process;
use process::Fixture;

fn command(root: &std::path::Path) -> std::process::Command {
    Fixture::new(
        &root.join("archive"),
        &root.join("home"),
        &root.join("config"),
    )
    .command()
}

mod deadlines;

fn test_process(root: &std::path::Path) -> std::process::Command {
    let fixture = Fixture::new(
        &root.join("archive"),
        &root.join("home"),
        &root.join("config"),
    );
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    fixture.environment(&mut command);
    command
}
