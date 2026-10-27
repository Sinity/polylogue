//! End to end: the real `polylogue` binary over a temporary archive and
//! home directory.
//!
//! - `cli`: the binary as a daemon, driven through the CLI;
//! - `browser_host`: the binary as Chrome's native messaging host.

mod browser_host;
mod cli;

const BIN: &str = env!("CARGO_BIN_EXE_polylogue");

mod onboarding;

mod deadlines;
