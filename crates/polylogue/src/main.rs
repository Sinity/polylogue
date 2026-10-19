//! The `polylogue` binary: daemon, CLI, MCP server, hook command and
//! browser-extension host. Every read and write goes through the daemon;
//! when it is not running, commands exit with code 6 (`daemon_required`).

mod briefing;
mod browser;
mod cli;
mod continuation;
mod daemon;
mod embeddings;
mod hook;
mod live;
mod mcp;
mod measure_text;
#[cfg(feature = "sinex")]
mod sinex;
#[cfg(test)]
mod testworld;
mod web;

use clap::Parser;
use std::process::ExitCode;

/// Exit code when a command needs the daemon and it is not running.
pub const EXIT_DAEMON_REQUIRED: u8 = 6;

fn main() -> ExitCode {
    // Chrome runs a native messaging host with the calling extension's
    // origin as its only argument; the host manifest names this binary.
    if std::env::args()
        .nth(1)
        .is_some_and(|a| a.starts_with("chrome-extension://"))
    {
        return browser::native::run();
    }
    let args = cli::Args::parse();
    cli::run(args)
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    #[test]
    fn the_command_tree_is_well_formed() {
        super::cli::Args::command().debug_assert();
    }
}
