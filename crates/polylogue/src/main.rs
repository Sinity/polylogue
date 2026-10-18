//! The `polylogue` binary: daemon, CLI, MCP server, hook command and
//! browser-extension host. Every read and write goes through the daemon;
//! when it is not running, commands exit with code 6 (`daemon_required`).

mod briefing;
mod cli;
mod continuation;
mod daemon;
mod embeddings;
mod hook;
mod live;
mod mcp;
mod pricing;
mod sinex;

use clap::Parser;
use std::process::ExitCode;

/// Exit code when a command needs the daemon and it is not running.
pub const EXIT_DAEMON_REQUIRED: u8 = 6;

fn main() -> ExitCode {
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
