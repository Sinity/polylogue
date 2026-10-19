//! The `polylogue` binary: daemon, CLI, MCP server, hook command and
//! browser-extension host. Every read and write goes through the daemon;
//! when it is not running, commands exit with code 6 (`daemon_required`).

mod briefing;
mod browser;
mod cli;
mod commands;
mod daemon;
mod embeddings;
mod hook;
mod mcp;
mod resume;
mod row_numbers;
mod signals;
#[cfg(feature = "sinex")]
mod sinex;
#[cfg(test)]
mod test_daemon;
mod web;

use std::process::ExitCode;

/// mimalloc instead of glibc's allocator: a first build's sixteen parse
/// threads fragmented glibc's arenas by about 1 GB beyond the live heap
/// (heaptrack against RSS, 2026-10-19); mimalloc kept RSS 650 MB lower at
/// the same speed.
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

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
    // `@N` names row N of the last list printed in this terminal.
    let argv: Option<Vec<String>> = std::env::args_os().map(|a| a.into_string().ok()).collect();
    let matches = match argv {
        Some(argv) if argv.iter().any(|a| a.contains('@')) => {
            let root = polylogue_store::Archive::default_root();
            match row_numbers::rewrite_args(argv, &root) {
                Ok(argv) => cli::command().get_matches_from(argv),
                Err(e) => {
                    eprintln!("polylogue: {e}");
                    return ExitCode::from(2);
                }
            }
        }
        _ => cli::command().get_matches(),
    };
    cli::run(&matches)
}

#[cfg(test)]
mod tests;
