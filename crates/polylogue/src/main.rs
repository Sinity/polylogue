//! The `polylogue` binary: daemon, CLI, MCP server, hook command and
//! browser-extension host. Interactive reads and writes go through the daemon;
//! offline schema commands read explicit stored revisions directly. Commands
//! requiring a stopped daemon exit with code 6 (`daemon_required`).

mod briefing;
mod browser;
mod cli;
mod client;
mod commands;
mod completion_catalog;
mod config;
mod daemon;
mod demo;
mod embeddings;
mod explain;
mod hook;
mod manual;
mod mcp;
mod protocol;
mod resume;
mod row_numbers;
mod runtime_dir;
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

/// Exit code when the daemon refuses an archive newer than this build or one
/// it has no migration from (sysexits' `EX_CONFIG`). Starting again cannot
/// help, so a service manager should not restart on it (systemd:
/// `RestartPreventExitStatus=78`).
pub const EXIT_SCHEMA_REFUSED: u8 = 78;

fn main() -> ExitCode {
    protocol::caller::initialize();
    // Chrome runs a native messaging host with the calling extension's
    // origin as its only argument; the host manifest names this binary.
    if std::env::args()
        .nth(1)
        .is_some_and(|a| a.starts_with("chrome-extension://"))
    {
        return browser::native::run();
    }
    // A command whose reader stops reading (`polylogue … | head`) ends
    // quietly, as other command-line tools do, instead of failing on the
    // closed pipe. The daemon keeps ignoring it: a client that goes away
    // must not stop it.
    if std::env::args().nth(1).as_deref() != Some("daemon") {
        // SAFETY: restoring the default disposition of one signal before any
        // other thread exists.
        #[allow(unsafe_code)]
        unsafe {
            libc::signal(libc::SIGPIPE, libc::SIG_DFL);
        }
    }
    // `COMPLETE=<shell>` set: the shell asks for completions.
    cli::completion::capture();
    clap_complete::CompleteEnv::with_factory(cli::command).complete();
    // `@N` names row N of the last list printed in this terminal.
    let argv: Option<Vec<String>> = std::env::args_os().map(|a| a.into_string().ok()).collect();
    let matches = match argv {
        Some(argv) if argv.iter().any(|a| a.contains('@')) => {
            let root = polylogue_store::Archive::default_root();
            match row_numbers::rewrite_args(argv, &root) {
                Ok(argv) => cli::command().get_matches_from(argv),
                Err(e) => {
                    eprintln!(
                        "{}",
                        polylogue_render::terminal::escape(&format!("polylogue: {e}"))
                    );
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
