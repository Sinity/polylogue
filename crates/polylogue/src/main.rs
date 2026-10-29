//! The `polylogue` binary: daemon, CLI, MCP server, hook command and
//! browser-extension host. Archive reads and writes go through the daemon;
//! schema diff compares explicit local artifacts. Commands
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
    let process = match config::ProcessInputs::capture() {
        Ok(process) => process,
        Err(error) => return startup_error(error),
    };
    let arguments = std::env::args_os().collect::<Vec<_>>();
    let native = arguments
        .get(1)
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.starts_with("chrome-extension://"));
    if arguments.get(1).and_then(|value| value.to_str()) != Some("daemon") {
        // SAFETY: this is the process boundary, before worker threads exist.
        #[allow(unsafe_code)]
        unsafe {
            libc::signal(libc::SIGPIPE, libc::SIG_DFL);
        }
    }
    if process.environment.get_os("COMPLETE").is_some() {
        let callback = arguments
            .iter()
            .position(|value| value == "--")
            .map(|separator| arguments[separator + 1..].to_vec())
            .unwrap_or_default();
        let index = process
            .environment
            .get_os("_CLAP_COMPLETE_INDEX")
            .and_then(|value| value.to_str())
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| callback.len().saturating_sub(1));
        let overrides = completion_overrides(&callback);
        let catalog = config::archive_location(&process, &overrides)
            .ok()
            .and_then(|archive| {
                let paths = config::Paths::resolve(&process.environment).ok()?;
                Some(cli::completion::Catalog {
                    path: paths.completion_catalog(&archive),
                    archive_key: runtime_dir::archive_key(&archive),
                })
            });
        cli::completion::capture(cli::completion::Context {
            arguments: callback,
            index,
            catalog,
            prefix: process
                .environment
                .get_os("_POLYLOGUE_COMPLETE_PREFIX")
                .and_then(|value| value.to_str())
                .map(str::to_owned),
        });
        clap_complete::CompleteEnv::with_factory(cli::command).complete();
    }
    let mut matches = if native {
        cli::command().get_matches_from([arguments.first().cloned().unwrap_or_default()])
    } else {
        cli::command().get_matches_from(arguments.clone())
    };
    if let Some(result) = cli::run_local(&matches) {
        return result;
    }
    let overrides = config::Overrides::from_matches(&matches);
    if !native
        && arguments
            .iter()
            .any(|value| value.to_str().is_some_and(|value| value.contains('@')))
    {
        let archive = match config::archive_location(&process, &overrides) {
            Ok(archive) => archive,
            Err(error) => return startup_error(error),
        };
        let text = arguments
            .iter()
            .map(|value| value.to_str().map(str::to_owned))
            .collect::<Option<Vec<_>>>();
        if let Some(text) = text {
            matches = match row_numbers::rewrite_args(text, &archive) {
                Ok(arguments) => cli::command().get_matches_from(arguments),
                Err(error) => return startup_error(error),
            };
        }
    }
    let startup = match config::Startup::load(&process, overrides) {
        Ok(startup) => startup,
        Err(error) => return startup_error(error),
    };
    protocol::caller::initialize(&process.environment);
    if native {
        return browser::native::run(
            &startup.engine,
            &polylogue_store::Archive::at(&startup.config.archive),
        );
    }
    let Some(cwd) = process.cwd.as_deref() else {
        return startup_error("the working directory is unavailable");
    };
    cli::run(&matches, &startup, cwd)
}

fn startup_error(error: impl std::fmt::Display) -> ExitCode {
    eprintln!(
        "{}",
        polylogue_render::terminal::escape(&format!("polylogue: {error}"))
    );
    ExitCode::from(2)
}

/// Completion resolves deployment flags without parsing unfinished query words.
fn completion_overrides(arguments: &[std::ffi::OsString]) -> config::Overrides {
    let mut flags = vec![std::ffi::OsString::from("polylogue")];
    let mut arguments = arguments.iter().skip(1);
    while let Some(argument) = arguments.next() {
        let Some(text) = argument.to_str() else {
            continue;
        };
        if ["--archive", "--config", "--listen", "--auth-token-file"].contains(&text) {
            flags.push(argument.clone());
            if let Some(value) = arguments.next() {
                flags.push(value.clone());
            }
        } else if ["--archive=", "--config=", "--listen=", "--auth-token-file="]
            .iter()
            .any(|prefix| text.starts_with(prefix))
        {
            flags.push(argument.clone());
        }
    }
    config::augment_command(clap::Command::new("polylogue"))
        .try_get_matches_from(flags)
        .map(|matches| config::Overrides::from_matches(&matches))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests;
