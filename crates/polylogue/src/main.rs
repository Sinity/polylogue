//! The `polylogue` binary: daemon, CLI, MCP server, hook command and
//! browser-extension host. Archive reads and writes go through the daemon;
//! schema diff compares explicit local artifacts. Commands
//! requiring a stopped daemon exit with code 6 (`daemon_required`).

mod advisory;
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
mod hosting;
mod live_acts;
mod live_evidence;
mod mailbox;
mod manual;
mod mcp;
mod operation;
mod protocol;
mod resume;
mod row_numbers;
mod runtime_dir;
mod signals;
#[cfg(test)]
mod test_answers;
#[cfg(test)]
mod test_daemon;
mod web;

use std::process::ExitCode;

/// mimalloc instead of glibc's allocator: a first build's sixteen parse
/// threads fragmented glibc's arenas by about 1 GB beyond the live heap
/// (heaptrack against RSS, 2026-10-19); mimalloc kept RSS 650 MB lower at
/// the same speed. Built without transparent huge pages (`no_thp`): marking
/// its arena for them cost every short-lived process about 2 ms at the
/// median and 15–30 ms at p95 (hook and `--version`), and the daemon parsed
/// a 457 MB first build faster without them (2026-11-09).
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
    // A harness treats a non-zero hook exit as a refusal of its tool call, so
    // a hook that cannot start says why on stderr and exits 0.
    let hook = hook_invocation(&std::env::args_os().collect::<Vec<_>>());
    let process = match config::ProcessInputs::capture() {
        Ok(process) => process,
        Err(error) => {
            return startup_error(
                hook,
                error,
                config::OutputConfig::new(false, false, true, None),
            );
        }
    };
    let arguments = std::env::args_os().collect::<Vec<_>>();
    if process.environment.get_os("COMPLETE").is_some() {
        client_pipe_signal();
        let separator = arguments.iter().position(|value| value == "--");
        let callback = separator
            .map(|separator| arguments[separator + 1..].to_vec())
            .unwrap_or_default();
        let index = process
            .environment
            .get_os("_CLAP_COMPLETE_INDEX")
            .and_then(|value| value.to_str())
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| callback.len().saturating_sub(1));
        let mut invocation = arguments[..separator.unwrap_or(arguments.len())].to_vec();
        invocation.extend(callback.iter().skip(1).cloned());
        let overrides = completion_overrides(&invocation);
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
        let basename = arguments
            .first()
            .and_then(|argument| std::path::Path::new(argument).file_name())
            .unwrap_or(std::ffi::OsStr::new("polylogue"));
        let bin_name = basename.to_string_lossy().into_owned();
        // Shell function names must distinguish '-' from '_' and other bytes.
        let name = format!(
            "polylogue_{}",
            basename
                .as_encoded_bytes()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        );
        clap_complete::CompleteEnv::with_factory(move || {
            cli::command().name(name.clone()).bin_name(bin_name.clone())
        })
        .complete();
    }
    let mut matches = match cli::command().try_get_matches_from(arguments.clone()) {
        Ok(matches) => matches,
        Err(error) => {
            client_pipe_signal();
            if hook {
                return hook_startup_error(error);
            }
            return cli::parse_error(error, process.output);
        }
    };
    let native = native_host(&matches);
    if !daemon_command(&matches) {
        client_pipe_signal();
    }
    if let Some(result) = cli::run_local(&matches, process.output) {
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
            Err(error) => return startup_error(hook, error, process.output),
        };
        let text = arguments
            .iter()
            .map(|value| value.to_str().map(str::to_owned))
            .collect::<Option<Vec<_>>>();
        if let Some(text) = text {
            matches = match row_numbers::rewrite_args(text, &archive) {
                Ok(arguments) => match cli::command().try_get_matches_from(arguments) {
                    Ok(matches) => matches,
                    Err(error) if hook => return hook_startup_error(error),
                    Err(error) => return cli::parse_error(error, process.output),
                },
                Err(error) => return startup_error(hook, error, process.output),
            };
        }
    }
    let mut startup = match config::Startup::load(&process, overrides) {
        Ok(startup) => startup,
        Err(error) => return startup_error(hook, error, process.output),
    };
    protocol::caller::initialize(&process.environment);
    if native {
        return browser::native::run(
            &startup.engine,
            &polylogue_store::Archive::at(&startup.config.archive),
        );
    }
    let Some(cwd) = process.cwd.as_deref() else {
        return startup_error(hook, "the working directory is unavailable", process.output);
    };
    if cli::needs_engine(&matches, &startup.mode) {
        startup.stopped_attempt = match executable_attempt() {
            Ok(attempt) => Some(attempt),
            Err(error) => return startup_error(hook, error, process.output),
        };
    }
    cli::run(&matches, &startup, cwd)
}

/// The process boundary captures the running image, rather than a build label.
fn executable_attempt() -> std::io::Result<polylogue_model::batch::StoppedAttempt> {
    // Opening the process image keeps the running executable even after an upgrade
    // replaces its former path.
    #[cfg(target_os = "linux")]
    let file = std::fs::File::open("/proc/self/exe")?;
    #[cfg(not(target_os = "linux"))]
    let file = std::fs::File::open(std::env::current_exe()?)?;
    attempt_from_executable(file)
}

fn attempt_from_executable(
    file: std::fs::File,
) -> std::io::Result<polylogue_model::batch::StoppedAttempt> {
    let mut digest = blake3::Hasher::new();
    digest.update_reader(file)?;
    Ok(polylogue_model::batch::StoppedAttempt {
        executable: polylogue_model::BlobHash(*digest.finalize().as_bytes()),
        // Native parser selection currently has no configurable parser settings.
        parser_settings: None,
    })
}

/// The daemon and `polylogue host` write to peers that may go away, so a
/// closed pipe is an error to them, never a reason to die.
fn daemon_command(matches: &clap::ArgMatches) -> bool {
    matches!(matches.subcommand_name(), Some("daemon" | "host"))
}

fn native_host(matches: &clap::ArgMatches) -> bool {
    if matches.subcommand().is_some() {
        return false;
    }
    let Some(mut query) = matches.get_many::<String>("query") else {
        return false;
    };
    query
        .next()
        .is_some_and(|word| word.starts_with("chrome-extension://"))
        && query.next().is_none()
}

fn client_pipe_signal() {
    // SAFETY: main selects the process mode before starting worker threads.
    #[allow(unsafe_code)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

fn startup_error(
    hook: bool,
    error: impl std::fmt::Display,
    output: config::OutputConfig,
) -> ExitCode {
    if hook {
        return hook_startup_error(error);
    }
    eprint!(
        "{}",
        cli::print::diagnostic("", &error.to_string(), None, output.stderr_color())
    );
    ExitCode::from(2)
}

fn hook_startup_error(error: impl std::fmt::Display) -> ExitCode {
    let text = error.to_string();
    let line = text
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("");
    eprintln!("polylogue hook: {}", line.trim_start_matches("error: "));
    ExitCode::SUCCESS
}

/// Whether the command line names the `hook` command: its first word that is
/// neither a global option nor that option's value.
fn hook_invocation(arguments: &[std::ffi::OsString]) -> bool {
    const VALUED: [&str; 8] = [
        "--deadline",
        "--timeout",
        "--archive",
        "--config",
        "--listen",
        "--auth-token-file",
        "--request",
        "--request-id",
    ];
    let mut words = arguments.iter().skip(1).filter_map(|word| word.to_str());
    while let Some(word) = words.next() {
        if VALUED.contains(&word) {
            words.next();
        } else if !word.starts_with('-') {
            return word == "hook";
        }
    }
    false
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
