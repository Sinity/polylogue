//! The `polylogue` binary: daemon, CLI, MCP server, hook command and
//! browser-extension host. Archive reads and writes go through the daemon;
//! schema diff compares explicit local artifacts. A command that needs
//! the daemon exits with code 6 when it is not running.

mod advisory;
mod brief;
mod browser;
mod cli;
pub use cli::picker;
mod auth;
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
mod installation;
mod live_acts;
mod live_events;
mod mailbox;
mod manual;
mod mcp;
mod operation;
mod protocol;
mod resume;
mod row_numbers;
mod runtime_dir;
mod signals;
mod socket;
#[cfg(test)]
mod test_answers;
#[cfg(test)]
mod test_daemon;
mod web;

use std::process::ExitCode;

/// mimalloc instead of glibc's allocator: it fragments less across parse
/// threads. It also serves `malloc` (`override`), so SQLite's page caches
/// come from it: memory a cache gives back is returned to the system, where
/// glibc kept it, and a build's resident memory follows the caches it sizes.
/// Built without transparent huge pages (`no_thp`): marking its arena for
/// them slows every short-lived process (hook, `--version`).
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Exit code when the daemon refuses an archive that is not in this build's
/// format (sysexits' `EX_CONFIG`). Starting again cannot
/// help, so a service manager should not restart on it (systemd:
/// `RestartPreventExitStatus=78`).
pub const EXIT_SCHEMA_REFUSED: u8 = 78;

/// Exit code when the disk failed a sync of the register (sysexits'
/// `EX_IOERR`). Writing stops for the rest of the boot, so a service manager
/// should not restart on it (systemd: `RestartPreventExitStatus=74`).
pub const EXIT_STORAGE_FAILED: u8 = 74;

fn main() -> ExitCode {
    // A panic no `caught` takes is one error record naming its thread, with
    // its backtrace on the journal lines after it.
    polylogue_model::install_panic_hook(|thread, panic, backtrace| {
        polylogue_store::log!(error, "thread {thread} {panic}");
        for line in backtrace.lines() {
            polylogue_store::log!(progress, "  {line}");
        }
    });
    let timing = std::env::var_os("POLYLOGUE_TIMING_WORKERS").is_some();
    polylogue_store::timing::set_workers_enabled(timing);
    let result = run();
    if timing {
        eprintln!(
            "{}",
            serde_json::json!({
                "pipeline_timings": polylogue_store::timing::stage_seconds().into_iter()
                    .map(|(stage, seconds)| (stage.name(), seconds))
                    .collect::<std::collections::BTreeMap<_, _>>(),
                "workers": polylogue_store::timing::workers(),
                "sqlite_memory_peak_bytes": polylogue_store::timing::sqlite_memory_peak(),
            })
        );
        // C codecs and SQLite use libc rather than Rust's global allocator.
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        #[allow(unsafe_code)]
        // SAFETY: libc synchronizes its own allocator statistics and takes no pointers.
        unsafe {
            libc::malloc_stats();
        }
    }
    result
}

fn run() -> ExitCode {
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
            polylogue_model::hex(basename.as_encoded_bytes())
        );
        clap_complete::CompleteEnv::with_factory(move || {
            cli::command().name(name.clone()).bin_name(bin_name.clone())
        })
        .complete();
    }
    let (overrides, words) = match cli::run_line::deployment_arguments(&arguments[1..]) {
        Ok(arguments) => arguments,
        Err(error) => return startup_error(hook, error, process.output),
    };
    let native = cli::run_line::native_host(&words);
    let empty_catalog = cli::line::FirstWordCatalog::default();
    let preliminary = cli::line::parse(&words, &empty_catalog, cli::run_line::terminal());
    let catalog = if preliminary
        .as_ref()
        .is_ok_and(|parsed| parsed.first == cli::line::FirstWord::Command)
        || words.as_slice() == ["--version"]
        || native
    {
        empty_catalog
    } else {
        let paths = match config::Paths::resolve(&process.environment) {
            Ok(paths) => paths,
            Err(error) => return startup_error(hook, error, process.output),
        };
        let archive = match config::archive_location(&process, &overrides) {
            Ok(archive) => archive,
            Err(error) => return startup_error(hook, error, process.output),
        };
        cli::line::FirstWordCatalog::load(&paths, &archive)
    };
    let parsed = match cli::line::parse(&words, &catalog, cli::run_line::terminal()) {
        Ok(parsed) => parsed,
        Err(error) => {
            client_pipe_signal();
            let mut failure = crate::protocol::failure(error.code, error.message);
            failure.hint = error.try_line.map(|line| format!("try: {line}"));
            if !error.candidates.is_empty() {
                failure
                    .hint
                    .get_or_insert_default()
                    .push_str(&format!("\n{}", error.candidates.join("\n")));
            }
            return cli::fail_with(
                process.output,
                words.iter().any(|word| word == "--json"),
                &failure,
                &[],
            );
        }
    };
    let command =
        parsed.first == cli::line::FirstWord::Command || words.as_slice() == ["--version"];
    let matches = if command {
        let mut command_words = words.clone();
        if matches!(
            command_words.first().map(String::as_str),
            Some("--help" | "-h")
        ) {
            command_words[0] = "help".into();
        }
        if command_words.first().is_some_and(|word| word == "help")
            && cli::run_line::help(&command_words[1..])
        {
            return ExitCode::SUCCESS;
        }
        let mut argv = vec!["polylogue".to_owned()];
        argv.extend(command_words);
        let matches = match cli::command().try_get_matches_from(argv) {
            Ok(matches) => matches,
            Err(error) => {
                client_pipe_signal();
                if hook {
                    return hook_startup_error(error);
                }
                return cli::parse_error(error, process.output);
            }
        };
        if !keeps_sigpipe_ignored(&matches) {
            client_pipe_signal();
        }
        if let Some(result) = cli::run_local(&matches, process.output) {
            return result;
        }
        Some(matches)
    } else {
        client_pipe_signal();
        None
    };
    let startup = match config::Startup::load(&process, overrides) {
        Ok(startup) => startup,
        Err(error) => return startup_error(hook, error, process.output),
    };
    crate::client::remember_this_process(&process.environment);
    if native {
        return browser::native::run(
            &startup.engine,
            &polylogue_store::Archive::at(&startup.config.archive),
            &startup.config.paths,
        );
    }
    match matches {
        Some(matches) => cli::run(&matches, &startup),
        None => cli::run_line::run(parsed, &startup),
    }
}

/// The daemon and `polylogue internal host` write to peers that may go away, so a
/// closed pipe is an error to them, never a reason to die.
fn keeps_sigpipe_ignored(matches: &clap::ArgMatches) -> bool {
    matches!(matches.subcommand_name(), Some("daemon"))
        || matches!(matches.subcommand(), Some(("internal", internal)) if internal.subcommand_name() == Some("host"))
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

/// Whether the command line names `internal hook`: its first words that are
/// neither a global option nor that option's value.
fn hook_invocation(arguments: &[std::ffi::OsString]) -> bool {
    let mut words = arguments.iter().skip(1).filter_map(|word| word.to_str());
    while let Some(word) = words.next() {
        if cli::GLOBAL_VALUED_OPTIONS.contains(&word) {
            words.next();
        } else if !word.starts_with('-') {
            return word == "internal" && words.next() == Some("hook");
        }
    }
    false
}

/// Completion resolves deployment flags without parsing unfinished query
/// words. Each flag is parsed alone, so an incomplete or invalid one drops
/// only itself.
fn completion_overrides(arguments: &[std::ffi::OsString]) -> config::Overrides {
    let mut groups: Vec<Vec<std::ffi::OsString>> = Vec::new();
    let mut arguments = arguments.iter().skip(1);
    while let Some(argument) = arguments.next() {
        let Some(text) = argument.to_str() else {
            continue;
        };
        if ["--archive", "--config", "--listen", "--auth-token-file"].contains(&text) {
            let mut group = vec![argument.clone()];
            if let Some(value) = arguments.next() {
                group.push(value.clone());
            }
            groups.push(group);
        } else if ["--archive=", "--config=", "--listen=", "--auth-token-file="]
            .iter()
            .any(|prefix| text.starts_with(prefix))
        {
            groups.push(vec![argument.clone()]);
        }
    }
    let mut merged = config::Overrides::default();
    for group in groups {
        let flags = std::iter::once(std::ffi::OsString::from("polylogue")).chain(group);
        let Ok(matches) =
            config::augment_command(clap::Command::new("polylogue")).try_get_matches_from(flags)
        else {
            continue;
        };
        let found = config::Overrides::from_matches(&matches);
        merged = config::Overrides {
            config_file: found.config_file.or(merged.config_file),
            archive: found.archive.or(merged.archive),
            listen: found.listen.or(merged.listen),
            auth_token_file: found.auth_token_file.or(merged.auth_token_file),
        };
    }
    merged
}

#[cfg(test)]
mod tests;
