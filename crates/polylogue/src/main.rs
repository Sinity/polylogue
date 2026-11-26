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

use cli::invocation::Shape;
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
    let process = config::ProcessInputs::capture();
    let arguments = std::env::args_os().collect::<Vec<_>>();
    let invocation = cli::invocation::scan(&arguments, &process.environment);
    let start = Start {
        // A harness treats a non-zero hook exit as a refusal of its tool
        // call, so a hook that cannot start says why on stderr and exits 0.
        hook: matches!(invocation.shape, Shape::Hook),
        json: invocation.json,
        output: process.output,
    };
    if let Shape::Completion(completion) = &invocation.shape {
        return complete(&process, &invocation, completion, &arguments);
    }
    if let Some(problem) = &invocation.problem {
        return start.failed(&protocol::failure(
            polylogue_model::Code::BadRequest,
            problem,
        ));
    }
    let words = &invocation.words;
    match invocation.shape {
        Shape::Completion(_) => ExitCode::SUCCESS,
        Shape::NativeHost => {
            client_pipe_signal();
            let startup = match load_startup(&process, &invocation, &start) {
                Ok(startup) => startup,
                Err(exit) => return exit,
            };
            browser::native::run(
                &startup.engine,
                &polylogue_store::Archive::at(&startup.config.archive),
                &startup.config.paths,
            )
        }
        Shape::Help if cli::run_line::help(&words[1..]) => ExitCode::SUCCESS,
        Shape::Help | Shape::Hook | Shape::Command(_) => {
            let mut argv = vec!["polylogue".to_owned()];
            argv.extend(words.iter().cloned());
            if matches!(invocation.shape, Shape::Help) {
                argv[1] = "help".into();
            }
            let matches = match cli::command().try_get_matches_from(argv) {
                Ok(matches) => matches,
                Err(error) => {
                    client_pipe_signal();
                    if start.hook {
                        return hook_startup_error(error);
                    }
                    return cli::parse_error(error, process.output);
                }
            };
            if !keeps_sigpipe_ignored(&matches) {
                client_pipe_signal();
            }
            match cli::configuration(&matches) {
                cli::Configuration::None => cli::run_local(&matches, process.output),
                cli::Configuration::Client => {
                    let client = match config::Client::load(&process, invocation.overrides) {
                        Ok(client) => client,
                        Err(error) => return start.failed(&error.failure()),
                    };
                    crate::client::remember_this_process(&process.environment);
                    cli::run(&matches, &client)
                }
                cli::Configuration::Deployment => {
                    let mut startup = match load_startup(&process, &invocation, &start) {
                        Ok(startup) => startup,
                        Err(exit) => return exit,
                    };
                    if cli::needs_engine(&matches, &startup.mode) {
                        startup.stopped_attempt = match executable_attempt() {
                            Ok(attempt) => Some(attempt),
                            Err(error) => {
                                return start.failed(&protocol::failure(
                                    polylogue_model::Code::Io,
                                    format!("reading this executable: {error}"),
                                ));
                            }
                        };
                    }
                    cli::run_configured(&matches, &startup)
                }
            }
        }
        Shape::Line => {
            client_pipe_signal();
            let client = match config::Client::load(&process, invocation.overrides.clone()) {
                Ok(client) => client,
                Err(error) => return start.failed(&error.failure()),
            };
            // Saved names are recognised from the catalog the daemon
            // publishes beside the archive.
            let catalog = cli::line::FirstWordCatalog::load(&client.paths, &client.archive);
            let parsed = match cli::line::parse(words, &catalog, cli::run_line::terminal()) {
                Ok(parsed) => parsed,
                Err(error) => {
                    let mut failure = protocol::failure(error.code, error.message);
                    failure.hint = error.try_line.map(|line| format!("try: {line}"));
                    if !error.candidates.is_empty() {
                        failure
                            .hint
                            .get_or_insert_default()
                            .push_str(&format!("\n{}", error.candidates.join("\n")));
                    }
                    return cli::fail_with(process.output, invocation.json, &failure, &[]);
                }
            };
            crate::client::remember_this_process(&process.environment);
            cli::run_line::run(parsed, &client)
        }
    }
}

/// How a failure before any command ran is reported.
struct Start {
    hook: bool,
    json: bool,
    output: config::OutputConfig,
}

impl Start {
    /// Exits with the failure's code from the failure table; a hook exits 0.
    fn failed(&self, failure: &command_surface::Failure) -> ExitCode {
        if self.hook {
            return hook_startup_error(&failure.message);
        }
        cli::fail_with(self.output, self.json, failure, &[])
    }
}

/// The whole deployment, for the daemon and the commands that run the
/// engine in this process.
fn load_startup(
    process: &config::ProcessInputs,
    invocation: &cli::invocation::Invocation,
    start: &Start,
) -> Result<config::Startup, ExitCode> {
    let startup = config::Client::load(process, invocation.overrides.clone())
        .and_then(|client| config::Startup::extend(client, process))
        .map_err(|error| start.failed(&error.failure()))?;
    crate::client::remember_this_process(&process.environment);
    Ok(startup)
}

/// Answers a shell's completion callback.
fn complete(
    process: &config::ProcessInputs,
    invocation: &cli::invocation::Invocation,
    completion: &cli::invocation::Completion,
    arguments: &[std::ffi::OsString],
) -> ExitCode {
    client_pipe_signal();
    // Unfinished words complete with what the arguments so far select.
    let catalog = config::Client::load(process, invocation.overrides.clone())
        .ok()
        .map(|client| cli::completion::Catalog {
            path: client.paths.completion_catalog(&client.archive),
            archive_key: runtime_dir::archive_key(&client.archive),
        });
    cli::completion::capture(cli::completion::Context {
        arguments: completion.arguments.clone(),
        index: completion.index,
        catalog,
        prefix: completion.prefix.clone(),
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
    ExitCode::SUCCESS
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
        // Parser selection currently has no configurable parser settings.
        parser_settings: None,
    })
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

fn hook_startup_error(error: impl std::fmt::Display) -> ExitCode {
    let text = error.to_string();
    let line = text
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("");
    eprintln!("polylogue hook: {}", line.trim_start_matches("error: "));
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests;
