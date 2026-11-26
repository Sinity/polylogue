//! Commands that run in the calling process rather than in the daemon:
//! the daemon itself, the MCP server, the hook command, the browser host,
//! setup outside the archive and completions.

use polylogue_model::JsonExt as _;
pub mod arguments;
mod instructions;
mod shell;

use super::fail;
use crate::commands::Command;
use std::process::ExitCode;

pub fn daemon(
    _: &Command,
    v: &command_surface::ParsedCommand,
    startup: &crate::config::Startup,
    json: bool,
) -> ExitCode {
    let run = crate::daemon::server::Run {
        reads_sources: !v.request.arguments.bool("no_sources").unwrap_or(false),
        lifetime: if v.request.arguments.bool("until_idle").unwrap_or(false) {
            polylogue_engine::Lifetime::UntilIdle
        } else {
            polylogue_engine::Lifetime::UntilStopped
        },
    };
    let served = crate::daemon::server::serve(startup, run);
    let end = DaemonEnd::of(&served, polylogue_store::storage_failure::recorded());
    match &served {
        Ok(None) => {}
        Ok(Some(summary)) => {
            if json {
                match serde_json::to_string(summary) {
                    Ok(text) => println!("{text}"),
                    Err(error) => eprintln!("polylogue: the run summary: {error}"),
                }
            } else {
                print!("{}", super::print::idle_summary(summary));
            }
        }
        Err(e) => eprint!(
            "{}",
            super::print::diagnostic(
                "",
                &e.to_string(),
                None,
                startup.client.output.stderr_color()
            )
        ),
    }
    if let DaemonEnd::StorageFailed(failure) = &end {
        // A run that ended without an error still says why its exit is 74.
        if served.is_ok() {
            eprint!(
                "{}",
                super::print::diagnostic(
                    "",
                    &failure.to_string(),
                    None,
                    startup.client.output.stderr_color()
                )
            );
        }
        crate::daemon::storage_stop::remember(
            &startup.config.paths,
            &startup.config.archive,
            failure,
        );
    }
    ExitCode::from(end.exit_code())
}

/// How a daemon run ended. One value decides the exit status and whether
/// the boot is remembered as failed, so no way of stopping can skip either.
#[derive(Debug, PartialEq, Eq)]
enum DaemonEnd {
    /// This process recorded a failed sync of the register. It comes first
    /// whatever else ended the run: a stop signal that arrives with the
    /// failure must not let the next start write again in this boot.
    StorageFailed(polylogue_store::storage_failure::StorageFailure),
    /// The start was refused because an earlier process of this boot
    /// recorded a storage failure.
    StorageFailedEarlier,
    /// The archive or the embeddings are in a format this build refuses.
    ArchiveRefused,
    /// Another process holds the archive lock.
    Locked,
    /// The daemon could not start serving, for a reason the failure table
    /// classifies: an archive directory it cannot create or open is `io`.
    StartFailed(command_surface::FailureKind),
    /// A run that served and then failed.
    Failed,
    /// Stopped by a signal, with nothing to report.
    Stopped,
    /// A run until idle ended; `complete` when it met no errors, since a
    /// run that met errors did not process everything it was given.
    Idle { complete: bool },
}

impl DaemonEnd {
    fn of(
        served: &Result<Option<polylogue_engine::RunSummary>, crate::daemon::server::Failed>,
        recorded: Option<polylogue_store::storage_failure::StorageFailure>,
    ) -> Self {
        use crate::daemon::server::Failed;
        if let Some(failure) = recorded {
            return Self::StorageFailed(failure);
        }
        let failed = match served {
            Ok(None) => return Self::Stopped,
            Ok(Some(summary)) => {
                return Self::Idle {
                    complete: summary.work.errors.is_empty(),
                };
            }
            Err(failed) => failed,
        };
        let inner = failed.error().get_ref();
        if inner.is_some_and(|inner| inner.is::<crate::daemon::storage_stop::StorageStopped>()) {
            return Self::StorageFailedEarlier;
        }
        let store = inner.and_then(|inner| inner.downcast_ref::<polylogue_store::StoreError>());
        match store {
            Some(
                polylogue_store::StoreError::NewerSchema { .. }
                | polylogue_store::StoreError::OtherFormat { .. }
                | polylogue_store::StoreError::EmbeddingsSchemaDiffers { .. }
                | polylogue_store::StoreError::OlderFormat { .. },
            ) => return Self::ArchiveRefused,
            Some(polylogue_store::StoreError::Locked { .. }) => return Self::Locked,
            _ => {}
        }
        match failed {
            Failed::Run(_) => Self::Failed,
            Failed::Start(error) => Self::StartFailed(match store {
                Some(store) => crate::protocol::store_failure(store).kind,
                None => crate::protocol::failure(polylogue_model::Code::Io, error.to_string()).kind,
            }),
        }
    }

    fn exit_code(&self) -> u8 {
        match self {
            Self::StorageFailed(_) | Self::StorageFailedEarlier => crate::EXIT_STORAGE_FAILED,
            Self::ArchiveRefused => crate::EXIT_SCHEMA_REFUSED,
            Self::StartFailed(kind) => kind.exit_code(),
            Self::Stopped | Self::Idle { complete: true } => 0,
            Self::Locked | Self::Failed | Self::Idle { complete: false } => 1,
        }
    }
}

pub fn mcp(
    _: &Command,
    _: &command_surface::ParsedCommand,
    client: &crate::config::Client,
    _: bool,
) -> ExitCode {
    match crate::mcp::serve_stdio(&polylogue_store::Archive::at(&client.archive)) {
        Ok(()) => ExitCode::SUCCESS,
        // Standard output carries the MCP protocol, so the failure goes to
        // standard error as text.
        Err(error) => fail(
            client.output,
            false,
            polylogue_model::Code::Io,
            &format!("mcp: {error}"),
        ),
    }
}

pub fn hook(
    _: &Command,
    v: &command_surface::ParsedCommand,
    startup: &crate::config::Startup,
    _: bool,
) -> ExitCode {
    let arguments: arguments::Hook = match serde_json::from_value(v.request.arguments.clone()) {
        Ok(arguments) => arguments,
        Err(error) => {
            // A non-zero exit would block the harness's tool call.
            eprintln!("polylogue hook: {error}");
            return ExitCode::SUCCESS;
        }
    };
    crate::hook::run(arguments.harness, &arguments.event, startup)
}

pub fn web(
    _: &Command,
    _: &command_surface::ParsedCommand,
    client: &crate::config::Client,
    json: bool,
) -> ExitCode {
    open_reader(client, json)
}

pub fn browser_host(
    _: &Command,
    _: &command_surface::ParsedCommand,
    startup: &crate::config::Startup,
    _: bool,
) -> ExitCode {
    crate::browser::native::run(
        &startup.engine,
        &polylogue_store::Archive::at(&startup.config.archive),
        &startup.config.paths,
    )
}

/// Install harness hooks, browser or shell integration, or Drive authorization.
pub fn install(
    _: &Command,
    v: &command_surface::ParsedCommand,
    startup: &crate::config::Startup,
    json: bool,
) -> ExitCode {
    install_with_output(
        v,
        startup,
        json,
        &mut std::io::stdout().lock(),
        &crate::auth::Drive::default(),
    )
}

fn install_with_output(
    v: &command_surface::ParsedCommand,
    startup: &crate::config::Startup,
    json: bool,
    output: &mut dyn std::io::Write,
    drive: &crate::auth::Drive,
) -> ExitCode {
    let arguments: arguments::Install = match serde_json::from_value(v.request.arguments.clone()) {
        Ok(arguments) => arguments,
        Err(error) => {
            return fail(
                startup.client.output,
                json,
                polylogue_model::Code::BadRequest,
                &error.to_string(),
            );
        }
    };
    let target = arguments.target.as_str();
    let not_built = |part: &str| {
        fail(
            startup.client.output,
            json,
            polylogue_model::Code::NotBuilt,
            &format!("install {part} is not built yet"),
        )
    };
    if arguments.instructions {
        return match instructions::install(startup, &arguments, json, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => fail(
                startup.client.output,
                json,
                if error.kind() == std::io::ErrorKind::InvalidInput {
                    polylogue_model::Code::BadRequest
                } else {
                    polylogue_model::Code::Io
                },
                &error.to_string(),
            ),
        };
    }
    if matches!(target, "zsh" | "bash" | "fish") {
        return match shell::install(startup, &arguments, json, output) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => fail(
                startup.client.output,
                json,
                if error.kind() == std::io::ErrorKind::InvalidInput {
                    polylogue_model::Code::BadRequest
                } else {
                    polylogue_model::Code::Io
                },
                &error.to_string(),
            ),
        };
    }
    if target == "chrome" {
        return install_chrome(
            startup,
            json,
            arguments.extension_id.as_deref(),
            arguments.remove,
            arguments.dry_run,
            output,
        );
    }
    if target == "drive" {
        return install_drive(v, startup, &arguments, json, output, drive);
    }
    let Some(harness) = crate::hook::Harness::parse(target) else {
        return fail(
            startup.client.output,
            json,
            polylogue_model::Code::BadRequest,
            &format!(
                "`{target}` is not an install target; targets are claude-code, codex, gemini-cli, hermes, chrome, zsh, bash, fish and drive"
            ),
        );
    };
    if arguments.dry_run {
        return not_built("--dry-run");
    }
    let path = match arguments.settings {
        Some(path) => std::path::PathBuf::from(path),
        None => match crate::hook::settings_path(harness, &startup.engine) {
            Ok(path) => path,
            Err(message) => {
                return fail(
                    startup.client.output,
                    json,
                    polylogue_model::Code::BadRequest,
                    &message,
                );
            }
        },
    };
    match crate::hook::install(
        &path,
        harness,
        || crate::hook::probe_version(harness),
        !arguments.remove,
    ) {
        Ok(()) => {
            let result = if json {
                writeln!(
                    output,
                    "{}",
                    serde_json::json!({"removed":arguments.remove,"files":[path]})
                )
            } else {
                writeln!(
                    output,
                    "{} Polylogue's hooks {} {}",
                    if arguments.remove {
                        "removed"
                    } else {
                        "installed"
                    },
                    if arguments.remove { "from" } else { "in" },
                    path.display()
                )
            };
            match result {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => fail(
                    startup.client.output,
                    json,
                    polylogue_model::Code::Io,
                    &error.to_string(),
                ),
            }
        }
        Err(error) => fail(
            startup.client.output,
            json,
            polylogue_model::Code::Io,
            &error.to_string(),
        ),
    }
}

fn install_drive(
    parsed: &command_surface::ParsedCommand,
    startup: &crate::config::Startup,
    arguments: &arguments::Install,
    json: bool,
    output: &mut dyn std::io::Write,
    drive: &crate::auth::Drive,
) -> ExitCode {
    let result = (|| -> std::io::Result<()> {
        let archive = polylogue_store::Archive::at(&startup.config.archive);
        let path = polylogue_sources::ai_studio::drive::token_path(&archive.root);
        let client_id = if arguments.remove || path.exists() || !archive.root.exists() {
            None
        } else {
            let pool = polylogue_store::ReadPool::serving(
                &archive,
                1,
                startup
                    .engine
                    .reader_memory_per_connection()
                    .map_err(std::io::Error::other)?,
            )
            .map_err(std::io::Error::other)?;
            let reader = pool.reader().map_err(std::io::Error::other)?;
            match reader
                .setting_value("drive.client_id", &startup.config.identity.machine)
                .map_err(std::io::Error::other)?
            {
                Some(polylogue_model::settings::Value::Text(value)) => Some(value),
                _ => None,
            }
        };
        let wait = match parsed.request.deadline {
            command_surface::Deadline::Default => Some(std::time::Duration::from_secs(600)),
            command_surface::Deadline::None => None,
            command_surface::Deadline::Milliseconds(ms) => {
                Some(std::time::Duration::from_millis(ms))
            }
        };
        let report = drive.install(
            &archive.root,
            client_id.as_deref(),
            arguments.remove,
            arguments.dry_run,
            wait,
        )?;
        if json {
            writeln!(
                output,
                "{}",
                serde_json::to_string(&report).map_err(std::io::Error::other)?
            )?;
        } else {
            use crate::auth::Installation;
            match report {
                Installation::Authorized { path } => {
                    writeln!(output, "Drive authorized; credentials: {}", path.display())?
                }
                Installation::AuthorizationDryRun { url, scope } => {
                    writeln!(output, "Authorize {scope}:\n{url}")?
                }
                Installation::Removed { path } => writeln!(
                    output,
                    "Drive authorization revoked and {} removed",
                    path.display()
                )?,
                Installation::RemovalDryRun { path } => writeln!(
                    output,
                    "would revoke Drive authorization and remove {}",
                    path.display()
                )?,
                Installation::NotAuthorized { path } => {
                    writeln!(output, "no Drive authorization at {}", path.display())?
                }
            }
        }
        Ok(())
    })();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => fail(
            startup.client.output,
            json,
            if error.kind() == std::io::ErrorKind::InvalidInput {
                polylogue_model::Code::BadValue
            } else {
                polylogue_model::Code::Io
            },
            &error.to_string(),
        ),
    }
}

/// The shell code that asks this binary for completions while a person types.
fn completion_script(shell: &str, program: &str) -> Result<String, String> {
    let shells = clap_complete::env::Shells::builtins();
    let completer = shells
        .completer(shell)
        .ok_or_else(|| format!("unknown shell {shell:?}"))?;
    let mut registration = Vec::new();
    // Shell function names are separate from the quoted command they run.
    let name = format!(
        "polylogue_{}",
        &blake3::hash(program.as_bytes()).to_hex()[..12]
    );
    completer
        .write_registration("COMPLETE", &name, program, program, &mut registration)
        .map_err(|error| error.to_string())?;
    String::from_utf8(registration)
        .map(|registration| {
            let registration = registration.replace(
                "COMPLETE=\"zsh\"",
                "_POLYLOGUE_COMPLETE_PREFIX=\"$PREFIX\" COMPLETE=\"zsh\"",
            );
            let registration = if shell == "zsh" {
                registration.replacen('\n', "\nif (( ! $+functions[compdef] )); then\n    autoload -Uz compinit && compinit\nfi\n", 1)
            } else {
                registration
            };
            format!("{registration}{}", shell::picker_script(shell, program))
        })
        .map_err(|error| error.to_string())
}

/// A command declared but not built: a typed refusal.
pub fn not_built(
    c: &Command,
    _: &command_surface::ParsedCommand,
    output: crate::config::OutputConfig,
    json: bool,
) -> ExitCode {
    fail(
        output,
        json,
        polylogue_model::Code::NotBuilt,
        &format!("`{}` is not built yet", c.name),
    )
}

/// Chrome's and Chromium's host manifest directories under `config`. An
/// existing browser config directory gets the manifest; with none, Chrome's
/// does.
pub fn host_manifest_dirs(config: &std::path::Path) -> Vec<std::path::PathBuf> {
    let dirs: Vec<_> = ["google-chrome", "chromium"]
        .iter()
        .map(|b| config.join(b))
        .filter(|d| d.is_dir())
        .collect();
    let dirs = if dirs.is_empty() {
        vec![config.join("google-chrome")]
    } else {
        dirs
    };
    dirs.into_iter()
        .map(|d| d.join("NativeMessagingHosts"))
        .collect()
}

fn install_chrome(
    startup: &crate::config::Startup,
    json: bool,
    extension_id: Option<&str>,
    remove: bool,
    dry_run: bool,
    output: &mut dyn std::io::Write,
) -> ExitCode {
    let manifest = if remove {
        None
    } else {
        let Some(id) =
            extension_id.filter(|id| !id.is_empty() && id.bytes().all(|b| b.is_ascii_lowercase()))
        else {
            return fail(
                startup.client.output,
                json,
                polylogue_model::Code::BadRequest,
                "`--extension-id` (the ID chrome://extensions shows) is required",
            );
        };
        let binary = match std::env::current_exe().and_then(std::fs::canonicalize) {
            Ok(binary) => binary,
            Err(error) => {
                return fail(
                    startup.client.output,
                    json,
                    polylogue_model::Code::Io,
                    &error.to_string(),
                );
            }
        };
        match serde_json::to_vec_pretty(&crate::browser::native::manifest(&binary, id)) {
            Ok(bytes) => Some(bytes),
            Err(error) => {
                return fail(
                    startup.client.output,
                    json,
                    polylogue_model::Code::Io,
                    &error.to_string(),
                );
            }
        }
    };
    let result = (|| -> std::io::Result<()> {
        if dry_run && !remove {
            return output.write_all(manifest.as_deref().expect("an install has a manifest"));
        }
        let mut changed = Vec::new();
        let name = format!("{}.json", crate::browser::HOST_NAME);
        for directory in host_manifest_dirs(&startup.config.paths.config) {
            let path = directory.join(&name);
            let file = crate::installation::File::read(&path).map_err(|error| {
                if path.is_symlink() {
                    std::io::Error::other(format!(
                        "{error}; print the managed manifest with `polylogue install chrome --extension-id ID --dry-run`"
                    ))
                } else {
                    error
                }
            })?;
            if remove {
                let installed = file.current().is_some_and(installed_chrome_manifest);
                if !installed && file.backup().is_none() {
                    continue;
                }
                let restored = file.current().is_some() && file.current() == file.backup();
                if file.current().is_some() && !installed && !restored {
                    return Err(std::io::Error::other(format!(
                        "{} changed after installation; its original bytes remain in {}",
                        path.display(),
                        crate::installation::backup_path(&path).display()
                    )));
                }
                if !dry_run {
                    eprintln!("uninstalling {}", path.display());
                    file.restore()?;
                }
            } else {
                eprintln!("installing {}", path.display());
                file.write(manifest.as_deref().expect("an install has a manifest"))?;
            }
            changed.push(path);
        }
        if json {
            let mut report = serde_json::json!({"removed":remove,"files":changed});
            if dry_run {
                report["dry_run"] = true.into();
            }
            writeln!(output, "{report}")?;
        } else {
            let action = match (remove, dry_run) {
                (true, true) => "would uninstall",
                (true, false) => "uninstalled",
                (false, _) => "installed",
            };
            for path in &changed {
                writeln!(output, "{action} {}", path.display())?;
            }
            if changed.is_empty() {
                writeln!(output, "nothing to remove")?;
            }
        }
        Ok(())
    })();
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => fail(
            startup.client.output,
            json,
            polylogue_model::Code::Io,
            &error.to_string(),
        ),
    }
}

fn installed_chrome_manifest(bytes: &[u8]) -> bool {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return false;
    };
    let Some(binary) = value.get("path").and_then(serde_json::Value::as_str) else {
        return false;
    };
    let Some(origins) = value
        .get("allowed_origins")
        .and_then(serde_json::Value::as_array)
    else {
        return false;
    };
    if origins.len() != 1 {
        return false;
    }
    let Some(extension) = origins[0]
        .as_str()
        .and_then(|origin| origin.strip_prefix("chrome-extension://"))
        .and_then(|origin| origin.strip_suffix('/'))
    else {
        return false;
    };
    serde_json::to_vec_pretty(&crate::browser::native::manifest(
        std::path::Path::new(binary),
        extension,
    ))
    .is_ok_and(|expected| expected == bytes)
}

/// Opens the web reader: reads the running daemon's port and token, checks
/// that the port's listener is this user's, and opens `/auth#<token>` in
/// the browser. The fragment never reaches a server log or history.
fn open_reader(client: &crate::config::Client, json: bool) -> ExitCode {
    use crate::web::access;
    let root = client.archive.clone();
    let auth = match access::reader_url(&client.paths, &root) {
        Ok(url) => url,
        Err(error) => {
            return fail(
                client.output,
                json,
                polylogue_model::Code::DaemonRequired,
                &format!(
                    "no running daemon serves the web reader ({error}); start it with `polylogue daemon`"
                ),
            );
        }
    };
    let url = auth.split("auth#").next().unwrap_or(&auth);
    let opened = std::process::Command::new("xdg-open")
        .arg(&auth)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if json {
        println!("{}", serde_json::json!({ "url": url, "opened": opened }));
    } else if opened {
        println!("opened {url}");
    } else {
        // The operator's own terminal: the sign-in link may be shown here.
        eprint!(
            "{}",
            super::print::diagnostic(
                "",
                &format!("could not start a browser; open {auth}"),
                None,
                client.output.stderr_color()
            )
        );
    }
    ExitCode::SUCCESS
}

/// `polylogue internal host`: holds a hosted agent's stdio for the daemon until the
/// agent exits (`crate::hosting`).
pub fn host(
    _: &Command,
    v: &command_surface::ParsedCommand,
    output: crate::config::OutputConfig,
    json: bool,
) -> ExitCode {
    let harness = v
        .request
        .arguments
        .str("harness")
        .map(str::to_owned)
        .unwrap_or_default();
    let session_id = v.request.arguments.str("session_id").map(str::to_owned);
    let (Some(argv), Ok(protocol)) = (
        crate::hosting::agent_command(&harness, session_id.as_deref()),
        polylogue_model::live::channel::host_protocol(&harness),
    ) else {
        return fail(
            output,
            json,
            polylogue_model::Code::BadRequest,
            &format!("{harness} is not a harness Polylogue hosts over stdio"),
        );
    };
    let path = |name: &str| {
        v.request
            .arguments
            .get(name)
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .map(std::path::PathBuf::from)
    };
    let (Some(socket), Some(cwd), Some(spool)) = (path("socket"), path("cwd"), path("spool"))
    else {
        return fail(
            output,
            json,
            polylogue_model::Code::BadRequest,
            "host takes --socket, --cwd and --spool",
        );
    };
    let config = crate::hosting::HostConfig {
        harness,
        protocol,
        argv,
        cwd,
        socket,
        spool,
        session_id,
    };
    match crate::hosting::serve(&config) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => fail(output, json, polylogue_model::Code::Io, &e.to_string()),
    }
}

#[cfg(test)]
mod tests;
