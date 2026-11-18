//! Commands that run in the calling process rather than in the daemon:
//! the daemon itself, the MCP server, the hook command, the browser host,
//! setup outside the archive and completions.

use polylogue_model::JsonExt as _;
pub mod arguments;

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
    match crate::daemon::server::serve(startup, run) {
        Ok(None) => ExitCode::SUCCESS,
        Ok(Some(summary)) => {
            if json {
                match serde_json::to_string(&summary) {
                    Ok(text) => println!("{text}"),
                    Err(error) => eprintln!("polylogue: the run summary: {error}"),
                }
            } else {
                print!("{}", super::print::idle_summary(&summary));
            }
            // A run that met errors did not process everything it was given.
            if summary.work.errors.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(e) => {
            eprint!(
                "{}",
                super::print::diagnostic("", &e.to_string(), None, startup.output.stderr_color())
            );
            if let Some(failure) = polylogue_store::storage_failure::recorded() {
                crate::daemon::storage_stop::remember(
                    &startup.config.paths,
                    &startup.config.archive,
                    &failure,
                );
                return ExitCode::from(crate::EXIT_STORAGE_FAILED);
            }
            if e.get_ref()
                .is_some_and(|inner| inner.is::<crate::daemon::storage_stop::StorageStopped>())
            {
                return ExitCode::from(crate::EXIT_STORAGE_FAILED);
            }
            let refused = e
                .get_ref()
                .and_then(|inner| inner.downcast_ref::<polylogue_store::StoreError>())
                .is_some_and(|inner| {
                    matches!(
                        inner,
                        polylogue_store::StoreError::NewerSchema { .. }
                            | polylogue_store::StoreError::OtherFormat { .. }
                            | polylogue_store::StoreError::EmbeddingsSchemaDiffers { .. }
                            | polylogue_store::StoreError::OlderFormat { .. }
                    )
                });
            if refused {
                ExitCode::from(crate::EXIT_SCHEMA_REFUSED)
            } else {
                ExitCode::FAILURE
            }
        }
    }
}

pub fn mcp(
    _: &Command,
    _: &command_surface::ParsedCommand,
    startup: &crate::config::Startup,
    _: bool,
) -> ExitCode {
    match crate::mcp::serve_stdio(&polylogue_store::Archive::at(&startup.config.archive)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprint!(
                "{}",
                super::print::diagnostic(
                    "",
                    &format!("mcp: {e}"),
                    None,
                    startup.output.stderr_color()
                )
            );
            ExitCode::FAILURE
        }
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
    startup: &crate::config::Startup,
    json: bool,
) -> ExitCode {
    open_reader(startup, json)
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

/// Install into a harness or Chrome, or print a shell completion script.
pub fn install(
    _: &Command,
    v: &command_surface::ParsedCommand,
    startup: &crate::config::Startup,
    json: bool,
) -> ExitCode {
    install_with_output(v, startup, json, &mut std::io::stdout().lock())
}

fn install_with_output(
    v: &command_surface::ParsedCommand,
    startup: &crate::config::Startup,
    json: bool,
    output: &mut dyn std::io::Write,
) -> ExitCode {
    let arguments: arguments::Install = match serde_json::from_value(v.request.arguments.clone()) {
        Ok(arguments) => arguments,
        Err(error) => {
            return fail(
                startup.output,
                json,
                polylogue_model::Code::BadRequest,
                &error.to_string(),
                2,
            );
        }
    };
    let target = arguments.target.as_str();
    let not_built = |part: &str| {
        fail(
            startup.output,
            json,
            polylogue_model::Code::NotBuilt,
            &format!("install {part} is not built yet"),
            2,
        )
    };
    if arguments.instructions {
        return not_built("--instructions");
    }
    if matches!(target, "zsh" | "bash" | "fish") {
        if !arguments.dry_run || arguments.remove {
            return not_built(target);
        }
        return match completion_script(target, arguments.program.as_deref().unwrap_or("polylogue"))
        {
            Ok(script) => match output.write_all(script.as_bytes()) {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => fail(
                    startup.output,
                    json,
                    polylogue_model::Code::Io,
                    &error.to_string(),
                    1,
                ),
            },
            Err(error) => fail(
                startup.output,
                json,
                polylogue_model::Code::BadRequest,
                &error,
                2,
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
    if arguments.dry_run {
        return not_built("--dry-run");
    }
    let Some(harness) = crate::hook::Harness::parse(target) else {
        return not_built(target);
    };
    let path = match arguments.settings {
        Some(path) => std::path::PathBuf::from(path),
        None => match crate::hook::settings_path(harness, &startup.engine) {
            Ok(path) => path,
            Err(message) => {
                return fail(
                    startup.output,
                    json,
                    polylogue_model::Code::BadRequest,
                    &message,
                    2,
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
                    startup.output,
                    json,
                    polylogue_model::Code::Io,
                    &error.to_string(),
                    1,
                ),
            }
        }
        Err(error) => fail(
            startup.output,
            json,
            polylogue_model::Code::Io,
            &error.to_string(),
            1,
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
    completer
        .write_registration("COMPLETE", program, program, program, &mut registration)
        .map_err(|error| error.to_string())?;
    String::from_utf8(registration)
        .map(|registration| {
            registration.replace(
                "COMPLETE=\"zsh\"",
                "_POLYLOGUE_COMPLETE_PREFIX=\"$PREFIX\" COMPLETE=\"zsh\"",
            )
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
        2,
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
                startup.output,
                json,
                polylogue_model::Code::BadRequest,
                "`--extension-id` (the ID chrome://extensions shows) is required",
                2,
            );
        };
        let binary = match std::env::current_exe().and_then(std::fs::canonicalize) {
            Ok(binary) => binary,
            Err(error) => {
                return fail(
                    startup.output,
                    json,
                    polylogue_model::Code::Io,
                    &error.to_string(),
                    1,
                );
            }
        };
        match serde_json::to_vec_pretty(&crate::browser::native::manifest(&binary, id)) {
            Ok(bytes) => Some(bytes),
            Err(error) => {
                return fail(
                    startup.output,
                    json,
                    polylogue_model::Code::Io,
                    &error.to_string(),
                    1,
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
            startup.output,
            json,
            polylogue_model::Code::Io,
            &error.to_string(),
            1,
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
fn open_reader(startup: &crate::config::Startup, json: bool) -> ExitCode {
    use crate::web::access;
    let root = startup.config.archive.clone();
    let auth = match access::reader_url(&startup.config.paths, &root) {
        Ok(url) => url,
        Err(error) => {
            return fail(
                startup.output,
                json,
                polylogue_model::Code::ReaderUnavailable,
                &error.to_string(),
                crate::EXIT_DAEMON_REQUIRED,
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
                startup.output.stderr_color()
            )
        );
    }
    ExitCode::SUCCESS
}

/// `polylogue host`: holds a hosted agent's stdio for the daemon until the
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
        polylogue_model::live::channel::host_channel(&harness),
    ) else {
        return fail(
            output,
            json,
            polylogue_model::Code::BadRequest,
            &format!("{harness} is not a harness Polylogue hosts over stdio"),
            2,
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
            2,
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
        Err(e) => fail(output, json, polylogue_model::Code::Io, &e.to_string(), 1),
    }
}

#[cfg(test)]
mod tests;
