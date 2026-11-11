//! Commands that run in the calling process rather than in the daemon:
//! the daemon itself, the MCP server, the hook command, the browser host,
//! setup outside the archive and completions.

use super::fail;
use crate::commands::{Command, Values};
use std::process::ExitCode;

pub fn daemon(_: &Command, v: &Values, startup: &crate::config::Startup, json: bool) -> ExitCode {
    let run = crate::daemon::server::Run {
        reads_sources: !v.flag("no_sources"),
        lifetime: if v.flag("until_idle") {
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
                crate::daemon::storage_stop::remember(&startup.config.archive, &failure);
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
                            | polylogue_store::StoreError::NoMigration { .. }
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

pub fn mcp(_: &Command, _: &Values, startup: &crate::config::Startup, _: bool) -> ExitCode {
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

pub fn hook(_: &Command, v: &Values, startup: &crate::config::Startup, _: bool) -> ExitCode {
    let name = v.text("harness").unwrap_or_else(|| "claude-code".into());
    let Some(harness) = crate::hook::Harness::parse(&name) else {
        // A non-zero exit would block the harness's tool call.
        eprintln!("polylogue hook: unknown harness {name:?}");
        return ExitCode::SUCCESS;
    };
    crate::hook::run(harness, &v.text("event").unwrap_or_default(), startup)
}

pub fn web(_: &Command, _: &Values, startup: &crate::config::Startup, json: bool) -> ExitCode {
    open_reader(startup, json)
}

pub fn browser_host(
    _: &Command,
    _: &Values,
    startup: &crate::config::Startup,
    _: bool,
) -> ExitCode {
    crate::browser::native::run(
        &startup.engine,
        &polylogue_store::Archive::at(&startup.config.archive),
    )
}

/// Prints what `COMPLETE=<shell> polylogue` prints: shell code that asks
/// this binary for completions as the operator types.
pub fn completions(
    _: &Command,
    v: &Values,
    output: crate::config::OutputConfig,
    _: bool,
) -> ExitCode {
    let shell = v.text("shell").unwrap_or_default();
    let shells = clap_complete::env::Shells::builtins();
    let Some(completer) = shells.completer(&shell) else {
        return fail(
            output,
            false,
            "bad_request",
            &format!("unknown shell {shell:?}"),
            2,
        );
    };
    // A wrapper on PATH that runs this binary (Sinnix's `polylogue-live`)
    // is completed and asked under its own name, so it keeps its archive.
    let name = v.text("program").unwrap_or_else(|| "polylogue".into());
    let mut registration = Vec::new();
    match completer.write_registration("COMPLETE", &name, &name, &name, &mut registration) {
        Ok(()) => {
            let registration = String::from_utf8_lossy(&registration);
            let registration = registration.replace(
                "COMPLETE=\"zsh\"",
                "_POLYLOGUE_COMPLETE_PREFIX=\"$PREFIX\" COMPLETE=\"zsh\"",
            );
            print!("{registration}");
            ExitCode::SUCCESS
        }
        Err(e) => fail(output, false, "io", &e.to_string(), 1),
    }
}

pub fn hooks_install(
    _: &Command,
    v: &Values,
    startup: &crate::config::Startup,
    json: bool,
) -> ExitCode {
    hooks(v, startup, json, true)
}

pub fn hooks_uninstall(
    _: &Command,
    v: &Values,
    startup: &crate::config::Startup,
    json: bool,
) -> ExitCode {
    hooks(v, startup, json, false)
}

fn hooks(v: &Values, startup: &crate::config::Startup, json: bool, on: bool) -> ExitCode {
    let name = v.text("harness").unwrap_or_else(|| "claude-code".into());
    let Some(harness) = crate::hook::Harness::parse(&name) else {
        // A non-zero exit would block the harness's tool call.
        eprintln!("polylogue hook: unknown harness {name:?}");
        return ExitCode::SUCCESS;
    };
    let path = match v.text("settings") {
        Some(path) => std::path::PathBuf::from(path),
        None => match crate::hook::settings_path(harness, &startup.engine) {
            Ok(path) => path,
            Err(message) => return fail(startup.output, json, "bad_request", &message, 2),
        },
    };
    match crate::hook::install(&path, harness, || crate::hook::probe_version(harness), on) {
        Ok(()) => {
            println!(
                "{} Polylogue's hooks {} {}",
                if on { "installed" } else { "removed" },
                if on { "in" } else { "from" },
                path.display()
            );
            ExitCode::SUCCESS
        }
        Err(e) => fail(startup.output, json, "io", &e.to_string(), 1),
    }
}

/// A command declared but not built: a typed refusal.
pub fn not_built(
    c: &Command,
    _: &Values,
    output: crate::config::OutputConfig,
    json: bool,
) -> ExitCode {
    fail(
        output,
        json,
        "not_built",
        &format!("`{}` is not built yet", c.name),
        2,
    )
}

pub fn install_agent(
    _: &Command,
    v: &Values,
    startup: &crate::config::Startup,
    json: bool,
) -> ExitCode {
    install_chrome(
        startup,
        json,
        &v.text("harness").unwrap_or_default(),
        v.text("extension_id").as_deref(),
        v.flag("remove"),
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
    harness: &str,
    extension_id: Option<&str>,
    remove: bool,
) -> ExitCode {
    if harness != "chrome" {
        return fail(
            startup.output,
            json,
            "not_built",
            &format!(
                "`install-agent {harness}` is not built yet; for Claude Code hooks use `polylogue hooks install`"
            ),
            1,
        );
    }
    let config = startup.config.paths.config.clone();
    let file = format!("{}.json", crate::browser::HOST_NAME);
    let mut changed = Vec::new();
    for dir in host_manifest_dirs(&config) {
        let path = dir.join(&file);
        let result = if remove {
            match std::fs::remove_file(&path) {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                other => other,
            }
        } else {
            let Some(id) = extension_id
                .filter(|id| !id.is_empty() && id.bytes().all(|b| b.is_ascii_lowercase()))
            else {
                return fail(
                    startup.output,
                    json,
                    "bad_request",
                    "`--extension-id` (the ID chrome://extensions shows) is required",
                    2,
                );
            };
            let binary = std::env::current_exe().and_then(std::fs::canonicalize);
            let Ok(binary) = binary else {
                return fail(
                    startup.output,
                    json,
                    "io",
                    "cannot resolve this binary's path",
                    1,
                );
            };
            let manifest = crate::browser::native::manifest(&binary, id);
            std::fs::create_dir_all(&dir).and_then(|()| {
                std::fs::write(
                    &path,
                    serde_json::to_vec_pretty(&manifest).unwrap_or_default(),
                )
            })
        };
        if let Err(e) = result {
            return fail(
                startup.output,
                json,
                "io",
                &format!("{}: {e}", path.display()),
                1,
            );
        }
        changed.push(path);
    }
    if json {
        println!(
            "{}",
            serde_json::json!({ "removed": remove, "files": changed })
        );
    } else {
        for p in &changed {
            println!(
                "{} {}",
                if remove { "removed" } else { "wrote" },
                p.display()
            );
        }
        if changed.is_empty() {
            println!("nothing to remove");
        }
    }
    ExitCode::SUCCESS
}

/// Opens the web reader: reads the running daemon's port and token, checks
/// that the port's listener is this user's, and opens `/auth#<token>` in
/// the browser. The fragment never reaches a server log or history.
fn open_reader(startup: &crate::config::Startup, json: bool) -> ExitCode {
    use crate::web::access;
    let root = startup.config.archive.clone();
    let auth = match access::reader_url(&root) {
        Ok(url) => url,
        Err(error) => {
            return fail(
                startup.output,
                json,
                "reader_unavailable",
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

/// The same standing instructions the MCP server supplies at initialization.
pub fn manual(
    _: &Command,
    _: &Values,
    _output: crate::config::OutputConfig,
    json: bool,
) -> ExitCode {
    let text = crate::manual::text();
    if json {
        println!(
            "{}",
            serde_json::to_string(&command_surface::Answer::ok(
                serde_json::json!({"manual": text})
            ))
            .expect("manual is serializable")
        );
    } else {
        print!("{text}");
    }
    ExitCode::SUCCESS
}

/// `polylogue host`: holds a hosted agent's stdio for the daemon until the
/// agent exits (`crate::hosting`).
pub fn host(_: &Command, v: &Values, output: crate::config::OutputConfig, json: bool) -> ExitCode {
    let harness = v.text("harness").unwrap_or_default();
    let session_id = v.text("session_id");
    let (Some(argv), Ok(protocol)) = (
        crate::hosting::agent_command(&harness, session_id.as_deref()),
        polylogue_model::live::channel::host_channel(&harness),
    ) else {
        return fail(
            output,
            json,
            "bad_request",
            &format!("{harness} is not a harness Polylogue hosts over stdio"),
            2,
        );
    };
    let path = |name: &str| v.text(name).map(std::path::PathBuf::from);
    let (Some(socket), Some(cwd), Some(spool)) = (path("socket"), path("cwd"), path("spool"))
    else {
        return fail(
            output,
            json,
            "bad_request",
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
        Err(e) => fail(output, json, "io", &e.to_string(), 1),
    }
}
