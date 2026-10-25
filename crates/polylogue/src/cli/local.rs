//! Commands that run in the calling process rather than in the daemon:
//! the daemon itself, the MCP server, the hook command, the browser host,
//! setup outside the archive and completions.

use super::fail;
use crate::commands::{Command, Values};
use std::process::ExitCode;

pub fn daemon(_: &Command, _: &Values, _: bool) -> ExitCode {
    match crate::daemon::server::serve() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("polylogue: {e}");
            let other_schema = e
                .get_ref()
                .and_then(|inner| inner.downcast_ref::<polylogue_store::StoreError>())
                .is_some_and(|inner| {
                    matches!(inner, polylogue_store::StoreError::OtherSchema { .. })
                });
            if other_schema {
                ExitCode::from(crate::EXIT_OTHER_SCHEMA)
            } else {
                ExitCode::FAILURE
            }
        }
    }
}

pub fn mcp(_: &Command, _: &Values, _: bool) -> ExitCode {
    match crate::mcp::serve_stdio() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("polylogue mcp: {e}");
            ExitCode::FAILURE
        }
    }
}

pub fn hook(_: &Command, v: &Values, _: bool) -> ExitCode {
    crate::hook::run(&v.text("event").unwrap_or_default())
}

pub fn web(_: &Command, _: &Values, json: bool) -> ExitCode {
    open_reader(json)
}

pub fn browser_host(_: &Command, _: &Values, _: bool) -> ExitCode {
    crate::browser::native::run()
}

pub fn completions(_: &Command, v: &Values, _: bool) -> ExitCode {
    use std::str::FromStr;
    let shell = v.text("shell").unwrap_or_default();
    let Ok(shell) = clap_complete::Shell::from_str(&shell) else {
        return fail(false, "bad_request", &format!("unknown shell {shell:?}"), 2);
    };
    clap_complete::generate(
        shell,
        &mut super::command(),
        "polylogue",
        &mut std::io::stdout(),
    );
    ExitCode::SUCCESS
}

pub fn hooks_install(_: &Command, v: &Values, json: bool) -> ExitCode {
    hooks(v, json, true)
}

pub fn hooks_uninstall(_: &Command, v: &Values, json: bool) -> ExitCode {
    hooks(v, json, false)
}

fn hooks(v: &Values, json: bool, on: bool) -> ExitCode {
    let path = v
        .text("settings")
        .map_or_else(crate::hook::settings_path, std::path::PathBuf::from);
    match crate::hook::install(&path, on) {
        Ok(()) => {
            println!(
                "{} Polylogue's hooks {} {}",
                if on { "installed" } else { "removed" },
                if on { "in" } else { "from" },
                path.display()
            );
            ExitCode::SUCCESS
        }
        Err(e) => fail(json, "io", &e.to_string(), 1),
    }
}

/// A command declared but not built: a typed refusal.
pub fn not_built(c: &Command, _: &Values, json: bool) -> ExitCode {
    fail(
        json,
        "not_built",
        &format!("`{}` is not built yet", c.name),
        2,
    )
}

pub fn install_agent(_: &Command, v: &Values, json: bool) -> ExitCode {
    install_chrome(
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

fn install_chrome(json: bool, harness: &str, extension_id: Option<&str>, remove: bool) -> ExitCode {
    if harness != "chrome" {
        return fail(
            json,
            "not_built",
            &format!(
                "`install-agent {harness}` is not built yet; for Claude Code hooks use `polylogue hooks install`"
            ),
            1,
        );
    }
    let config = std::env::var_os("XDG_CONFIG_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".config")))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
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
                    json,
                    "bad_request",
                    "`--extension-id` (the ID chrome://extensions shows) is required",
                    2,
                );
            };
            let binary = std::env::current_exe().and_then(std::fs::canonicalize);
            let Ok(binary) = binary else {
                return fail(json, "io", "cannot resolve this binary's path", 1);
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
            return fail(json, "io", &format!("{}: {e}", path.display()), 1);
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
fn open_reader(json: bool) -> ExitCode {
    use crate::web::access;
    let root = polylogue_store::Archive::default_root();
    let auth = match access::reader_url(&root) {
        Ok(url) => url,
        Err(error) => {
            return fail(
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
        eprintln!("polylogue: could not start a browser; open {auth}");
    }
    ExitCode::SUCCESS
}
