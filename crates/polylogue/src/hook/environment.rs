//! Process, terminal and Git observations the hook command makes and states in its spool line.
use crate::config::Paths;
use polylogue_model::JsonExt as _;
use serde_json::{Value, json};
use std::path::Path;
use std::time::Instant;

pub(super) struct Environment {
    host: String,
    boot: Option<String>,
    kitty_pid: Option<u32>,
    kitty_window: Option<String>,
}

impl Environment {
    pub(super) fn observe(host: &str, inputs: &super::Inputs) -> Self {
        Self {
            host: host.to_owned(),
            boot: crate::live_events::boot_id(),
            kitty_pid: inputs.kitty_pid,
            kitty_window: inputs.kitty_window.clone(),
        }
    }
}

pub(super) fn snapshot(
    event: &str,
    payload: &Value,
    environment: &Environment,
    paths: &Paths,
    deadline: Instant,
) -> Value {
    let mut result = json!({});
    if let Some(boot) = &environment.boot {
        result["boot_id"] = json!(boot);
    }
    if !environment.host.is_empty()
        && let (Some(pid), Some(window)) = (environment.kitty_pid, &environment.kitty_window)
    {
        result["terminal_identity"] = json!({
            "host": environment.host, "adapter": "kitty", "process_id": pid,
            "parent_process_id": parent(pid), "window_id": window,
            "boot_id": environment.boot,
        });
    }
    let starting = event == "SessionStart"
        || (event == "pre_llm_call"
            && payload
                .pointer("/extra/is_first_turn")
                .and_then(Value::as_bool)
                == Some(true));
    if starting {
        if let Some(cwd) = payload.str("cwd") {
            match git_identity(Path::new(cwd), paths, deadline) {
                Ok((identity, repository, unavailable)) => {
                    result["git_identity"] = identity;
                    if !unavailable.is_empty() {
                        result["unstated"] = json!(unavailable);
                    }
                    if let Some(repository) = repository {
                        result["repository"] = json!(repository);
                    }
                }
                Err(reason) => result["unstated"] = json!([reason]),
            }
        } else {
            result["unstated"] = json!(["git identity: no source cwd"]);
        }
    }
    result
}

fn parent(pid: u32) -> Option<u32> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    stat.rsplit_once(") ")?
        .1
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

/// The session's repository through common's `git-identity`, the same rule
/// Sinex and the repository lookup use: root commits without shallow-clone
/// boundaries, the `origin` remote without credentials.
fn git_identity(
    cwd: &Path,
    paths: &Paths,
    deadline: Instant,
) -> Result<(Value, Option<String>, Vec<String>), String> {
    let cache = ::git_identity::repository_of(cwd).map(|repository| paths.git_roots(&repository));
    let known: Option<::git_identity::KnownRoots> = cache
        .as_ref()
        .and_then(|path| std::fs::read(path).ok())
        .and_then(|bytes| serde_json::from_slice(&bytes).ok());
    let (identity, next) = ::git_identity::observe_reusing(cwd, deadline, known.as_ref())
        .map_err(|reason| format!("git identity: {reason}"))?;
    // This is disposable derived state. An unwritable cache never discards a
    // successful observation, and concurrent hooks publish complete tuples.
    if known.as_ref() != Some(&next)
        && let Some(path) = cache
    {
        let _ = save_roots(&path, &next);
    }
    if identity.root_commits.is_empty() {
        return Err("git identity: history reaches no root commit".to_owned());
    }
    let mut unavailable = Vec::new();
    if identity.branch.is_none() {
        unavailable.push("branch: HEAD is detached".to_owned());
    }
    if identity.remote.is_none() {
        unavailable.push("repository: no origin remote".to_owned());
    }
    Ok((
        json!({
            "common_dir": identity.common_dir,
            "root_commits": identity.root_commits,
            "head": identity.head,
            "branch": identity.branch,
        }),
        identity.remote,
        unavailable,
    ))
}

fn save_roots(path: &Path, roots: &::git_identity::KnownRoots) -> std::io::Result<()> {
    let parent = path.parent().expect("Git roots path has a parent");
    std::fs::create_dir_all(parent)?;
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer(file.as_file_mut(), roots)?;
    file.persist(path).map_err(|error| error.error)?;
    Ok(())
}

#[cfg(test)]
mod tests;
