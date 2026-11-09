//! Native process, terminal and Git observations made by the hook command.
use serde_json::{Value, json};
use std::path::Path;
use std::time::Duration;

/// A shared budget for optional Git observation at session start.
const GIT_BUDGET: Duration = Duration::from_millis(50);

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
            boot: std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
                .ok()
                .map(|s| s.trim_end_matches('\n').to_owned()),
            kitty_pid: inputs.kitty_pid,
            kitty_window: inputs.kitty_window.clone(),
        }
    }
}

pub(super) fn snapshot(event: &str, payload: &Value, environment: &Environment) -> Value {
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
        if let Some(cwd) = payload.get("cwd").and_then(Value::as_str) {
            match git_identity(Path::new(cwd), GIT_BUDGET) {
                Ok((identity, repository, unavailable)) => {
                    result["git_identity"] = identity;
                    if !unavailable.is_empty() {
                        result["native_unavailable"] = json!(unavailable);
                    }
                    if let Some(repository) = repository {
                        result["repository"] = json!(repository);
                    }
                }
                Err(reason) => result["native_unavailable"] = json!([reason]),
            }
        } else {
            result["native_unavailable"] = json!(["git identity: no source cwd"]);
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
    budget: Duration,
) -> Result<(Value, Option<String>, Vec<String>), String> {
    let identity = ::git_identity::observe_within(cwd, budget)
        .map_err(|reason| format!("git identity: {reason}"))?;
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

#[cfg(test)]
mod tests;
