//! Native process, terminal and Git observations made by the hook command.
use serde_json::{Value, json};
use std::io::{Read, Seek};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

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

fn git_identity(
    cwd: &Path,
    budget: Duration,
) -> Result<(Value, Option<String>, Vec<String>), &'static str> {
    let until = Instant::now() + budget;
    let common = git(
        cwd,
        &["rev-parse", "--path-format=absolute", "--git-common-dir"],
        until,
    )?;
    let head = git(cwd, &["rev-parse", "--verify", "HEAD"], until)?;
    let roots = git(cwd, &["rev-list", "--max-parents=0", &head], until)?;
    let mut roots: Vec<_> = roots.lines().map(str::to_owned).collect();
    roots.sort();
    roots.dedup();
    if common.is_empty() || head.is_empty() || roots.is_empty() {
        return Err("git identity: incomplete native identity");
    }
    let mut unavailable = Vec::new();
    let mut optional = |args: &[&str], fact: &str| match git(cwd, args, until) {
        Ok(value) if !value.is_empty() => Some(value),
        Ok(_) => {
            unavailable.push(format!("{fact}: empty native value"));
            None
        }
        Err(reason) => {
            unavailable.push(format!("{fact}: {reason}"));
            None
        }
    };
    let branch = optional(&["symbolic-ref", "--quiet", "--short", "HEAD"], "branch");
    let repository = optional(&["config", "--get", "remote.origin.url"], "repository");
    Ok((
        json!({"common_dir":common,"root_commits":roots,"head":head,"branch":branch}),
        repository,
        unavailable,
    ))
}

fn git(cwd: &Path, args: &[&str], until: Instant) -> Result<String, &'static str> {
    if Instant::now() >= until {
        return Err("git identity: observation budget elapsed");
    }
    let mut output = tempfile::tempfile().map_err(|_| "git identity: output unavailable")?;
    let mut child = Command::new("git")
        .arg("--no-optional-locks")
        .arg("-C")
        .arg(cwd)
        .args(args)
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .stdout(Stdio::from(
            output
                .try_clone()
                .map_err(|_| "git identity: output unavailable")?,
        ))
        .spawn()
        .map_err(|_| "git identity: git unavailable")?;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < until => std::thread::sleep(Duration::from_millis(1)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("git identity: observation budget elapsed");
            }
        }
    };
    if !status.success() {
        return Err("git identity: native query unavailable");
    }
    output
        .rewind()
        .map_err(|_| "git identity: output unavailable")?;
    let mut value = String::new();
    output
        .read_to_string(&mut value)
        .map_err(|_| "git identity: non-text native output")?;
    Ok(value.strip_suffix('\n').unwrap_or(&value).to_owned())
}

#[cfg(test)]
mod tests;
