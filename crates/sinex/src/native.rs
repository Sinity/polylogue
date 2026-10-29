//! Native session attribution, without interpreting prose or resolving cwd paths.
use crate::kinds::{GitIdentity, Objective, RecordRef, TerminalIdentity};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Default)]
pub(crate) struct Facts {
    pub repo: Option<String>,
    pub objectives: Vec<Objective>,
    pub git_identity: Option<GitIdentity>,
    pub terminal: Option<TerminalIdentity>,
    pub boot_id: Option<String>,
    pub sources: Vec<RecordRef>,
    pub unavailable: Vec<String>,
}

fn distinct<T: PartialEq>(values: Vec<T>) -> Option<T> {
    let mut values = values.into_iter();
    let first = values.next()?;
    values.all(|value| value == first).then_some(first)
}

fn named(value: &Value) -> Option<String> {
    value.as_str().filter(|s| !s.is_empty()).map(str::to_owned)
}

fn observed<T: DeserializeOwned>(
    body: &Value,
    key: &str,
    unavailable: &mut Vec<String>,
) -> Option<T> {
    let value = body.get(key).filter(|value| !value.is_null())?;
    match serde_json::from_value(value.clone()) {
        Ok(value) => Some(value),
        Err(_) => {
            unavailable.push(format!("invalid native {key}"));
            None
        }
    }
}

/// Accumulates only explicitly structured facts, paired with the record carrying them.
#[derive(Default)]
pub(crate) struct Native {
    out: Facts,
    repositories: BTreeSet<String>,
    git: Vec<GitIdentity>,
    terminals: Vec<TerminalIdentity>,
    boots: BTreeSet<String>,
}

impl Native {
    pub fn new(record_repo: Option<String>) -> Self {
        Self {
            repositories: record_repo.into_iter().collect(),
            ..Self::default()
        }
    }

    pub fn observe(&mut self, body: &Value, source: RecordRef) {
        if ![
            "repository",
            "objectives",
            "git_identity",
            "terminal_identity",
            "boot_id",
            "native_unavailable",
        ]
        .iter()
        .any(|key| body.get(key).is_some())
        {
            return;
        }
        let Self {
            out,
            repositories,
            git,
            terminals,
            boots,
        } = self;
        if !out.sources.contains(&source) {
            out.sources.push(source);
        }
        repositories.extend(body.get("repository").and_then(named));
        boots.extend(body.get("boot_id").and_then(named));
        if let Some(values) = observed::<Vec<Objective>>(body, "objectives", &mut out.unavailable) {
            for objective in values {
                if matches!(objective.kind.as_str(), "task" | "pull_request")
                    && !objective.reference.is_empty()
                    && !objective.repository.is_empty()
                {
                    if !out.objectives.contains(&objective) {
                        out.objectives.push(objective);
                    }
                } else {
                    out.unavailable
                        .push("invalid native objective reference".into());
                }
            }
        }
        if let Some(value) = observed::<GitIdentity>(body, "git_identity", &mut out.unavailable) {
            if !value.common_dir.is_empty()
                && !value.root_commits.is_empty()
                && value.root_commits.iter().all(|root| !root.is_empty())
            {
                git.push(value);
            } else {
                out.unavailable
                    .push("incomplete native git identity".into());
            }
        }
        if let Some(mut value) =
            observed::<TerminalIdentity>(body, "terminal_identity", &mut out.unavailable)
        {
            if value.boot_id.is_none() {
                value.boot_id = body.get("boot_id").and_then(named);
            }
            if !value.host.is_empty()
                && !value.adapter.is_empty()
                && value.process_id > 0
                && !value.window_id.is_empty()
            {
                terminals.push(value);
            } else {
                out.unavailable
                    .push("incomplete native terminal identity".into());
            }
        }
        out.unavailable.extend(
            body.get("native_unavailable")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(named),
        );
    }

    pub fn finish(mut self) -> Facts {
        let Self {
            ref mut out,
            repositories,
            git,
            terminals,
            boots,
        } = self;
        if repositories.len() > 1 {
            out.unavailable
                .push("conflicting native repository observations".into());
        }
        out.repo = distinct(repositories.into_iter().collect());
        if !git.is_empty() && git.iter().any(|value| value != &git[0]) {
            out.unavailable
                .push("conflicting native git observations".into());
        }
        if !terminals.is_empty() && terminals.iter().any(|value| value != &terminals[0]) {
            out.unavailable
                .push("conflicting native terminal observations".into());
        }
        out.git_identity = distinct(git);
        out.terminal = distinct(terminals);
        if boots.len() > 1 {
            out.unavailable
                .push("conflicting native boot observations".into());
        }
        out.boot_id = distinct(boots.into_iter().collect());
        out.objectives.sort_by(|a, b| {
            (&a.kind, &a.repository, &a.reference).cmp(&(&b.kind, &b.repository, &b.reference))
        });
        out.unavailable.sort();
        out.unavailable.dedup();
        self.out
    }
}

#[cfg(test)]
mod tests;
