//! The work graph's derive step: goals, attempts, verifications and
//! artifacts from a session's tool calls, found by their structure. A Bead
//! is claimed only by a successful `bd update <id> --claim` (or `--status
//! in_progress`) and ends at a successful `bd close` or release in the same
//! session; an ID mentioned in a prompt or a message never makes a claim.
//! Todo and plan items are what the agent declared, shown as text and never
//! turned into a task identity. Commits and pull requests come from the output of the commands
//! that made them. The session's live task is its open claim.

use crate::Result;
use crate::encoding::{unpack_block, unpack_text};
use polylogue_model::command::{Stage, effective_stages};
use polylogue_model::{Block, Json, WorkEdgeKind, WorkNodeKind};
use rusqlite::{OptionalExtension, Transaction, params};
use std::collections::BTreeMap;

pub(crate) const DERIVATION: super::Derivation = super::Derivation {
    name: "work",
    // Bumped when the rules change; every session is derived again.
    version: super::Version::Rules(2),
    // Reads tool calls' commands and outcome reasons.
    after: &["tool_calls"],
    run: |tx, session_pk, _| {
        let id: String = tx
            .prepare_cached("SELECT id FROM session WHERE pk = ?1")?
            .query_row([session_pk], |r| r.get(0))?;
        derive_session(tx, session_pk, &id)
    },
    finish: Some(sweep),
};

/// Commands whose outcome checks work: tests, builds, lints and the
/// repository's own check runners. A grouping by command shape, never by
/// what the output says.
fn is_check(stage: &Stage) -> bool {
    let sub = stage.subcommand();
    let args = |w: &str| stage.args.iter().any(|a| a == w);
    match stage.program.as_str() {
        "cargo" => match sub {
            Some("test" | "nextest" | "check" | "clippy" | "build" | "fmt") => true,
            Some("run" | "xtask") => args("xtask") || sub == Some("xtask"),
            _ => false,
        },
        "pytest" | "tox" | "nox" | "mypy" | "ruff" | "eslint" | "tsc" => true,
        "npm" | "pnpm" | "yarn" => {
            matches!(sub, Some("test" | "run"))
                && (args("test") || args("check") || sub == Some("test"))
        }
        "go" => matches!(sub, Some("test" | "vet" | "build")),
        "nix" => match sub {
            Some("build") => true,
            Some("flake") => args("check"),
            _ => false,
        },
        "make" | "just" => args("test") || args("check"),
        "xtask" => true,
        _ => false,
    }
}

/// What a successful `bd` stage does to the Beads it names.
enum BeadChange {
    Claim,
    Close,
    Release,
}

fn bead_change(stage: &Stage) -> Option<(BeadChange, Vec<String>)> {
    if stage.program != "bd" {
        return None;
    }
    let sub = stage.subcommand()?;
    let ops = stage.operands();
    let mut ids = Vec::new();
    let mut status: Option<&str> = None;
    let mut claim = false;
    let mut it = ops.iter();
    while let Some(a) = it.next() {
        if let Some(v) = a
            .strip_prefix("--status=")
            .or_else(|| a.strip_prefix("-s="))
        {
            status = Some(v);
        } else if a == "--status" || a == "-s" {
            status = it.next().map(String::as_str);
        } else if a == "--claim" {
            claim = true;
        } else if a.starts_with('-') {
            // Options with values this rule does not need are skipped with
            // their value when they take one.
            if matches!(
                a.as_str(),
                "--actor"
                    | "--reason"
                    | "-r"
                    | "--assignee"
                    | "-a"
                    | "--notes"
                    | "--title"
                    | "-t"
                    | "--description"
                    | "-d"
                    | "--priority"
                    | "-p"
                    | "--append-notes"
                    | "--parent"
                    | "--body-file"
                    | "--design-file"
                    | "--acceptance"
                    | "-l"
                    | "--label"
            ) {
                it.next();
            }
        } else {
            ids.push(a.clone());
        }
    }
    if ids.is_empty() {
        return None;
    }
    let change = match sub {
        "close" => BeadChange::Close,
        "update" if claim || status == Some("in_progress") => BeadChange::Claim,
        "update" if matches!(status, Some("open" | "blocked" | "deferred")) => BeadChange::Release,
        "reopen" => BeadChange::Release,
        _ => return None,
    };
    Some((change, ids))
}

struct Call {
    pk: i64,
    at: Option<i64>,
    tool: String,
    command: Option<String>,
    outcome: String,
    input: Option<Json>,
}

/// A goal the session pursued, with its attempt's span.
struct Pursuit {
    goal_key: String,
    goal_title: String,
    shared: bool,
    from: Option<i64>,
    until: Option<i64>,
    state: &'static str,
    checks_ok: u32,
    checks_failed: u32,
    checks_unknown: u32,
    commits: u32,
}

impl Pursuit {
    fn open_at(&self, at: Option<i64>) -> bool {
        match (self.from, at) {
            (Some(from), Some(at)) => from <= at && self.until.is_none_or(|u| at <= u),
            // Without times, a call counts against every attempt it follows
            // in session order, which `calls` already is.
            _ => true,
        }
    }
}

/// An excision removes a session's nodes (and, by cascade, their edges)
/// but leaves a shared goal no attempt pursues any more.
fn sweep(index: &Transaction<'_>) -> Result<()> {
    index.execute(
        "DELETE FROM work_node WHERE session_pk IS NULL AND kind = 'goal'
             AND NOT EXISTS (SELECT 1 FROM work_edge WHERE to_pk = work_node.pk)",
        [],
    )?;
    Ok(())
}

fn calls(tx: &Transaction<'_>, session_pk: i64) -> Result<Vec<Call>> {
    let mut stmt = tx.prepare_cached(
        "SELECT t.pk, t.at, t.tool, t.command, t.outcome
         FROM tool_call t
         WHERE t.session_pk = ?1
         ORDER BY t.at IS NULL, t.at, t.pk",
    )?;
    let rows = stmt.query_map([session_pk], |r| {
        Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, Option<i64>>(1)?,
            r.get::<_, String>(2)?,
            r.get::<_, Option<String>>(3)?,
            r.get::<_, String>(4)?,
        ))
    })?;
    let mut out = Vec::new();
    for row in rows {
        let (pk, at, tool, command, outcome) = row?;
        // A block this build cannot decode has no declared input to read;
        // it never stops the derivation of the rest.
        let input = if matches!(tool.as_str(), "TodoWrite" | "update_plan") {
            let body: Vec<u8> = tx
                .prepare_cached(
                    "SELECT b.body FROM tool_call t JOIN block b ON b.pk = t.call_block_pk
                     WHERE t.pk = ?1",
                )?
                .query_row([pk], |r| r.get(0))?;
            match unpack_block(&body) {
                Ok(Block::ToolCall { input, .. }) if !input.is_null() => Some(input),
                _ => None,
            }
        } else {
            None
        };
        out.push(Call {
            pk,
            at,
            tool,
            command,
            outcome,
            input,
        });
    }
    Ok(out)
}

/// Only artifact-producing commands read their output. Checks and claims
/// use the recorded command and outcome without reading stored result bodies.
fn result(tx: &Transaction<'_>, call_pk: i64) -> Result<Option<String>> {
    let (text, body): (Option<Vec<u8>>, Option<Vec<u8>>) = tx
        .prepare_cached(
            "SELECT rt.body, rb.body FROM tool_call t
             LEFT JOIN block rb ON rb.pk = t.result_block_pk
             LEFT JOIN text rt ON rt.id = rb.text_id WHERE t.pk = ?1",
        )?
        .query_row([call_pk], |r| Ok((r.get(0)?, r.get(1)?)))?;
    match text {
        Some(text) => Ok(Some(unpack_text(&text)?)),
        None => Ok(match body.map(|body| unpack_block(&body)) {
            Some(Ok(Block::ToolResult { output, .. })) => output,
            _ => None,
        }),
    }
}

/// `[branch abc1234] subject` lines of `git commit` output.
fn committed(output: &str) -> Vec<String> {
    output
        .lines()
        .filter_map(|l| {
            let inner = l.trim().strip_prefix('[')?.split_once(']')?.0;
            let hash = inner.rsplit(' ').next()?;
            (hash.len() >= 7 && hash.chars().all(|c| c.is_ascii_hexdigit()))
                .then(|| hash.to_ascii_lowercase())
        })
        .collect()
}

/// Pull request URLs in `gh pr create` output: `(owner/repo, number, url)`.
fn pull_requests(output: &str) -> Vec<(String, i64, String)> {
    output
        .split_whitespace()
        .filter_map(|w| {
            let rest = w.strip_prefix("https://github.com/")?;
            let mut parts = rest.split('/');
            let (owner, repo, pull, number) =
                (parts.next()?, parts.next()?, parts.next()?, parts.next()?);
            let n: i64 = number
                .trim_end_matches(|c: char| !c.is_ascii_digit())
                .parse()
                .ok()?;
            (pull == "pull").then(|| (format!("{owner}/{repo}"), n, w.to_owned()))
        })
        .collect()
}

/// Declared todo or plan items: `(item, status)`.
fn declared_items(call: &Call) -> Vec<(String, String)> {
    let Some(input) = &call.input else {
        return vec![];
    };
    let (list, text_key, status_key) = match call.tool.as_str() {
        "TodoWrite" => (input.get("todos"), "content", "status"),
        "update_plan" => (input.get("plan"), "step", "status"),
        _ => return vec![],
    };
    list.and_then(Json::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let text = item.get(text_key)?.as_str()?.trim();
            let status = item
                .get(status_key)
                .and_then(Json::as_str)
                .unwrap_or("pending");
            (!text.is_empty()).then(|| (text.to_owned(), status.to_owned()))
        })
        .collect()
}

fn derive_session(tx: &Transaction<'_>, session_pk: i64, session_id: &str) -> Result<()> {
    clear(tx, session_pk)?;
    let calls = calls(tx, session_pk)?;
    // Pursuits in the order their goals were first declared or claimed.
    let mut pursuits: BTreeMap<String, Pursuit> = BTreeMap::new();
    let mut order: Vec<String> = Vec::new();
    let mut checks: Vec<(&Call, String)> = Vec::new();
    let mut artifacts: Vec<(&Call, String, String)> = Vec::new();
    for call in &calls {
        for (item, status) in declared_items(call) {
            let key = format!("todo:{session_id}:{item}");
            let p = pursuits.entry(key.clone()).or_insert_with(|| {
                order.push(key.clone());
                Pursuit {
                    goal_key: key.clone(),
                    goal_title: item.clone(),
                    shared: false,
                    from: call.at,
                    until: None,
                    state: "pending",
                    checks_ok: 0,
                    checks_failed: 0,
                    checks_unknown: 0,
                    commits: 0,
                }
            });
            p.state = match status.as_str() {
                "completed" => {
                    p.until.get_or_insert(call.at.unwrap_or(i64::MAX));
                    "completed"
                }
                "in_progress" => "open",
                _ => "pending",
            };
        }
        if call.outcome != "ok" {
            if let Some(check) = check_of(call) {
                checks.push((call, check));
            }
            continue;
        }
        let Some(command) = &call.command else {
            continue;
        };
        let stages = effective_stages(command);
        let output = if stages.iter().any(|stage| {
            (stage.program == "git" && stage.subcommand() == Some("commit"))
                || (stage.program == "gh"
                    && stage.subcommand() == Some("pr")
                    && stage.operands().first().is_some_and(|o| o == "create"))
        }) {
            result(tx, call.pk)?
        } else {
            None
        };
        for stage in &stages {
            if let Some((change, ids)) = bead_change(stage) {
                for id in ids {
                    let key = format!("bead:{id}");
                    match change {
                        BeadChange::Claim => {
                            let p = pursuits.entry(key.clone()).or_insert_with(|| {
                                order.push(key.clone());
                                Pursuit {
                                    goal_key: key.clone(),
                                    goal_title: id.clone(),
                                    shared: true,
                                    from: call.at,
                                    until: None,
                                    state: "open",
                                    checks_ok: 0,
                                    checks_failed: 0,
                                    checks_unknown: 0,
                                    commits: 0,
                                }
                            });
                            if p.state != "open" {
                                // Claimed again after a release or close.
                                p.state = "open";
                                p.until = None;
                            }
                        }
                        BeadChange::Close | BeadChange::Release => {
                            if let Some(p) = pursuits.get_mut(&key)
                                && p.state == "open"
                            {
                                p.state = if matches!(change, BeadChange::Close) {
                                    "closed"
                                } else {
                                    "released"
                                };
                                p.until = Some(call.at.unwrap_or(i64::MAX));
                            }
                        }
                    }
                }
            }
            match (stage.program.as_str(), stage.subcommand()) {
                ("git", Some("commit")) => {
                    for hash in committed(output.as_deref().unwrap_or("")) {
                        artifacts.push((call, format!("commit:{session_id}:{hash}"), hash));
                    }
                }
                ("gh", Some("pr")) if stage.operands().first().is_some_and(|o| o == "create") => {
                    for (repo, n, url) in pull_requests(output.as_deref().unwrap_or("")) {
                        tx.prepare_cached(
                            "INSERT OR IGNORE INTO pr_link(session_pk, repo, number, url, evidence)
                             VALUES (?1, ?2, ?3, ?4, 'gh-output')",
                        )?
                        .execute(params![session_pk, repo, n, url])?;
                        artifacts.push((
                            call,
                            format!("pr:{session_id}:{repo}#{n}"),
                            format!("{repo}#{n}"),
                        ));
                    }
                }
                _ => {}
            }
        }
        if let Some(check) = check_of(call) {
            checks.push((call, check));
        }
    }
    // A spawned session pursues the task it was given: its first prompt,
    // shown as text. Everything it checks and makes belongs to that attempt.
    if let Some(task) = spawned_task(tx, session_pk)? {
        let key = format!("task:{session_id}");
        order.insert(0, key.clone());
        pursuits.insert(
            key.clone(),
            Pursuit {
                goal_key: key,
                goal_title: task,
                shared: false,
                from: None,
                until: None,
                state: "",
                checks_ok: 0,
                checks_failed: 0,
                checks_unknown: 0,
                commits: 0,
            },
        );
    }
    // Attempts, with what was checked and made while each was open.
    let mut attempt_pk: BTreeMap<String, i64> = BTreeMap::new();
    for (call, _) in &checks {
        for p in pursuits.values_mut().filter(|p| p.open_at(call.at)) {
            match call.outcome.as_str() {
                "ok" => p.checks_ok += 1,
                "error" => p.checks_failed += 1,
                _ => p.checks_unknown += 1,
            }
        }
    }
    for (call, _, _) in &artifacts {
        for p in pursuits.values_mut().filter(|p| p.open_at(call.at)) {
            p.commits += 1;
        }
    }
    for key in &order {
        let p = &pursuits[key];
        let goal = node(
            tx,
            &p.goal_key,
            WorkNodeKind::Goal,
            &p.goal_title,
            (!p.shared).then_some(session_pk),
            p.from,
            None,
            None,
            None,
        )?;
        let attrs = serde_json::json!({
            "checks_passed": p.checks_ok,
            "checks_failed": p.checks_failed,
            "checks_unknown": p.checks_unknown,
            "artifacts": p.commits,
        });
        let attempt = node(
            tx,
            &format!("attempt:{session_id}:{}", p.goal_key),
            WorkNodeKind::Attempt,
            &p.goal_title,
            Some(session_pk),
            p.from,
            // A spawned session's attempt has no state its records show.
            (!p.state.is_empty()).then_some(p.state),
            None,
            Some(&attrs.to_string()),
        )?;
        edge(tx, attempt, goal, WorkEdgeKind::Pursues)?;
        attempt_pk.insert(key.clone(), attempt);
    }
    for (call, title) in &checks {
        let check = node(
            tx,
            &format!("check:{}", call.pk),
            WorkNodeKind::Verification,
            title,
            Some(session_pk),
            call.at,
            Some(&call.outcome),
            Some(call.pk),
            None,
        )?;
        for (key, attempt) in &attempt_pk {
            if pursuits[key].open_at(call.at) {
                edge(tx, check, *attempt, WorkEdgeKind::Checks)?;
            }
        }
    }
    for (call, key, title) in &artifacts {
        if key.starts_with("commit:") {
            let hash = title;
            tx.prepare_cached(
                "INSERT OR IGNORE INTO commit_link(session_pk, commit_hash, evidence_block_pk, evidence)
                 SELECT ?1, ?2, call_block_pk, 'commit-output' FROM tool_call WHERE pk = ?3",
            )?
            .execute(params![session_pk, hash, call.pk])?;
        }
        let artifact = node(
            tx,
            key,
            WorkNodeKind::Artifact,
            title,
            Some(session_pk),
            call.at,
            None,
            Some(call.pk),
            None,
        )?;
        for (k, attempt) in &attempt_pk {
            if pursuits[k].open_at(call.at) {
                edge(tx, *attempt, artifact, WorkEdgeKind::Produces)?;
            }
        }
    }
    // The live task: the session's latest Bead claim still open.
    let task = order
        .iter()
        .rev()
        .map(|k| &pursuits[k])
        .find(|p| p.shared && p.state == "open")
        .map(|p| p.goal_title.clone());
    tx.prepare_cached(
        "UPDATE session SET live_task = ?2,
             live_task_evidence = CASE WHEN ?2 IS NULL THEN NULL ELSE 'claim_command' END
         WHERE pk = ?1",
    )?
    .execute(params![session_pk, task])?;
    Ok(())
}

/// The first line of a spawned session's first prompt, when a spawn link
/// names the session as its child (a fork or continuation is no new task).
fn spawned_task(tx: &Transaction<'_>, session_pk: i64) -> Result<Option<String>> {
    let spawned: bool = tx
        .prepare_cached(
            "SELECT EXISTS (SELECT 1 FROM link WHERE child_pk = ?1 AND kind = 'subagent')",
        )?
        .query_row([session_pk], |r| r.get(0))?;
    if !spawned {
        return Ok(None);
    }
    let first: Option<Vec<u8>> = tx
        .prepare_cached(
            "SELECT x.body FROM message m JOIN block b ON b.message_pk = m.pk
             JOIN text x ON x.id = b.text_id
             WHERE m.session_pk = ?1 AND m.role = 'user' AND b.kind = 'text'
             ORDER BY m.order_input_id, m.seq, b.idx LIMIT 1",
        )?
        .query_row([session_pk], |r| r.get(0))
        .optional()?;
    Ok(first
        .map(|b| unpack_text(&b))
        .transpose()?
        .and_then(|t| task_line(&t)))
}

/// The first line of a prompt that states the task: harness wrapper blocks
/// (`<fork-boilerplate>…</fork-boilerplate>`, `<system-reminder>…`) and
/// lone tags are skipped, never taken as the goal.
fn task_line(prompt: &str) -> Option<String> {
    let mut inside: Option<String> = None;
    // A block that never closes still yields its first plain line.
    let mut fallback: Option<&str> = None;
    for line in prompt.lines().map(str::trim) {
        if let Some(tag) = &inside {
            if line.starts_with(&format!("</{tag}")) {
                inside = None;
            } else if fallback.is_none() && !line.is_empty() && tag_name(line).is_none() {
                fallback = Some(line);
            }
            continue;
        }
        if line.is_empty() {
            continue;
        }
        if let Some(name) = tag_name(line) {
            if line.ends_with(&format!("</{name}>")) || name.starts_with('/') {
                // A whole block on one line, or a stray closing tag.
                continue;
            }
            if line.find('>') == Some(line.len() - 1) {
                // An opening tag alone on its line: its block ends further down.
                inside = Some(name.to_owned());
                continue;
            }
        }
        return Some(line.chars().take(200).collect());
    }
    fallback.map(|l| l.chars().take(200).collect())
}

/// The tag name when a line starts with one (`<name`, `</name`).
fn tag_name(line: &str) -> Option<&str> {
    let rest = line.strip_prefix('<')?;
    let name = rest.split(|c: char| c == '>' || c.is_whitespace()).next()?;
    let ok = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '/'));
    ok.then_some(name)
}

/// The check a call ran, as its stage's normalized form: the last stage
/// that is not a directory change or an output filter.
fn check_of(call: &Call) -> Option<String> {
    effective_stages(call.command.as_deref()?)
        .iter()
        .rev()
        .find(|s| {
            !matches!(
                s.program.as_str(),
                "cd" | "pushd" | "tail" | "head" | "tee" | "grep"
            )
        })
        .filter(|s| is_check(s))
        .map(Stage::normalized)
}

/// Removes the session's own work rows, the edges touching them, and the
/// links this step derived, then shared goals no attempt pursues any more.
fn clear(tx: &Transaction<'_>, session_pk: i64) -> Result<()> {
    tx.prepare_cached(
        "DELETE FROM work_edge WHERE from_pk IN (SELECT pk FROM work_node WHERE session_pk = ?1)
             OR to_pk IN (SELECT pk FROM work_node WHERE session_pk = ?1)",
    )?
    .execute([session_pk])?;
    tx.prepare_cached("DELETE FROM work_node WHERE session_pk = ?1")?
        .execute([session_pk])?;
    tx.prepare_cached(
        "DELETE FROM commit_link WHERE session_pk = ?1 AND evidence = 'commit-output'",
    )?
    .execute([session_pk])?;
    tx.prepare_cached("DELETE FROM pr_link WHERE session_pk = ?1 AND evidence = 'gh-output'")?
        .execute([session_pk])?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn node(
    tx: &Transaction<'_>,
    key: &str,
    kind: WorkNodeKind,
    title: &str,
    session_pk: Option<i64>,
    at: Option<i64>,
    state: Option<&str>,
    tool_call_pk: Option<i64>,
    attrs: Option<&str>,
) -> Result<i64> {
    let existing: Option<i64> = tx
        .prepare_cached("SELECT pk FROM work_node WHERE key = ?1")?
        .query_row([key], |r| r.get(0))
        .optional()?;
    if let Some(pk) = existing {
        // A shared goal keeps its earliest time.
        tx.prepare_cached(
            "UPDATE work_node SET at = CASE WHEN at IS NULL OR (?2 IS NOT NULL AND ?2 < at) THEN ?2 ELSE at END
             WHERE pk = ?1",
        )?
        .execute(params![pk, at])?;
        return Ok(pk);
    }
    tx.prepare_cached(
        "INSERT INTO work_node(key, kind, title, session_pk, at, state, tool_call_pk, attrs_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
    )?
    .execute(params![
        key,
        kind.as_str(),
        title,
        session_pk,
        at,
        state,
        tool_call_pk,
        attrs
    ])?;
    Ok(tx.last_insert_rowid())
}

fn edge(tx: &Transaction<'_>, from: i64, to: i64, kind: WorkEdgeKind) -> Result<()> {
    tx.prepare_cached("INSERT OR IGNORE INTO work_edge(from_pk, to_pk, kind) VALUES (?1, ?2, ?3)")?
        .execute(params![from, to, kind.as_str()])?;
    Ok(())
}

#[cfg(test)]
mod tests;
