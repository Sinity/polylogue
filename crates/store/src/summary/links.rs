//! Session links from tool calls, found by their structure: the commits and
//! pull requests a session's commands made (from the output of `git commit` and
//! `gh pr create`). A session's live task never comes from here: a command's
//! name or arguments establish no task (`polylogue_model::live`).
//!
//! A fold over the session's finished calls in session order: a commit's
//! evidence is the first call in that order to report it. The fold keeps no
//! state: each run finds the links of the rows it reads, and its write
//! replaces the links found after its start (`row_key`), so a link found
//! before keeps its earlier evidence.

use super::Output;
use super::fold::{Context, Flow, Fold, Key, SessionRow};
use crate::Result;
use polylogue_model::Block;
use polylogue_model::command::{Stage, effective_stages};
use rusqlite::{Connection, Transaction, params};
use std::collections::BTreeMap;

pub(crate) struct Links;

/// The links the calls of a run's rows made, each with the row that first
/// reported it.
#[derive(Debug, Default, Clone)]
pub(crate) struct Found {
    /// Each commit hash a call reported, with the first such call's block.
    commits: BTreeMap<String, (i64, Key)>,
    /// Each pull request a call created: `(owner/repo, number, url)`.
    pulls: BTreeMap<(String, i64, String), Key>,
}

impl Found {
    #[cfg(test)]
    pub(crate) fn commits(&self) -> &BTreeMap<String, (i64, Key)> {
        &self.commits
    }
}

impl Fold for Links {
    type State = ();
    const NAME: &'static str = "links";
    const VERSION: i64 = 5;

    fn step(_: &mut (), row: &SessionRow, cx: &Context<'_>, out: &mut Output) -> Result<Flow> {
        let SessionRow::Message(m) = row else {
            return Ok(Flow::Continue);
        };
        let state = out.links.get_or_insert_default();
        for call in &m.calls {
            let Some(command) = call.command.as_deref().filter(|_| call.outcome == "ok") else {
                continue;
            };
            let stages = effective_stages(command);
            let output = if stages.iter().any(makes_artifact) {
                result(cx.conn, call.pk)?
            } else {
                None
            };
            for stage in &stages {
                match (stage.program.as_str(), stage.subcommand()) {
                    ("git", Some("commit")) => {
                        for hash in committed(output.as_deref().unwrap_or("")) {
                            state
                                .commits
                                .entry(hash)
                                .or_insert((call.block_pk, m.key.clone()));
                        }
                    }
                    ("gh", Some("pr")) if is_create(stage) => {
                        for pull in pull_requests(output.as_deref().unwrap_or("")) {
                            state.pulls.entry(pull).or_insert_with(|| m.key.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(Flow::Continue)
    }

    fn end(_: &(), _: &Context<'_>, out: &mut Output) -> Result<()> {
        out.links.get_or_insert_default();
        Ok(())
    }
}

/// The stored output of a call. Only commands that make a commit or pull
/// request read it.
fn result(conn: &Connection, call_pk: i64) -> Result<Option<String>> {
    let (text, piece): (Option<String>, Option<crate::content::Piece>) = conn
        .prepare_cached(
            "SELECT rg.body, rt.start, rt.length, rb.content_group, rb.content_start,
                 rb.content_length, rb.data_length FROM tool_call t
             LEFT JOIN message rm ON rm.id = t.result_message_id
             LEFT JOIN block rb ON rb.message_pk = rm.pk AND rb.idx = t.result_idx
             LEFT JOIN text rt ON rt.id = rb.text_id
             LEFT JOIN content_group rg ON rg.id = rt.group_id WHERE t.pk = ?1",
        )?
        .query_and_then([call_pk], |r| {
            let piece = match r.get::<_, Option<i64>>(3)? {
                Some(_) => Some(crate::content::Piece::of(r, 3)?),
                None => None,
            };
            Ok::<_, crate::StoreError>((crate::text_rows::column(r, 0)?, piece))
        })?
        .next()
        .ok_or(rusqlite::Error::QueryReturnedNoRows)??;
    match (text, piece) {
        (Some(text), _) => Ok(Some(text)),
        (None, Some(piece)) => Ok(
            match crate::ordered_blocks::body(
                conn,
                &mut crate::content::Reader::default(),
                piece,
                false,
            )? {
                Block::ToolResult { output, .. } => output,
                _ => None,
            },
        ),
        (None, None) => Ok(None),
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
/// The URL is the pull request's own, so a link does not depend on which
/// mention of it (a `/files` suffix, trailing punctuation) was read first.
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
            (pull == "pull").then(|| {
                let repo = format!("{owner}/{repo}");
                let url = format!("https://github.com/{repo}/pull/{n}");
                (repo, n, url)
            })
        })
        .collect()
}

/// Replaces the links this summary found for the session after `after`
/// (the row a run started after) with `found`; those found before stay,
/// and so does a link another source states (a commit trailer).
pub(super) fn apply(
    tx: &Transaction<'_>,
    session_pk: i64,
    after: &Key,
    found: &Found,
) -> Result<()> {
    tx.prepare_cached(
        "DELETE FROM commit_link
         WHERE session_pk = ?1 AND evidence = 'commit-output' AND row_key > ?2",
    )?
    .execute(params![session_pk, after.0])?;
    tx.prepare_cached(
        "DELETE FROM pull_request_link WHERE session_pk = ?1 AND evidence = 'gh-output' AND row_key > ?2",
    )?
    .execute(params![session_pk, after.0])?;
    let mut commit = tx.prepare_cached(
        "INSERT INTO commit_link(session_pk, commit_hash, evidence_block_pk, evidence, row_key)
         VALUES (?1, ?2, ?3, 'commit-output', ?4) ON CONFLICT(session_pk, commit_hash) DO NOTHING",
    )?;
    for (hash, (block, at)) in &found.commits {
        commit.execute(params![session_pk, hash, block, at.0])?;
    }
    let mut pull = tx.prepare_cached(
        "INSERT OR IGNORE INTO pull_request_link(session_pk, repo, number, url, evidence, row_key)
         VALUES (?1, ?2, ?3, ?4, 'gh-output', ?5)",
    )?;
    for ((repo, n, url), at) in &found.pulls {
        pull.execute(params![session_pk, repo, n, url, at.0])?;
    }
    Ok(())
}

fn is_create(stage: &Stage) -> bool {
    stage.operands().first().is_some_and(|o| o == "create")
}

/// Only artifact-producing commands read their output.
fn makes_artifact(stage: &Stage) -> bool {
    (stage.program == "git" && stage.subcommand() == Some("commit"))
        || (stage.program == "gh" && stage.subcommand() == Some("pr") && is_create(stage))
}

#[cfg(test)]
mod tests;
