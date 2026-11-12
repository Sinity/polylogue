//! Session links from tool calls, found by their structure: the commits and
//! pull requests a session's commands made (from the output of `git commit` and
//! `gh pr create`). A session's live task never comes from here: a command's
//! name or arguments establish no task (`polylogue_model::live`).
//!
//! A fold: the transaction that writes a call's message or its result adds
//! the links of the calls it finished ([`fold`]); one that rewrites a stored
//! message of the session marks it due, and the derive step derives the
//! session's links again.

use crate::Result;
use crate::encoding::{unpack_block, unpack_text};
use polylogue_model::Block;
use polylogue_model::command::{Stage, effective_stages};
use rusqlite::{Connection, Transaction, params};

pub(super) const DERIVATION: super::Derivation = super::Derivation {
    name: "links",
    class: super::Class::Fold,
    // Bumped when the rules change; every session is derived again.
    products: &[],
    version: super::Version::Rules(3),
    // Reads tool calls' commands and outcomes.
    after: &["tool_calls"],
    dirtied_by: &[super::Cause::MessagesRewritten],
    compute: |conn, session_pk, _| compute(conn, session_pk).map(super::Output::Links),
};

/// The links some of a session's calls made, in the order the calls were
/// read.
#[derive(Debug, Default)]
pub(crate) struct Links {
    /// Each commit hash a call reported, with the call's key.
    commits: Vec<(String, i64)>,
    /// Each pull request a call created: `(owner/repo, number, url)`.
    pulls: Vec<(String, i64, String)>,
}

/// The stored output of a call. Only commands that make a commit or pull
/// request read it.
fn result(conn: &Connection, call_pk: i64) -> Result<Option<String>> {
    let (text, body): (Option<Vec<u8>>, Option<Vec<u8>>) = conn
        .prepare_cached(
            "SELECT rt.body, rv.body FROM tool_call t
             LEFT JOIN block rb ON rb.pk = t.result_block_pk
             LEFT JOIN block_value rv ON rv.pk=rb.value_pk
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

/// The links of every finished call of the session.
fn compute(conn: &Connection, session_pk: i64) -> Result<Links> {
    let calls = conn
        .prepare_cached(
            "SELECT t.pk, t.command FROM tool_call t
             WHERE t.session_pk = ?1 AND t.outcome = 'ok' AND t.command IS NOT NULL
             ORDER BY t.at IS NULL, t.at, t.pk",
        )?
        .query_map([session_pk], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    links_of(conn, calls)
}

/// Brings the links of every session the transaction changed (`sessions`)
/// up to date in that transaction: the finished calls whose call or result
/// block is in one of its new messages add their links. A session with a
/// rewritten message is marked due instead; the derive step derives its
/// links again.
pub(crate) fn fold(
    tx: &Transaction<'_>,
    sessions: &std::collections::BTreeMap<i64, crate::change_summary::SessionChange>,
) -> Result<()> {
    for (&session_pk, change) in sessions {
        let exists: bool = tx
            .prepare_cached("SELECT EXISTS(SELECT 1 FROM session WHERE pk = ?1)")?
            .query_row([session_pk], |r| r.get(0))?;
        if !exists {
            continue;
        }
        if change.whole || change.rewritten {
            tx.prepare_cached(
                "INSERT INTO derive_due(session_pk, rows_changed, rewritten) VALUES (?1, 1, 1)
                 ON CONFLICT(session_pk) DO UPDATE SET rewritten = 1, marked = marked + 1",
            )?
            .execute([session_pk])?;
            continue;
        }
        let mut calls = Vec::new();
        let mut finished = tx.prepare_cached(
            "SELECT t.pk, t.command FROM block b JOIN tool_call t
                 ON t.call_block_pk = b.pk OR t.result_block_pk = b.pk
             WHERE b.message_pk = ?1 AND t.session_pk = ?2
               AND t.outcome = 'ok' AND t.command IS NOT NULL",
        )?;
        for message_pk in &change.messages {
            for call in finished.query_map(rusqlite::params![message_pk, session_pk], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
            })? {
                calls.push(call?);
            }
        }
        calls.sort_unstable();
        calls.dedup();
        add(tx, session_pk, &links_of(tx, calls)?)?;
    }
    Ok(())
}

/// The links the finished `calls` (call key and command) made.
fn links_of(conn: &Connection, calls: Vec<(i64, String)>) -> Result<Links> {
    let mut links = Links::default();
    for (call_pk, command) in calls {
        let stages = effective_stages(&command);
        let output = if stages.iter().any(makes_artifact) {
            result(conn, call_pk)?
        } else {
            None
        };
        for stage in &stages {
            match (stage.program.as_str(), stage.subcommand()) {
                ("git", Some("commit")) => {
                    for hash in committed(output.as_deref().unwrap_or("")) {
                        links.commits.push((hash, call_pk));
                    }
                }
                ("gh", Some("pr")) if is_create(stage) => {
                    links
                        .pulls
                        .extend(pull_requests(output.as_deref().unwrap_or("")));
                }
                _ => {}
            }
        }
    }
    Ok(links)
}

/// Adds `links` to the session's.
fn add(tx: &Transaction<'_>, session_pk: i64, links: &Links) -> Result<()> {
    for (hash, call_pk) in &links.commits {
        // The evidence is the first call in session order to report the
        // hash, whichever order the calls arrived in.
        tx.prepare_cached(
            "INSERT INTO commit_link(session_pk, commit_hash, evidence_block_pk, evidence)
             SELECT ?1, ?2, call_block_pk, 'commit-output' FROM tool_call WHERE pk = ?3
             ON CONFLICT(session_pk, commit_hash) DO UPDATE
               SET evidence_block_pk = excluded.evidence_block_pk
             WHERE commit_link.evidence = 'commit-output' AND EXISTS (
               SELECT 1 FROM tool_call new, tool_call old
               WHERE new.call_block_pk = excluded.evidence_block_pk
                 AND old.call_block_pk = commit_link.evidence_block_pk
                 AND (new.at IS NULL, ifnull(new.at, 0), new.pk)
                   < (old.at IS NULL, ifnull(old.at, 0), old.pk))",
        )?
        .execute(params![session_pk, hash, call_pk])?;
    }
    for (repo, n, url) in &links.pulls {
        tx.prepare_cached(
            "INSERT OR IGNORE INTO pr_link(session_pk, repo, number, url, evidence)
             VALUES (?1, ?2, ?3, ?4, 'gh-output')",
        )?
        .execute(params![session_pk, repo, n, url])?;
    }
    Ok(())
}

/// Replaces the links this step derived for the session with `links`.
pub(super) fn replace(tx: &Transaction<'_>, session_pk: i64, links: &Links) -> Result<()> {
    clear(tx, session_pk)?;
    add(tx, session_pk, links)
}

fn is_create(stage: &Stage) -> bool {
    stage.operands().first().is_some_and(|o| o == "create")
}

/// Only artifact-producing commands read their output.
fn makes_artifact(stage: &Stage) -> bool {
    (stage.program == "git" && stage.subcommand() == Some("commit"))
        || (stage.program == "gh" && stage.subcommand() == Some("pr") && is_create(stage))
}

/// Removes the links this step derived for the session.
fn clear(tx: &Transaction<'_>, session_pk: i64) -> Result<()> {
    tx.prepare_cached(
        "DELETE FROM commit_link WHERE session_pk = ?1 AND evidence = 'commit-output'",
    )?
    .execute([session_pk])?;
    tx.prepare_cached("DELETE FROM pr_link WHERE session_pk = ?1 AND evidence = 'gh-output'")?
        .execute([session_pk])?;
    Ok(())
}

#[cfg(test)]
mod tests;
