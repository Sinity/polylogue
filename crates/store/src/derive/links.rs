//! Session links from tool calls, found by their structure: the commits and
//! pull requests a session's commands made (from the output of `git commit` and
//! `gh pr create`). A session's live task never comes from here: a command's
//! name or arguments establish no task (`polylogue_model::live`).

use crate::Result;
use crate::encoding::{unpack_block, unpack_text};
use polylogue_model::Block;
use polylogue_model::command::{Stage, effective_stages};
use rusqlite::{Transaction, params};

pub(crate) const DERIVATION: super::Derivation = super::Derivation {
    name: "links",
    // Bumped when the rules change; every session is derived again.
    products: &[],
    version: super::Version::Rules(2),
    // Reads tool calls' commands and outcomes.
    after: &["tool_calls"],
    run: |tx, session_pk, _| derive_session(tx, session_pk),
    finish: None,
};

/// The stored output of a call. Only commands that make a commit or pull
/// request read it.
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

fn derive_session(tx: &Transaction<'_>, session_pk: i64) -> Result<()> {
    clear(tx, session_pk)?;
    let mut stmt = tx.prepare_cached(
        "SELECT t.pk, t.command FROM tool_call t
         WHERE t.session_pk = ?1 AND t.outcome = 'ok' AND t.command IS NOT NULL
         ORDER BY t.at IS NULL, t.at, t.pk",
    )?;
    let calls = stmt
        .query_map([session_pk], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for (call_pk, command) in calls {
        let stages = effective_stages(&command);
        let output = if stages.iter().any(makes_artifact) {
            result(tx, call_pk)?
        } else {
            None
        };
        for stage in &stages {
            match (stage.program.as_str(), stage.subcommand()) {
                ("git", Some("commit")) => {
                    for hash in committed(output.as_deref().unwrap_or("")) {
                        tx.prepare_cached(
                            "INSERT OR IGNORE INTO commit_link(session_pk, commit_hash, evidence_block_pk, evidence)
                             SELECT ?1, ?2, call_block_pk, 'commit-output' FROM tool_call WHERE pk = ?3",
                        )?
                        .execute(params![session_pk, hash, call_pk])?;
                    }
                }
                ("gh", Some("pr")) if is_create(stage) => {
                    for (repo, n, url) in pull_requests(output.as_deref().unwrap_or("")) {
                        tx.prepare_cached(
                            "INSERT OR IGNORE INTO pr_link(session_pk, repo, number, url, evidence)
                             VALUES (?1, ?2, ?3, ?4, 'gh-output')",
                        )?
                        .execute(params![session_pk, repo, n, url])?;
                    }
                }
                _ => {}
            }
        }
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
