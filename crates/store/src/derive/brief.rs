//! What a session-start brief says about a session: its last answer, the
//! files it changed, its commits and pull requests, and whether it ended on
//! an unanswered prompt or an unfinished tool call. Kept per session so a
//! brief reads one row for each session it lists, however long the session.

use super::{Derivation, Version};
use crate::Result;
use rusqlite::{OptionalExtension, Transaction, params};

/// How many changed files a brief names; the rest are counted.
pub(crate) const RECENT_FILES: usize = 4;

pub(super) const DERIVATION: Derivation = Derivation {
    name: "brief",
    products: &[],
    version: Version::Rules(1),
    // Reads duplicates, tool-call outcomes and the commit and pull request
    // links.
    after: &["duplicates", "tool_calls", "links"],
    run: |tx, pk, _| run(tx, pk),
    finish: None,
};

pub(crate) fn run(tx: &Transaction<'_>, session_pk: i64) -> Result<()> {
    let last_answer: Option<i64> = tx
        .prepare_cached(
            "SELECT m.pk FROM message m
              WHERE m.session_pk = ?1 AND m.written_by = 'model' AND m.duplicate_of IS NULL
                AND EXISTS (SELECT 1 FROM block b WHERE b.message_pk = m.pk AND b.kind = 'text')
              ORDER BY m.order_input_id DESC, m.seq DESC, m.record_position DESC, m.id DESC
              LIMIT 1",
        )?
        .query_row([session_pk], |r| r.get(0))
        .optional()?;
    let files: Vec<String> = tx
        .prepare_cached(
            "SELECT path FROM tool_call
              WHERE session_pk = ?1 AND action IN ('file_write', 'file_edit')
                AND outcome = 'ok' AND path IS NOT NULL
              GROUP BY path ORDER BY max(at) DESC, path",
        )?
        .query_map([session_pk], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let commits: Vec<String> = tx
        .prepare_cached(
            "SELECT commit_hash FROM commit_link WHERE session_pk = ?1 ORDER BY commit_hash",
        )?
        .query_map([session_pk], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let pull_requests: Vec<String> = tx
        .prepare_cached(
            "SELECT repo, number FROM pr_link WHERE session_pk = ?1 ORDER BY repo, number",
        )?
        .query_map([session_pk], |r| {
            Ok(format!(
                "{}#{}",
                r.get::<_, String>(0)?,
                r.get::<_, i64>(1)?
            ))
        })?
        .collect::<rusqlite::Result<_>>()?;
    let (last_by, last_kind): (Option<String>, Option<String>) = tx
        .prepare_cached(
            "SELECT m.written_by, m.prompt_kind FROM message m
              WHERE m.session_pk = ?1 AND m.duplicate_of IS NULL
              ORDER BY m.order_input_id DESC, m.seq DESC, m.record_position DESC, m.id DESC
              LIMIT 1",
        )?
        .query_row([session_pk], |r| Ok((r.get(0)?, r.get(1)?)))
        .optional()?
        .unwrap_or_default();
    let unfinished_tool_call: bool = tx
        .prepare_cached(
            "SELECT coalesce((SELECT c.result_block_pk IS NULL FROM tool_call c
                               WHERE c.session_pk = ?1 ORDER BY c.pk DESC LIMIT 1), 0)",
        )?
        .query_row([session_pk], |r| r.get(0))?;
    let unanswered_prompt =
        last_by.as_deref() == Some("human") && last_kind.as_deref() != Some("command");
    tx.prepare_cached(
        "INSERT INTO session_brief(session_pk, last_answer_pk, files_changed, recent_files_json,
             commits_json, pull_requests_json, unanswered_prompt, unfinished_tool_call)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
         ON CONFLICT(session_pk) DO UPDATE SET last_answer_pk = excluded.last_answer_pk,
             files_changed = excluded.files_changed, recent_files_json = excluded.recent_files_json,
             commits_json = excluded.commits_json, pull_requests_json = excluded.pull_requests_json,
             unanswered_prompt = excluded.unanswered_prompt,
             unfinished_tool_call = excluded.unfinished_tool_call",
    )?
    .execute(params![
        session_pk,
        last_answer,
        i64::try_from(files.len()).unwrap_or(i64::MAX),
        serde_json::to_string(&files[..files.len().min(RECENT_FILES)])?,
        serde_json::to_string(&commits)?,
        serde_json::to_string(&pull_requests)?,
        unanswered_prompt,
        unfinished_tool_call,
    ])?;
    Ok(())
}
