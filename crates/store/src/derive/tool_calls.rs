//! What each tool call is: its kind of action, its normalized command
//! (analysis.md), and the outcome reason a call without a result earns once
//! its session moved on.

use super::Derivation;
use crate::Result;
use rusqlite::{Transaction, params};

pub(super) const DERIVATION: Derivation = Derivation {
    name: "tool_calls",
    version: 1,
    after: &[],
    run,
};

fn run(tx: &Transaction<'_>, session_pk: i64) -> Result<()> {
    let calls: Vec<(i64, String, Option<String>)> = tx
        .prepare_cached("SELECT pk, tool, command FROM tool_call WHERE session_pk = ?1")?
        .query_map([session_pk], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    let mut set = tx.prepare_cached(
        "UPDATE tool_call SET action = ?2, command_norm = ?3
         WHERE pk = ?1 AND (action IS NOT ?2 OR command_norm IS NOT ?3)",
    )?;
    for (pk, tool, command) in &calls {
        let action = action(tool);
        let norm = (action == "shell")
            .then(|| {
                command
                    .as_deref()
                    .and_then(polylogue_model::command::normalize)
            })
            .flatten();
        set.execute(params![pk, action, norm])?;
    }
    passed_without_result(tx, session_pk)
}

/// The kind of action a tool performs, from its name.
pub(crate) fn action(tool: &str) -> &'static str {
    match tool {
        "Bash" | "BashOutput" | "KillShell" | "shell" | "exec_command" | "local_shell" => "shell",
        "Read" | "NotebookRead" => "file_read",
        "Write" => "file_write",
        "Edit" | "MultiEdit" | "NotebookEdit" | "apply_patch" => "file_edit",
        "Grep" | "Glob" | "LS" | "ToolSearch" => "search",
        "WebFetch" | "WebSearch" | "web_search" => "web",
        "Task" | "Agent" => "subagent",
        // Codex code mode: a script whose `exec_command` calls are counted
        // in its place.
        "exec" => "code",
        t if t.starts_with("mcp__") => "mcp",
        _ => "other",
    }
}

/// Model 6 (outcome reasons): a call still without a result once a later
/// model response exists in its session never gets one, since a harness
/// answers only after every call of the earlier response has its result.
/// The reason is a fact of the records; whether a `no_result` call is
/// pending is decided at read time from live state.
fn passed_without_result(tx: &Transaction<'_>, session_pk: i64) -> Result<()> {
    tx.prepare_cached(
        "UPDATE tool_call SET outcome_reason = CASE WHEN EXISTS (
             SELECT 1 FROM block cb JOIN message cm ON cm.pk = cb.message_pk
             JOIN message later ON later.session_pk = cm.session_pk
             WHERE cb.pk = tool_call.call_block_pk AND later.role = 'assistant'
               AND (later.order_input_pk, later.seq) > (cm.order_input_pk, cm.seq))
           THEN 'passed_without_result' ELSE 'no_result' END
         WHERE session_pk = ?1 AND result_block_pk IS NULL AND outcome = 'unknown'
           AND outcome_reason IN ('no_result', 'passed_without_result')",
    )?
    .execute([session_pk])?;
    Ok(())
}
