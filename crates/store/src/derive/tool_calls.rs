//! A shell call's normalized command (a row value the writer writes with the
//! call), and the outcome reason a call without a result earns once its
//! session moved on. The call's action comes from its origin's parser.

use super::{Class, Derivation, Output, Version};
use crate::Result;
use rusqlite::{Connection, Transaction, params};

pub(super) const DERIVATION: Derivation = Derivation {
    name: "tool_calls",
    class: Class::Session,
    products: &[],
    version: Version::Rules(3),
    after: &["command_norm"],
    dirtied_by: &[],
    compute: |conn, pk, _| passed_without_result(conn, pk).map(Output::ToolCalls),
};

pub(super) const COMMAND_NORM: Derivation = Derivation {
    name: "command_norm",
    class: Class::Row,
    products: &[],
    version: Version::Rules(1),
    after: &[],
    dirtied_by: &[],
    compute: |conn, pk, _| normalize(conn, pk).map(Output::CommandNorm),
};

/// A session's shell calls. The unary `+` keeps the planner off
/// `tool_call_action`: one action can hold most of the archive's calls, and
/// a lookup by it would visit them all for every session.
pub(super) const SHELL_CALLS: &str =
    "SELECT pk, command FROM tool_call WHERE session_pk = ?1 AND +action = 'shell'";

/// Every shell call of the session, with its normalized command.
fn normalize(conn: &Connection, session_pk: i64) -> Result<Vec<(i64, Option<String>)>> {
    let calls: Vec<(i64, Option<String>)> = conn
        .prepare_cached(SHELL_CALLS)?
        .query_map([session_pk], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(calls
        .into_iter()
        .map(|(pk, command)| {
            let norm = command
                .as_deref()
                .and_then(polylogue_model::command::normalize);
            (pk, norm)
        })
        .collect())
}

pub(super) fn apply_command_norm(
    tx: &Transaction<'_>,
    calls: &[(i64, Option<String>)],
) -> Result<()> {
    let mut set = tx.prepare_cached(
        "UPDATE tool_call SET command_norm = ?2 WHERE pk = ?1 AND command_norm IS NOT ?2",
    )?;
    for (pk, norm) in calls {
        set.execute(params![pk, norm])?;
    }
    Ok(())
}

/// A call still without a result once a later model response exists in its
/// session never gets one, since a harness answers only after every call of
/// the earlier response has its result.
/// The reason is a fact of the records; whether a `no_result` call is
/// pending is decided at read time from live state. Returns the calls whose
/// reason changes, with the new reason.
fn passed_without_result(conn: &Connection, session_pk: i64) -> Result<Vec<(i64, &'static str)>> {
    let calls: Vec<(i64, bool, String)> = conn
        .prepare_cached(
            "SELECT pk, EXISTS (
                 SELECT 1 FROM block cb JOIN message cm ON cm.pk = cb.message_pk
                 JOIN message later ON later.session_pk = cm.session_pk
                 WHERE cb.pk = tool_call.call_block_pk AND later.role = 'assistant' AND later.written_by='model'
                   AND coalesce(json_extract(later.attrs_json,'$.production.response_boundary_unknown'),0)=0
                   AND coalesce(json_extract(cm.attrs_json,'$.production.response_boundary_unknown'),0)=0
                   AND later.session_order > cm.session_order),
                 outcome_reason
             FROM tool_call
             WHERE session_pk = ?1 AND result_block_pk IS NULL AND outcome = 'unknown'
               AND outcome_reason IN ('no_result', 'passed_without_result')",
        )?
        .query_map([session_pk], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(calls
        .into_iter()
        .filter_map(|(pk, passed, reason)| {
            let wanted = if passed {
                "passed_without_result"
            } else {
                "no_result"
            };
            (reason != wanted).then_some((pk, wanted))
        })
        .collect())
}

pub(super) fn apply_reasons(tx: &Transaction<'_>, calls: &[(i64, &'static str)]) -> Result<()> {
    let mut set = tx.prepare_cached("UPDATE tool_call SET outcome_reason = ?2 WHERE pk = ?1")?;
    for (pk, reason) in calls {
        set.execute(params![pk, reason])?;
    }
    Ok(())
}
