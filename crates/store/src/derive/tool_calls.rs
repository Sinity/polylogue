//! A shell call's normalized command (analysis.md), and the outcome reason
//! a call without a result earns once its session moved on. The call's
//! action comes from its origin's parser.

use super::{Derivation, Version};
use crate::Result;
use rusqlite::{Transaction, params};

pub(super) const DERIVATION: Derivation = Derivation {
    name: "tool_calls",
    version: Version::Rules(3),
    after: &[],
    run: |tx, pk, _| run(tx, pk),
    finish: None,
};

fn run(tx: &Transaction<'_>, session_pk: i64) -> Result<()> {
    let calls: Vec<(i64, Option<String>)> = tx
        .prepare_cached(
            "SELECT pk, command FROM tool_call WHERE session_pk = ?1 AND action = 'shell'",
        )?
        .query_map([session_pk], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    let mut set = tx.prepare_cached(
        "UPDATE tool_call SET command_norm = ?2 WHERE pk = ?1 AND command_norm IS NOT ?2",
    )?;
    for (pk, command) in &calls {
        let norm = command
            .as_deref()
            .and_then(polylogue_model::command::normalize);
        set.execute(params![pk, norm])?;
    }
    passed_without_result(tx, session_pk)
}

/// Model 6 (outcome reasons): a call still without a result once a later
/// model response exists in its session never gets one, since a harness
/// answers only after every call of the earlier response has its result.
/// The reason is a fact of the records; whether a `no_result` call is
/// pending is decided at read time from live state.
fn passed_without_result(tx: &Transaction<'_>, session_pk: i64) -> Result<()> {
    tx.prepare_cached(
        "UPDATE tool_call SET outcome_reason = CASE WHEN EXISTS (
             SELECT 1 FROM block cb JOIN current_message cm ON cm.pk = cb.message_pk
             JOIN current_message later ON later.session_pk = cm.session_pk
             WHERE cb.pk = tool_call.call_block_pk AND later.role = 'assistant' AND later.written_by='model'
               AND coalesce(json_extract(later.attrs_json,'$.production.response_boundary_unknown'),0)=0
               AND coalesce(json_extract(cm.attrs_json,'$.production.response_boundary_unknown'),0)=0
               AND (later.order_input_id, later.seq) > (cm.order_input_id, cm.seq))
           THEN 'passed_without_result' ELSE 'no_result' END
         WHERE session_pk = ?1 AND result_block_pk IS NULL AND outcome = 'unknown'
           AND outcome_reason IN ('no_result', 'passed_without_result')",
    )?
    .execute([session_pk])?;
    Ok(())
}
