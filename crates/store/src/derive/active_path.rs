//! Which of a branched session's messages lie on the path from its newest
//! active leaf to the root (`on_active_path` 1) and which on an abandoned
//! branch (0). An explicit empty-path marker puts every message off the path.
//! Without a valid path marker activity stays NULL: unknown, read as on the path.
//!
//! Repeated markers share an event row. Revision, snapshot and record order
//! choose each input's latest marker; known event time and content identity
//! resolve different inputs without depending on row insertion order.

use super::{Derivation, Version};
use crate::Result;
use rusqlite::Transaction;

pub(super) const DERIVATION: Derivation = Derivation {
    name: "active_path",
    version: Version::Rules(5),
    after: &[],
    run: |tx, pk, _| run(tx, pk),
    finish: None,
};

fn run(tx: &Transaction<'_>, session_pk: i64) -> Result<()> {
    tx.prepare_cached(
        "WITH RECURSIVE
           observations AS (
             SELECT m.id, e.kind, e.at, e.id AS event_id, r.id AS input_id,
                    row_number() OVER (
                      PARTITION BY r.id ORDER BY
                        e.source_chain DESC, e.seq DESC, e.id
                    ) AS newest
             FROM event e LEFT JOIN message m ON m.pk = e.after_message_pk
             JOIN input_ref r ON r.pk = e.input_pk
             WHERE e.session_pk = ?1 AND r.excised=0
               AND ((e.kind = 'active_leaf' AND m.pk IS NOT NULL)
                 OR (e.kind IN ('active_path_empty','active_path_unknown') AND e.after_message_pk IS NULL))),
           marker(id, kind) AS (
             SELECT id, kind FROM observations WHERE newest = 1
             ORDER BY at DESC, event_id, input_id LIMIT 1),
           path(id) AS (
             SELECT id FROM marker WHERE kind = 'active_leaf' AND id IS NOT NULL
             UNION
             SELECT m.parent_id FROM message m JOIN path ON m.id = path.id
             WHERE m.parent_id IS NOT NULL),
           marked(pk, v) AS (
             SELECT pk, CASE WHEN EXISTS (SELECT 1 FROM marker WHERE kind <> 'active_path_unknown')
                             THEN (id IN (SELECT id FROM path)) END
             FROM message WHERE session_pk = ?1)
         UPDATE message SET on_active_path = marked.v FROM marked
         WHERE message.pk = marked.pk AND message.on_active_path IS NOT marked.v",
    )?
    .execute([session_pk])?;
    Ok(())
}
