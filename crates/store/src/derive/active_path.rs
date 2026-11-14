//! Which of a branched session's messages lie on the path from its newest
//! active leaf to the root (`on_active_path` 1) and which on an abandoned
//! branch (0). An explicit empty-path marker puts every message off the path.
//! Without a valid path marker activity stays NULL: unknown, read as on the path.
//!
//! Repeated markers share an event row. Revision, capture and record order
//! choose each input's latest marker; known event time and content identity
//! resolve different inputs without depending on row insertion order.

use super::{Class, Derivation, Output, Version};
use crate::Result;
use rusqlite::{Connection, Transaction};

pub(super) const DERIVATION: Derivation = Derivation {
    name: "active_path",
    class: Class::Session,
    products: &[],
    version: Version::Rules(5),
    after: &[],
    dirtied_by: &[],
    compute: |conn, pk, _| compute(conn, pk).map(Output::ActivePath),
};

/// The session's messages whose place on the path changes, with their new
/// place.
fn compute(conn: &Connection, session_pk: i64) -> Result<Vec<(i64, Option<i64>)>> {
    Ok(conn
        .prepare_cached(
            "WITH RECURSIVE
           observations AS (
             SELECT m.id, e.kind, e.at, e.id AS event_id, r.id AS input_id,
                    row_number() OVER (
                      PARTITION BY r.id ORDER BY
                        e.source_chain DESC, e.seq DESC, e.id
                    ) AS newest
             FROM event e LEFT JOIN message m ON m.pk = e.after_message_pk
             JOIN input_ref r ON r.pk = e.input_pk
             WHERE e.session_pk = ?1
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
           marked(pk, was, v) AS (
             SELECT pk, on_active_path,
                    CASE WHEN EXISTS (SELECT 1 FROM marker WHERE kind <> 'active_path_unknown')
                         THEN (id IN (SELECT id FROM path)) END
             FROM message WHERE session_pk = ?1)
         SELECT pk, v FROM marked WHERE was IS NOT v",
        )?
        .query_map([session_pk], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?)
}

pub(super) fn apply(
    tx: &Transaction<'_>,
    moved: &[(i64, Option<i64>)],
    changes: &crate::change_summary::ChangeSummary,
) -> Result<()> {
    let mut set = tx.prepare_cached("UPDATE message SET on_active_path = ?2 WHERE pk = ?1")?;
    for (pk, place) in moved {
        // The session and parent stay as they were; only the place changed.
        changes.message_metadata(tx, *pk)?;
        set.execute(rusqlite::params![pk, place])?;
    }
    Ok(())
}
