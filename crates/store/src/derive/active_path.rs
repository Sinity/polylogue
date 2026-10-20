//! Which of a branched session's messages lie on the path from its newest
//! active leaf to the root (`on_active_path` 1) and which on an abandoned
//! branch (0). A session whose source names no leaf keeps NULL: unknown,
//! read as on the path.

use super::{Derivation, Version};
use crate::Result;
use rusqlite::Transaction;

pub(super) const DERIVATION: Derivation = Derivation {
    name: "active_path",
    version: Version::Rules(1),
    after: &[],
    run: |tx, pk, _| run(tx, pk),
    finish: None,
};

fn run(tx: &Transaction<'_>, session_pk: i64) -> Result<()> {
    tx.prepare_cached(
        "WITH RECURSIVE
           leaf(id) AS (
             SELECT m.id FROM current_event e JOIN current_message m ON m.pk = e.after_message_pk
             WHERE e.session_pk = ?1 AND e.kind = 'active_leaf'
             ORDER BY e.pk DESC LIMIT 1),
           path(id) AS (
             SELECT id FROM leaf
             UNION
             SELECT m.parent_id FROM current_message m JOIN path ON m.id = path.id
             WHERE m.parent_id IS NOT NULL),
           marked(pk, v) AS (
             SELECT pk, CASE WHEN EXISTS (SELECT 1 FROM leaf)
                             THEN (id IN (SELECT id FROM path)) END
             FROM current_message WHERE session_pk = ?1)
         UPDATE message SET on_active_path = marked.v FROM marked
         WHERE message.pk = marked.pk AND message.on_active_path IS NOT marked.v",
    )?
    .execute([session_pk])?;
    Ok(())
}
