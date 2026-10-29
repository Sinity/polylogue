//! A title for a session whose records name none: the first line of the
//! first prompt a person typed.
//! Kept apart from the recorded title, which always wins when present and
//! is never replaced by this display fallback in exports.

use super::{Derivation, Version};
use crate::Result;
use rusqlite::{OptionalExtension, Transaction, params};

pub(super) const DERIVATION: Derivation = Derivation {
    name: "title",
    products: &[],
    version: Version::Rules(4),
    after: &["duplicates", "session"],
    run: |tx, pk, _| run(tx, pk),
    finish: None,
};

/// Characters kept of the prompt's first line.
const LENGTH: usize = 80;

fn run(tx: &Transaction<'_>, session_pk: i64) -> Result<()> {
    // The first text block a person typed as prose, in reading order.
    let first: Option<i64> = tx
        .prepare_cached(
            "SELECT b.pk FROM message m JOIN block b ON b.message_pk=m.pk
             WHERE m.session_pk=?1 AND m.duplicate_of IS NULL AND m.written_by='human'
               AND COALESCE(m.prompt_kind, 'prose')='prose' AND b.kind='text'
             ORDER BY m.order_input_id, m.seq, m.record_position, m.id, b.idx LIMIT 1",
        )?
        .query_row([session_pk], |r| r.get(0))
        .optional()?;
    let title = match first {
        None => None,
        Some(pk) => crate::ordered_blocks::one(tx, pk)?
            .text()
            .and_then(title_of),
    };
    tx.prepare_cached(
        "UPDATE session SET derived_title = ?2 WHERE pk = ?1 AND derived_title IS NOT ?2",
    )?
    .execute(params![session_pk, title])?;
    Ok(())
}

pub(crate) fn title_of(prompt: &str) -> Option<String> {
    let line = prompt.lines().map(str::trim).find(|l| !l.is_empty())?;
    let mut t: String = line.chars().take(LENGTH).collect();
    if line.chars().count() > LENGTH {
        t.push('…');
    }
    Some(t)
}
