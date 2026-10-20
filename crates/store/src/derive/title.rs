//! A title for a session whose records name none: the first line of the
//! first prompt a person typed. Kept apart from the recorded title, which
//! always wins when there is one, and never exported as if recorded.

use super::{Derivation, Version};
use crate::Result;
use rusqlite::{OptionalExtension, Transaction, params};

pub(super) const DERIVATION: Derivation = Derivation {
    name: "title",
    version: Version::Rules(1),
    after: &["session"],
    run: |tx, pk, _| run(tx, pk),
    finish: None,
};

/// Characters kept of the prompt's first line.
const LENGTH: usize = 80;

fn run(tx: &Transaction<'_>, session_pk: i64) -> Result<()> {
    let body: Option<Vec<u8>> = tx
        .prepare_cached(
            "SELECT t.body FROM message m
             JOIN block b ON b.message_pk = m.pk AND b.kind = 'text'
             JOIN text t ON t.id = b.text_id
             WHERE m.session_pk = ?1 AND m.written_by = 'human'
               AND coalesce(m.prompt_kind, 'prose') = 'prose'
             ORDER BY m.order_input_pk, m.seq, b.idx LIMIT 1",
        )?
        .query_row([session_pk], |r| r.get(0))
        .optional()?;
    let title = match body {
        Some(b) => title_of(&crate::encoding::unpack_text(&b)?),
        None => None,
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
