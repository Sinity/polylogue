//! A title for a session whose records name none: the first line of the
//! first prompt a person typed, masked in source context before truncation.
//! Kept apart from the recorded title, which always wins when present and
//! is never replaced by this display fallback in exports.

use super::{Derivation, Version};
use crate::Result;
use polylogue_model::Block;
use rusqlite::{OptionalExtension, Transaction, params};
use std::ops::ControlFlow;

pub(super) const DERIVATION: Derivation = Derivation {
    name: "title",
    version: Version::Rules(3),
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
        Some(pk) => {
            let alone = crate::ordered_blocks::masked_alone(tx, session_pk, &[pk].into())?;
            let block = if alone.contains(&pk) {
                // No PEM block is open before it, so it masks the same alone.
                let mut block = crate::ordered_blocks::one(tx, pk)?;
                polylogue_model::masking::MessageMasker::default().block(&mut block);
                Some(block)
            } else {
                masked_in_order(tx, session_pk, pk)?
            };
            block.as_ref().and_then(Block::text).and_then(title_of)
        }
    };
    tx.prepare_cached(
        "UPDATE session SET derived_title = ?2 WHERE pk = ?1 AND derived_title IS NOT ?2",
    )?
    .execute(params![session_pk, title])?;
    Ok(())
}

/// Block `pk` masked after every earlier block of the session.
fn masked_in_order(tx: &Transaction<'_>, session_pk: i64, pk: i64) -> Result<Option<Block>> {
    let mut masker = polylogue_model::masking::MessageMasker::default();
    let mut found = None;
    crate::ordered_blocks::visit(tx, session_pk, |mut source| {
        masker.block(&mut source.block);
        if source.pk == pk {
            found = Some(source.block);
            return Ok(ControlFlow::Break(()));
        }
        Ok(ControlFlow::Continue(()))
    })?;
    Ok(found)
}

pub(crate) fn title_of(prompt: &str) -> Option<String> {
    let line = prompt.lines().map(str::trim).find(|l| !l.is_empty())?;
    let mut t: String = line.chars().take(LENGTH).collect();
    if line.chars().count() > LENGTH {
        t.push('…');
    }
    Some(t)
}
