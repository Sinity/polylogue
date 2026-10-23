//! A title for a session whose records name none: the first line of the
//! first prompt a person typed, masked in source context before truncation.
//! Kept apart from the recorded title, which always wins when present and
//! is never replaced by this display fallback in exports.

use super::{Derivation, Version};
use crate::Result;
use rusqlite::{Transaction, params};

pub(super) const DERIVATION: Derivation = Derivation {
    name: "title",
    version: Version::Rules(3),
    after: &["session"],
    run: |tx, pk, _| run(tx, pk),
    finish: None,
};

/// Characters kept of the prompt's first line.
const LENGTH: usize = 80;

fn run(tx: &Transaction<'_>, session_pk: i64) -> Result<()> {
    let mut masker = polylogue_model::masking::MessageMasker::default();
    let mut title = None;
    crate::ordered_blocks::visit(tx, session_pk, |mut source| {
        use std::ops::ControlFlow;
        masker.block(&mut source.block);
        if source.written_by == "human"
            && source
                .prompt_kind
                .as_deref()
                .is_none_or(|kind| kind == "prose")
            && matches!(source.block, polylogue_model::Block::Text { .. })
        {
            title = source.block.text().and_then(title_of);
            return Ok(ControlFlow::Break(()));
        }
        Ok(ControlFlow::Continue(()))
    })?;
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
