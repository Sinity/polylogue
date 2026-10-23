//! A title for a session whose records name none: the first line of the
//! first prompt a person typed, masked in source context before truncation. Kept apart from the recorded title, which
//! always wins when there is one, and never exported as if recorded.

use super::{Derivation, Version};
use crate::Result;
use rusqlite::{Transaction, params};

pub(super) const DERIVATION: Derivation = Derivation {
    name: "title",
    version: Version::Rules(2),
    after: &["session"],
    run: |tx, pk, _| run(tx, pk),
    finish: None,
};

/// Characters kept of the prompt's first line.
const LENGTH: usize = 80;

fn run(tx: &Transaction<'_>, session_pk: i64) -> Result<()> {
    let mut masker = polylogue_model::masking::MessageMasker::default();
    let mut statement = tx.prepare_cached(
        "SELECT m.written_by, m.prompt_kind, b.kind, b.body, t.body
         FROM current_message m JOIN block b ON b.message_pk=m.pk
         LEFT JOIN text t ON t.id=b.text_id WHERE m.session_pk=?1
         ORDER BY m.order_input_id, m.seq, b.idx",
    )?;
    let mut rows = statement.query([session_pk])?;
    let mut title = None;
    while let Some(row) = rows.next()? {
        let mut block = crate::encoding::unpack_block(&row.get::<_, Vec<u8>>(3)?)?;
        if let Some(text) = row.get::<_, Option<Vec<u8>>>(4)? {
            crate::encoding::restore_text(&mut block, crate::encoding::unpack_text(&text)?);
        }
        masker.block(&mut block);
        if row.get::<_, String>(0)? == "human"
            && row
                .get::<_, Option<String>>(1)?
                .is_none_or(|kind| kind == "prose")
            && row.get::<_, String>(2)? == "text"
        {
            title = block.text().and_then(title_of);
            break;
        }
    }
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
