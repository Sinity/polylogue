//! Injection fields a message version's sources declared, as they state them.
use crate::row::{Json, Rows as _, row};
use crate::{Reader, Result, StoreError};
use polylogue_model::{
    BlobHash, MessageId,
    injection::{InjectionEvidence, StatedHookField},
};
use rusqlite::params;

row! {
    /// An injection field a reading of a message version declared, on the
    /// block at `source_block_position`, and that reading's layout.
    struct DeclaredField {
        source_block_position: String,
        field: Json<StatedHookField>,
        layout: Vec<u8>,
    }
    statements {
        const OF_VERSION = "SELECT DISTINCT f.source_block_position, f.field, r.layout FROM message_parse r
            JOIN message_injection_field f ON f.input_pk=r.input_pk AND f.chain=r.chain AND f.message_id=r.message_id
            WHERE r.message_id=?1 AND r.hash=?2 ORDER BY length(f.source_block_position), f.source_block_position, f.field";
    }
}

impl Reader<'_> {
    /// A missing authenticated receiver stays unknown. Values the parser read
    /// and imported observations are both declarations in this storage path.
    pub fn injection_evidence(
        &self,
        message: &MessageId,
        source_version: BlobHash,
    ) -> Result<Vec<InjectionEvidence>> {
        let declared = self.conn()?.all(
            &DeclaredField::OF_VERSION,
            params![message.to_string(), source_version.0.as_slice()],
        )?;
        let mut grouped = std::collections::BTreeMap::<u64, Vec<StatedHookField>>::new();
        let mut block_selection = None;
        for DeclaredField {
            source_block_position: ordinal,
            field: Json(field),
            layout,
        } in declared
        {
            let block: u64 = ordinal
                .parse()
                .ok()
                .filter(|n: &u64| n.to_string() == ordinal)
                .ok_or_else(|| StoreError::stored(0, "invalid stored injection block ordinal"))?;
            if block_selection.is_none() {
                block_selection = Some(crate::encoding::message_block_selection(
                    &source_version.0,
                    Some(&layout),
                    0,
                )?);
            }
            let block_selection = block_selection
                .as_ref()
                .expect("block selection loaded for stored field");
            let injection = InjectionEvidence {
                message: message.clone(),
                source_version,
                block,
                observed: None,
                declared: vec![field.clone()],
            };
            injection.validate_for(message, block_selection)?;
            grouped.entry(block).or_default().push(field);
        }
        Ok(grouped
            .into_iter()
            .map(|(block, mut declared)| {
                declared.sort();
                declared.dedup();
                InjectionEvidence {
                    message: message.clone(),
                    source_version,
                    block,
                    observed: None,
                    declared,
                }
            })
            .collect())
    }
}

/// The rows this module declares statements for, for the every-statement test.
#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<DeclaredField>();
}

#[cfg(test)]
mod tests;
