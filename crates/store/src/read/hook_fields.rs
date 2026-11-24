//! Injection fields a message version's sources declared, as they state them.
use crate::{Reader, Result, StoreError};
use polylogue_model::{
    BlobHash, MessageId,
    injection::{InjectionEvidence, StatedHookField},
};
use rusqlite::params;

impl Reader<'_> {
    /// A missing authenticated receiver stays unknown. Values the parser read
    /// and imported observations are both declarations in this storage path.
    pub fn injection_evidence(
        &self,
        message: &MessageId,
        source_version: BlobHash,
    ) -> Result<Vec<InjectionEvidence>> {
        let conn = self.conn()?;
        let mut statement=conn.prepare_cached(
            "SELECT f.block_ordinal,f.field,p.layout FROM message_injection_field f
             JOIN message m ON m.pk=f.message_pk
             JOIN message_block_selection p ON p.version_pk=f.version_pk
             JOIN message_parse s ON s.message_pk=f.message_pk AND s.input_pk=f.input_pk AND s.chain=f.chain AND s.version_pk=f.version_pk
             WHERE m.id=?1 AND f.version_pk=(SELECT pk FROM message_version WHERE version_hash=?2) ORDER BY length(f.block_ordinal),f.block_ordinal,f.field",
        )?;
        let mut rows =
            statement.query(params![message.to_string(), source_version.0.as_slice()])?;
        let mut grouped = std::collections::BTreeMap::<u64, Vec<StatedHookField>>::new();
        let mut block_selection = None;
        while let Some(row) = rows.next()? {
            let ordinal: String = row.get(0)?;
            let block: u64 = ordinal
                .parse()
                .ok()
                .filter(|n: &u64| n.to_string() == ordinal)
                .ok_or_else(|| StoreError::stored(0, "invalid stored injection block ordinal"))?;
            let field: StatedHookField = serde_json::from_str(&row.get::<_, String>(1)?)?;
            if block_selection.is_none() {
                let layout: Vec<u8> = row.get(2)?;
                block_selection = Some(crate::versions::message_block_selection(
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

#[cfg(test)]
mod tests;
