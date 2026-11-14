//! Exact retained field claims, independent of canonical message identity.
use crate::{Reader, Result, StoreError};
use polylogue_model::{
    BlobHash, MessageId,
    injection::{InjectionEvidence, NativeHookField},
};
use rusqlite::params;

impl Reader<'_> {
    /// Missing authenticated receiver provenance stays unknown. Native parser
    /// claims and imported observations are declarations in this storage path.
    pub fn injection_evidence(
        &self,
        message: &MessageId,
        source_version: BlobHash,
    ) -> Result<Vec<InjectionEvidence>> {
        let conn = self.conn()?;
        let mut statement=conn.prepare_cached(
            "SELECT f.block_ordinal,f.field,p.layout FROM message_injection_field f
             JOIN message m ON m.pk=f.message_pk
             JOIN message_projection p ON p.version_pk=f.version_pk
             JOIN message_source s ON s.message_pk=f.message_pk AND s.input_pk=f.input_pk AND s.chain=f.chain AND s.version_pk=f.version_pk
             WHERE m.id=?1 AND f.version_pk=(SELECT pk FROM message_version WHERE version_hash=?2) ORDER BY length(f.block_ordinal),f.block_ordinal,f.field",
        )?;
        let mut rows =
            statement.query(params![message.to_string(), source_version.0.as_slice()])?;
        let mut grouped = std::collections::BTreeMap::<u64, Vec<NativeHookField>>::new();
        let mut projection = None;
        while let Some(row) = rows.next()? {
            let ordinal: String = row.get(0)?;
            let block: u64 = ordinal
                .parse()
                .ok()
                .filter(|n: &u64| n.to_string() == ordinal)
                .ok_or_else(|| {
                    StoreError::Invalid("invalid stored injection block ordinal".into())
                })?;
            let field: NativeHookField = serde_json::from_str(&row.get::<_, String>(1)?)?;
            if projection.is_none() {
                let layout: Vec<u8> = row.get(2)?;
                projection = Some(crate::versions::message_projection(
                    &source_version.0,
                    Some(&layout),
                    0,
                )?);
            }
            let projection = projection
                .as_ref()
                .expect("projection loaded for retained field");
            let evidence = InjectionEvidence {
                message: message.clone(),
                source_version,
                block,
                observed: None,
                declared: vec![field.clone()],
            };
            evidence.validate_for(message, projection)?;
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
