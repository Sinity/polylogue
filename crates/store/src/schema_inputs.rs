//! Read-only selection of exact input revisions for offline inspection.
use crate::{Reader, Result, StoreError};
use rusqlite::{OptionalExtension, params};

#[derive(Debug, Clone)]
pub struct InputRevision {
    pub input: String,
    pub format: String,
    pub revision: i64,
    pub chain: i64,
    pub length: u64,
    /// Used only to choose the origin's framing, never emitted in a schema.
    pub path: String,
}

impl Reader<'_> {
    /// `None` selects the latest revision of this exact input, never another input.
    pub fn schema_input(
        &self,
        input: &str,
        revision: Option<i64>,
    ) -> Result<Option<InputRevision>> {
        let conn = self.conn()?;
        let row = conn.query_row(
            "SELECT i.format,r.number,r.chain,r.length,
             coalesce((SELECT o.path FROM source_observation o WHERE o.input=i.id AND o.revision=r.number ORDER BY o.rowid DESC LIMIT 1),'')
             FROM source_input i JOIN source_revision r ON r.input=i.id
             WHERE i.id=?1 AND (?2 IS NULL OR r.number=?2) ORDER BY r.number DESC LIMIT 1",
            params![input, revision], |r| Ok((r.get::<_,String>(0)?,r.get::<_,i64>(1)?,r.get::<_,i64>(2)?,r.get::<_,i64>(3)?,r.get::<_,String>(4)?))
        ).optional()?;
        row.map(|(format, revision, chain, length, path)| {
            Ok(InputRevision {
                input: input.into(),
                format,
                revision,
                chain,
                length: u64::try_from(length)
                    .map_err(|_| StoreError::Invalid("negative input length".into()))?,
                path,
            })
        })
        .transpose()
    }
}

#[cfg(test)]
mod tests;
