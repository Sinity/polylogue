//! Captured partial-text fragments as their sender states them. Their local
//! source memberships and commits are independent of every coordinate and
//! commit the sender's payload states.
use crate::{Result, StoreError};
use polylogue_model::bundle::fragments::TextSourceVersion;
use polylogue_model::{BlobHash, Output, batch::FragmentRow};
use rusqlite::{Connection, params};

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PreparedFragments(Vec<PreparedFragment>);

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct PreparedFragment {
    row: FragmentRow,
    hash: BlobHash,
    version: String,
    body: Vec<u8>,
}

pub(crate) fn version_key(version: &TextSourceVersion) -> Result<String> {
    let bytes =
        canonical_json::canonical_json(version).map_err(crate::encoding::identity_encoding)?;
    String::from_utf8(bytes).map_err(crate::StoreError::invalid)
}

impl PreparedFragments {
    /// Hashing, validation and compression happen before the serial writer.
    pub(crate) fn prepare(output: &Output) -> Result<Self> {
        output
            .fragments
            .iter()
            .map(|row| {
                row.record.validate().map_err(crate::StoreError::invalid)?;
                if row.offset > row.end {
                    return Err(StoreError::Invalid(
                        "receiving fragment byte range ends before it starts".into(),
                    ));
                }
                if i64::try_from(row.end).is_err() {
                    return Err(StoreError::Invalid(
                        "receiving fragment byte range exceeds SQLite's integer range".into(),
                    ));
                }
                let hash = canonical_json::canonical_hash(
                    "polylogue-text-fragment-record-v1",
                    &row.record,
                )
                .map_err(crate::encoding::identity_encoding)?;
                Ok(PreparedFragment {
                    row: row.clone(),
                    hash: BlobHash(*hash.as_bytes()),
                    version: version_key(&row.record.fragment.source_version)?,
                    body: crate::encoding::pack_json(&row.record)?,
                })
            })
            .collect::<Result<Vec<_>>>()
            .map(Self)
    }

    pub(crate) fn validate(&self, output: &Output) -> Result<()> {
        if self.0.len() != output.fragments.len() {
            return Err(StoreError::Invalid(
                "fragment rows were not prepared".into(),
            ));
        }
        for (prepared, row) in self.0.iter().zip(&output.fragments) {
            if &prepared.row != row {
                return Err(StoreError::Invalid(
                    "prepared fragment differs from its reading".into(),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn apply(
        &self,
        conn: &Connection,
        input: i64,
        chain: i64,
        through: u64,
        commit: impl Fn(&FragmentRow) -> Result<Option<i64>>,
    ) -> Result<()> {
        for value in &self.0 {
            if value.row.end > through {
                return Err(StoreError::Invalid(
                    "fragment extends beyond its receiving reading".into(),
                ));
            }
            let fragment = &value.row.record.fragment;
            let start = fragment.coverage.start.to_be_bytes();
            let end = fragment.coverage.end.to_be_bytes();
            let total = fragment.coverage.total_bytes.map(u64::to_be_bytes);
            conn.execute(
                "INSERT OR IGNORE INTO text_fragment(hash,container,version,start,end,total,record) VALUES (?1,?2,?3,?4,?5,?6,?7)",
                params![value.hash.0.as_slice(),fragment.container.to_string(),value.version,start.as_slice(),end.as_slice(),total.as_ref().map(|n|n.as_slice()),value.body],
            )?;
            conn.execute(
                "INSERT INTO text_fragment_source(hash,input_pk,chain,start,end,committed) VALUES (?1,?2,?3,?4,?5,?6)
                 ON CONFLICT(hash,input_pk,chain,start,end) DO UPDATE SET committed=excluded.committed
                 WHERE committed IS NULL AND excluded.committed IS NOT NULL",
                params![value.hash.0.as_slice(),input,chain,value.row.offset as i64,value.row.end as i64,commit(&value.row)?],
            )?;
        }
        Ok(())
    }
}

/// Replacing a reading retracts its membership, not captured bytes or a
/// different input's stated fragment. Unreferenced values belong to the derived cache.
pub(crate) fn retract(conn: &Connection, input: i64, chain: i64) -> Result<()> {
    conn.execute(
        "DELETE FROM text_fragment_source WHERE input_pk=?1 AND chain=?2",
        params![input, chain],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;
