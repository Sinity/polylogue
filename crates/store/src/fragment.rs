//! Captured partial-text claims. Their local source memberships and admissions
//! are independent of every coordinate and receipt in the foreign payload.
use crate::{Result, StoreError};
use polylogue_model::bundle::fragments::{TextFragmentError, TextSourceVersion};
use polylogue_model::{BlobHash, Output, batch::FragmentRow};
use rusqlite::{Connection, params};

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PreparedFragments(Vec<PreparedFragment>);

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct PreparedFragment {
    row: FragmentRow,
    receipt: Option<i64>,
    hash: BlobHash,
    version: String,
    body: Vec<u8>,
}

pub(crate) fn invalid(error: TextFragmentError) -> StoreError {
    StoreError::Invalid(error.to_string())
}

pub(crate) fn version_key(version: &TextSourceVersion) -> Result<String> {
    let mut value = serde_json::to_value(version)?;
    value.sort_all_objects();
    Ok(serde_json::to_string(&value)?)
}

pub(crate) fn receipt(value: u64) -> Result<i64> {
    i64::try_from(value).ok().filter(|n| *n > 0).ok_or_else(|| {
        StoreError::Invalid("fragment admission is outside the positive SQLite range".into())
    })
}

impl PreparedFragments {
    /// Hashing, validation and compression happen before the serial writer.
    pub(crate) fn prepare(output: &Output) -> Result<Self> {
        for (&index, &admitted) in &output.fragment_receipts {
            if index >= output.fragments.len() {
                return Err(StoreError::Invalid("fragment receipt has no record".into()));
            }
            receipt(admitted)?;
        }
        output
            .fragments
            .iter()
            .enumerate()
            .map(|(index, row)| {
                row.record.validate().map_err(invalid)?;
                if row.offset > row.end || i64::try_from(row.end).is_err() {
                    return Err(StoreError::Invalid(
                        "invalid receiving fragment byte range".into(),
                    ));
                }
                let mut value = serde_json::to_value(&row.record)?;
                value.sort_all_objects();
                let bytes = serde_json::to_vec(&value)?;
                let mut hash = blake3::Hasher::new();
                hash.update(b"polylogue-text-fragment-record-v1\0");
                hash.update(&bytes);
                Ok(PreparedFragment {
                    row: row.clone(),
                    receipt: output
                        .fragment_receipts
                        .get(&index)
                        .copied()
                        .map(receipt)
                        .transpose()?,
                    hash: BlobHash(*hash.finalize().as_bytes()),
                    version: version_key(&row.record.fragment.source_version)?,
                    body: crate::encoding::pack_json(&value)?,
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
        for ((prepared, row), index) in self.0.iter().zip(&output.fragments).zip(0..) {
            if &prepared.row != row
                || prepared.receipt
                    != output
                        .fragment_receipts
                        .get(&index)
                        .copied()
                        .map(receipt)
                        .transpose()?
            {
                return Err(StoreError::Invalid(
                    "prepared fragment differs from its reading".into(),
                ));
            }
        }
        if output
            .fragment_receipts
            .keys()
            .any(|index| *index >= self.0.len())
        {
            return Err(StoreError::Invalid("fragment receipt has no record".into()));
        }
        Ok(())
    }

    pub(crate) fn apply(
        &self,
        conn: &Connection,
        input: i64,
        chain: i64,
        through: u64,
        native_admitted: Option<i64>,
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
                "INSERT INTO text_fragment_source(hash,input_pk,chain,start,end,admitted) VALUES (?1,?2,?3,?4,?5,?6)
                 ON CONFLICT(hash,input_pk,chain,start,end) DO UPDATE SET admitted=excluded.admitted
                 WHERE admitted IS NULL AND excluded.admitted IS NOT NULL",
                params![value.hash.0.as_slice(),input,chain,value.row.offset as i64,value.row.end as i64,value.receipt.or(native_admitted)],
            )?;
        }
        Ok(())
    }
}

/// Ordinary replacement retracts this reading's membership, not captured bytes
/// or a different input's claim. Unreferenced values belong to the derived cache.
pub(crate) fn retract(conn: &Connection, input: i64, chain: i64) -> Result<()> {
    conn.execute(
        "DELETE FROM text_fragment_source WHERE input_pk=?1 AND chain=?2",
        params![input, chain],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;
