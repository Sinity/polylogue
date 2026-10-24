//! Retained source bytes and unfinished canonical ownership.
use crate::{Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::{Connection, OptionalExtension};
use serde::Serialize;

/// Private staging tied to one archive and its configured read roots. Only
/// the Writer can publish it after reserving canonical ownership.
pub struct PreparedBytes {
    pub(crate) owner: std::path::PathBuf,
    pub(crate) store: std::sync::Arc<hashed_blobs::Store>,
    pub(crate) prepared: hashed_blobs::PreparedPlain,
}

impl PreparedBytes {
    pub fn hash(&self) -> BlobHash {
        BlobHash(self.prepared.hash().0)
    }
    pub fn len(&self) -> u64 {
        self.prepared.decoded_len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl crate::Archive {
    /// Inspect the same private bytes that will be prepared, without reopening
    /// a mutable provider or installing a public CAS name. The decoded scratch
    /// file costs one streamed copy and is dropped after preparation.
    pub fn inspect_and_prepare_bytes<T>(
        &self,
        roots: &[std::path::PathBuf],
        mut reader: impl std::io::Read,
        inspect: impl FnOnce(&mut std::fs::File) -> Result<T>,
    ) -> Result<(PreparedBytes, T)> {
        use std::io::Seek;
        let mut scratch = tempfile::tempfile()?;
        std::io::copy(&mut reader, &mut scratch)?;
        scratch.rewind()?;
        let facts = inspect(&mut scratch)?;
        scratch.rewind()?;
        Ok((self.prepare_bytes(roots, &mut scratch)?, facts))
    }

    /// Compress and hash without publishing a CAS name. Dropping the returned
    /// handle discards only its private stage.
    pub fn prepare_bytes(
        &self,
        roots: &[std::path::PathBuf],
        reader: impl std::io::Read,
    ) -> Result<PreparedBytes> {
        let store = std::sync::Arc::new(hashed_blobs::Store::open(
            self.root.clone(),
            roots.to_vec(),
            0o600,
        )?);
        let prepared =
            store.prepare_plain_checked::<StoreError>(reader, crate::blobs::LEVEL, || Ok(()))?;
        Ok(PreparedBytes {
            owner: self.root.clone(),
            store,
            prepared,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Retained {
    pub input: String,
    pub chain: i64,
    pub length: u64,
    pub already_stored: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UnfinishedState {
    Writing,
    Missing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Unfinished {
    pub input: String,
    pub blob: BlobHash,
    pub length: u64,
    pub state: UnfinishedState,
}

#[derive(Debug, Default)]
pub struct Reconciled {
    pub completed: u64,
    pub missing: Vec<Unfinished>,
}

pub(crate) fn require_present(
    conn: &Connection,
    schema: &str,
    input: &str,
    chain: Option<i64>,
) -> Result<()> {
    check(conn, schema, input, chain, None)
}

pub(crate) fn require_range(
    conn: &Connection,
    schema: &str,
    input: &str,
    chain: i64,
    range: std::ops::Range<u64>,
) -> Result<()> {
    check(conn, schema, input, Some(chain), Some(range))
}

fn check(
    conn: &Connection,
    schema: &str,
    input: &str,
    chain: Option<i64>,
    range: Option<std::ops::Range<u64>>,
) -> Result<()> {
    let position = |value| {
        i64::try_from(value)
            .map_err(|_| StoreError::Invalid("retained range exceeds archive address space".into()))
    };
    let from = range
        .as_ref()
        .map(|range| position(range.start))
        .transpose()?;
    let until = range
        .as_ref()
        .map(|range| position(range.end))
        .transpose()?;
    let sql=format!("SELECT b.state FROM {schema}extent e JOIN {schema}blob b ON b.hash=e.blob WHERE e.input=?1 AND (?2 IS NULL OR e.chain=?2)
        AND (?3 IS NULL OR (e.offset<?4 AND e.offset+e.length>?3) OR (?3=?4 AND e.offset=?3 AND e.length=0))
        AND b.state<>'present' LIMIT 1");
    let pending: Option<String> = conn
        .query_row(&sql, rusqlite::params![input, chain, from, until], |row| {
            row.get(0)
        })
        .optional()?;
    if let Some(state) = pending {
        return Err(StoreError::InputUnfinished {
            input: input.into(),
            reason: format!("retained bytes are {state}"),
        });
    }
    Ok(())
}

pub(crate) fn unfinished(conn: &Connection, schema: &str) -> Result<Vec<Unfinished>> {
    let sql=format!("SELECT DISTINCT i.id,b.hash,b.size FROM {schema}input i JOIN {schema}revision r ON r.input=i.id
        JOIN {schema}extent e ON e.input=r.input AND e.chain=r.chain JOIN {schema}blob b ON b.hash=e.blob
        WHERE b.state='writing' ORDER BY i.id,b.hash");
    let mut stmt = conn.prepare(&sql)?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let hash: Vec<u8> = row.get(1)?;
        let length = row.get::<_, i64>(2)?;
        out.push(Unfinished {
            input: row.get(0)?,
            blob: BlobHash(
                hash.try_into()
                    .map_err(|_| StoreError::Invalid("invalid unfinished blob hash".into()))?,
            ),
            length: u64::try_from(length)
                .map_err(|_| StoreError::Invalid("negative unfinished source length".into()))?,
            state: UnfinishedState::Writing,
        });
    }
    Ok(out)
}
