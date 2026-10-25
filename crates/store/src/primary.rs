//! Primary mode: Sinex holds the archive. Each piece of an input's bytes
//! goes to Sinex as one part of that input, with the [`InputRecord`] that
//! places it; a step that found no new bytes, and every request a person or
//! agent makes, goes as a record. The writer commits a register row only
//! after Sinex acknowledged what it describes, so the local register
//! (`sinex.db`) and `index.db` are caches: deleted, they are rebuilt from
//! what Sinex holds ([`load`]). `polylogue-sinex` implements [`Sinex`].
use crate::input_record::InputRecord;
use crate::user_writes::UserRequest;
use crate::{ReadPool, Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

/// The connection to Sinex the writer and the loop send through.
pub trait Sinex: Send + Sync {
    /// Sends `bytes` (`part.length` long, hashing to `part.hash`) as one part
    /// of its input and waits until Sinex acknowledged it. Returns the
    /// material Sinex registered.
    fn send_part(&self, part: &Part, bytes: &mut dyn std::io::Read) -> Result<String>;
    /// Sends a record and the events it makes, and waits until Sinex
    /// acknowledged them.
    fn send_record(&self, record: &Record) -> Result<()>;
    /// One page of what Sinex holds from this archive, in the order it was
    /// sent, after `after` (0 for the first page). An empty page is the end.
    fn held(&self, after: i64) -> Result<Vec<Held>>;
    /// Sends the events of every session whose rows changed since they were
    /// last sent, read from the index.
    fn send_sessions(&self, reads: &ReadPool) -> Result<()>;
}

/// One part of an input's bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Part {
    pub input: InputName,
    /// Its place in its revision chain (0 for a chain's first part), and the
    /// material of the part before it.
    pub index: u32,
    pub previous: Option<String>,
    pub hash: BlobHash,
    pub length: u64,
    pub record: InputRecord,
}

/// What an input is, in words Sinex keeps: the origin that wrote it, its
/// path (or `drive:<id>`, an import's file name, …) and the host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputName {
    pub origin: String,
    pub path: String,
    pub host: String,
}

/// What Sinex keeps for the archive besides input bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "record", rename_all = "snake_case")]
pub enum Record {
    /// A look at an input that found no new bytes (it disappeared, came
    /// back, or an import of bytes already stored).
    Observation {
        input: InputName,
        stored: InputRecord,
    },
    /// A request a person or agent made, with the note family it joined
    /// when it adds or withdraws a note.
    Request {
        request: UserRequest,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        root: Option<String>,
    },
}

/// One thing Sinex holds from this archive.
#[derive(Debug, Clone, PartialEq)]
pub enum Held {
    Part {
        number: i64,
        material: String,
        hash: BlobHash,
        length: u64,
        index: u32,
        record: InputRecord,
    },
    Record {
        number: i64,
        record: Record,
    },
}

impl Held {
    pub fn number(&self) -> i64 {
        match self {
            Held::Part { number, .. } | Held::Record { number, .. } => *number,
        }
    }
}

/// The Sinex material holding each part, by where its bytes sit in their
/// chain. Only `sinex.db` has it.
pub(crate) const SCHEMA: &str = "
CREATE TABLE material (
  input         TEXT NOT NULL,
  chain         INTEGER NOT NULL,
  offset        INTEGER NOT NULL,
  part          INTEGER NOT NULL,        -- its place in the chain, from 0
  id            TEXT NOT NULL UNIQUE,    -- the Sinex material
  PRIMARY KEY (input, chain, offset)
) STRICT, WITHOUT ROWID;
";

/// Where a new part of `input`'s chain goes: its place and the material
/// before it.
pub(crate) fn next_part(conn: &Connection, input: &str, chain: i64) -> Result<(u32, Option<String>)> {
    let last: Option<(i64, String)> = conn
        .prepare_cached(
            "SELECT part, id FROM material WHERE input = ?1 AND chain = ?2 ORDER BY offset DESC LIMIT 1",
        )?
        .query_row(params![input, chain], |r| Ok((r.get(0)?, r.get(1)?)))
        .optional()?;
    Ok(match last {
        Some((part, id)) => (
            u32::try_from(part + 1)
                .map_err(|_| StoreError::Invalid("too many parts in one chain".into()))?,
            Some(id),
        ),
        None => (0, None),
    })
}

pub(crate) fn record_material(
    conn: &Connection,
    record: &InputRecord,
    part: u32,
    material: &str,
) -> Result<()> {
    let extent = record
        .extent
        .as_ref()
        .ok_or_else(|| StoreError::Invalid("a part needs an extent".into()))?;
    conn.prepare_cached(
        "INSERT OR REPLACE INTO material(input, chain, offset, part, id) VALUES (?1, ?2, ?3, ?4, ?5)",
    )?
    .execute(params![
        record.input,
        extent.chain,
        extent.offset,
        i64::from(part),
        material
    ])?;
    Ok(())
}

/// Fills an empty register from everything Sinex holds for the archive, in
/// the order it was sent: each part's rows with its blob present in Sinex's
/// content store, each record's rows. `store` reads Sinex's content store.
pub(crate) fn load(
    conn: &mut Connection,
    sinex: &dyn Sinex,
    store: &hashed_blobs::Store,
) -> Result<u64> {
    let tx = conn.transaction()?;
    let seq = crate::archive::bump_counter(&tx)?;
    let mut after = 0;
    let mut loaded = 0;
    loop {
        let page = sinex.held(after)?;
        let Some(last) = page.last() else {
            break;
        };
        after = last.number();
        for held in page {
            match held {
                Held::Part {
                    material,
                    hash,
                    length,
                    index,
                    record,
                    ..
                } => {
                    let stored_size = store.stored_size(&hashed_blobs::Hash(hash.0))?;
                    tx.prepare_cached(
                        "INSERT OR IGNORE INTO blob(hash, size, stored_size, place, encoding, state)
                         VALUES (?1, ?2, ?3, 'file', 'plain', 'present')",
                    )?
                    .execute(params![&hash.0[..], length as i64, stored_size as i64])?;
                    record.insert(&tx, seq)?;
                    record_material(&tx, &record, index, &material)?;
                }
                Held::Record {
                    record: Record::Observation { stored, .. },
                    ..
                } => stored.insert(&tx, seq)?,
                Held::Record {
                    record: Record::Request { request, .. },
                    ..
                } => {
                    crate::user_writes::apply(&tx, &request, seq)?;
                }
            }
            loaded += 1;
        }
    }
    tx.commit()?;
    Ok(loaded)
}

#[cfg(test)]
mod tests;
