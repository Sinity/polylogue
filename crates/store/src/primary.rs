//! Primary mode: Sinex holds the archive. Each step of an input's appended
//! bytes goes to Sinex as one part of that input, with the [`InputRows`]
//! that place it; every other change to the register (a look that found no
//! new bytes or a request a person or agent makes) goes as a
//! register material. The writer commits a register row only
//! after Sinex acknowledged what it describes, so the local register
//! (`sinex.db`) and `index.db` are caches: deleted, they are rebuilt from
//! what Sinex holds ([`load`]). `polylogue-sinex` implements [`Sinex`].
pub mod products;
pub(crate) mod projection;
pub use products::{
    Capture, CaptureGroup, CapturedInput, Confirmed, ConfirmedChange, Material, PendingCapture,
    PreparedInput, PreparedWrite, Product, ProductWrite, Reading, SourceLocation,
};

use crate::input_rows::InputRows;
use crate::{ReadPool, Result, StoreError};
use polylogue_model::BlobHash;
use polylogue_model::user::{UserOutcome, UserRequest};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

/// The connection to Sinex the writer and the loop send through.
pub trait Sinex: Send + Sync {
    fn prepare_input(
        &self,
        captured: &CapturedInput,
        bytes: &mut dyn std::io::Read,
    ) -> Result<PreparedInput>;
    fn prepare_blob(
        &self,
        hash: BlobHash,
        length: u64,
        bytes: &mut dyn std::io::Read,
    ) -> Result<PreparedWrite>;
    fn pending_captures(&self) -> Result<Vec<PendingCapture>>;
    fn complete_captures(
        &self,
        identities: &[String],
        products: &[ProductWrite],
    ) -> Result<PreparedWrite>;
    fn prepare_products(&self, products: &[ProductWrite]) -> Result<PreparedWrite>;
    fn prepare_request(&self, request: &UserRequest) -> Result<UserOutcome>;
    fn confirmed(&self, after: i64, cache: &str) -> Result<Vec<Confirmed>>;
    fn acknowledge(&self, position: i64) -> Result<()>;
}

/// The material and committed descriptors Sinex acknowledged for these bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfirmedBlob {
    pub id: String,
    pub extents: Vec<hashed_blobs::Extent>,
}

impl ConfirmedBlob {
    /// Polylogue stores one prepared blob in one segment extent.
    pub fn extent(&self, hash: BlobHash, length: u64) -> Result<&hashed_blobs::Extent> {
        let [extent] = self.extents.as_slice() else {
            return Err(StoreError::Sinex(
                "a blob needs one committed extent".into(),
            ));
        };
        if extent.hash.0 != hash.0 || extent.decoded_len != length || extent.len == 0 {
            return Err(StoreError::Sinex(
                "committed extent differs from its blob".into(),
            ));
        }
        Ok(extent)
    }
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
    pub rows: InputRows,
}

/// What an input is, in words Sinex keeps: the origin that wrote it, its
/// path (or `drive:<id>`, an import's file name, …) and the host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct InputName {
    pub origin: String,
    pub path: String,
    pub host: String,
}

/// A change to the register that is not input bytes, which Sinex keeps as
/// a register material.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "change", rename_all = "snake_case")]
pub enum RegisterChange {
    /// A look at an input that found no new bytes (it disappeared, came
    /// back, or an import of bytes already stored).
    Observation {
        input: InputName,
        stored: Box<InputRows>,
    },
    /// A request a person or agent made, with the first note of the note's
    /// revisions when it adds or withdraws a note.
    Request {
        request: Box<UserRequest>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        first_note: Option<String>,
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
        extents: Vec<hashed_blobs::Extent>,
        index: u32,
        rows: Box<InputRows>,
    },
    Blob {
        number: i64,
        material: String,
        hash: BlobHash,
        length: u64,
        extents: Vec<hashed_blobs::Extent>,
    },
    Change {
        number: i64,
        material: String,
        change: RegisterChange,
    },
}

impl Held {
    pub fn number(&self) -> i64 {
        match self {
            Held::Part { number, .. } | Held::Blob { number, .. } | Held::Change { number, .. } => {
                *number
            }
        }
    }
}

/// The Sinex material holding each part, by where its bytes sit in their
/// chain, and each register material, by the observation, request or
/// it holds. Only `sinex.db` has them.
pub(crate) const SCHEMA: &str = "
CREATE TABLE material (
  input         TEXT NOT NULL,
  chain         INTEGER NOT NULL,
  offset        INTEGER NOT NULL,
  part          INTEGER NOT NULL,        -- its place in the chain, from 0
  id            TEXT NOT NULL UNIQUE,    -- the Sinex material
  PRIMARY KEY (input, chain, offset)
) STRICT, WITHOUT ROWID;
CREATE TABLE register_material (
  id            TEXT PRIMARY KEY,        -- the observation, request ID
  input         TEXT,                    -- an observation's input
  material      TEXT NOT NULL UNIQUE     -- the Sinex material
) STRICT, WITHOUT ROWID;
CREATE INDEX register_material_input ON register_material(input) WHERE input IS NOT NULL;
";

/// Where a new part of `input`'s chain goes: its place and the material
/// before it.
pub(crate) fn next_part(
    conn: &Connection,
    input: &str,
    chain: i64,
) -> Result<(u32, Option<String>)> {
    let last: Option<(i64, String)> = conn
        .prepare_cached(
            "SELECT part, id FROM material WHERE input = ?1 AND chain = ?2 ORDER BY offset DESC LIMIT 1",
        )?
        .query_row(params![input, chain], |r| Ok((r.get(0)?, r.get(1)?)))
        .optional()?;
    Ok(match last {
        Some((part, id)) => (
            u32::try_from(part + 1)
                .map_err(|_| StoreError::Invalid("too many parts for one chain".into()))?,
            Some(id),
        ),
        None => (0, None),
    })
}

/// Keeps which register material holds `change`.
pub(crate) fn change_material(
    conn: &Connection,
    change: &RegisterChange,
    material: &str,
) -> Result<()> {
    let (id, input) = match change {
        RegisterChange::Observation { stored, .. } => (
            stored
                .observation
                .as_ref()
                .map(|o| o.id.as_str())
                .ok_or_else(|| {
                    StoreError::Invalid("an observation change has no observation".into())
                })?,
            Some(stored.input.as_str()),
        ),
        RegisterChange::Request { request, .. } => (request.id.as_str(), None),
    };
    conn.prepare_cached(
        "INSERT OR REPLACE INTO register_material(id, input, material) VALUES (?1, ?2, ?3)",
    )?
    .execute(params![id, input, material])?;
    Ok(())
}

/// Keeps which Sinex material holds a part.
pub(crate) fn part_material(
    conn: &Connection,
    rows: &InputRows,
    part: u32,
    material: &str,
) -> Result<()> {
    let extent = rows
        .extent
        .as_ref()
        .ok_or_else(|| StoreError::Invalid("appended bytes need a stored byte range".into()))?;
    conn.prepare_cached(
        "INSERT OR REPLACE INTO material(input, chain, offset, part, id) VALUES (?1, ?2, ?3, ?4, ?5)",
    )?
    .execute(params![
        rows.input,
        extent.chain,
        extent.offset,
        i64::from(part),
        material
    ])?;
    Ok(())
}

/// Fills an empty register from everything Sinex holds for the archive, in
/// the order it was sent: each part's rows with its blob present in Sinex's
/// content store, each register change's rows. `store` reads Sinex's
/// content store.
pub(crate) fn load(
    conn: &mut Connection,
    sinex: &dyn Sinex,
    store: &hashed_blobs::Reader,
) -> Result<u64> {
    let tx = conn.transaction()?;
    let seq = crate::archive::bump_counter(&tx)?;
    let mut after = 0;
    let mut loaded = 0;
    // Where each part's bytes sit, with its observation and blob: a later
    // part at the same place is the one whose rows committed, and the
    // earlier one was sent by a write that stopped before its commit.
    let mut placed: std::collections::HashMap<(String, i64, i64), (Option<String>, BlobHash)> =
        std::collections::HashMap::new();
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
                    extents,
                    index,
                    rows,
                    ..
                } => {
                    store_blob(&tx, store, hash, length, &extents)?;
                    if rows.extent.as_ref().is_none_or(|extent| {
                        extent.blob != hash || u64::try_from(extent.length).ok() != Some(length)
                    }) {
                        return Err(StoreError::Sinex(
                            "part rows differ from their material bytes".into(),
                        ));
                    }
                    rows.insert(&tx, seq)?;
                    part_material(&tx, &rows, index, &material)?;
                    if let Some(extent) = &rows.extent {
                        let place = (rows.input.clone(), extent.chain, extent.offset);
                        let observation = rows.observation.as_ref().map(|o| o.id.clone());
                        if let Some((earlier, blob)) =
                            placed.insert(place, (observation, extent.blob))
                        {
                            forget_uncommitted(&tx, earlier.as_deref(), &blob, &extent.blob)?;
                        }
                    }
                }
                Held::Blob {
                    hash,
                    length,
                    extents,
                    ..
                } => {
                    store_blob(&tx, store, hash, length, &extents)?;
                }
                Held::Change {
                    material, change, ..
                } => {
                    match &change {
                        RegisterChange::Observation { stored, .. } => stored.insert(&tx, seq)?,
                        RegisterChange::Request { request, .. } => {
                            crate::user_writes::apply(&tx, request, seq)?;
                        }
                    }
                    change_material(&tx, &change, &material)?;
                }
            }
            loaded += 1;
        }
    }
    tx.commit()?;
    Ok(loaded)
}

fn store_blob(
    conn: &Connection,
    store: &hashed_blobs::Reader,
    hash: BlobHash,
    length: u64,
    extents: &[hashed_blobs::Extent],
) -> Result<()> {
    let confirmed = ConfirmedBlob {
        id: String::new(),
        extents: extents.to_vec(),
    };
    let extent = confirmed.extent(hash, length)?;
    let mut bytes = store.open(extents, None)?;
    let mut digest = blake3::Hasher::new();
    let read = std::io::copy(&mut bytes, &mut digest)?;
    if read != length || digest.finalize().as_bytes() != &hash.0 {
        return Err(StoreError::BlobCorrupt(hash));
    }
    conn.execute(
        "INSERT OR IGNORE INTO blob(hash,size,stored_size,storage) VALUES (?1,?2,?3,?4)",
        params![
            &hash.0[..],
            length as i64,
            extent.len as i64,
            extent.encode()
        ],
    )?;
    Ok(())
}

/// Drops the rows of a part a stopped write sent but never committed: its
/// observation, and its blob unless the part that replaced it has the same.
fn forget_uncommitted(
    tx: &Connection,
    observation: Option<&str>,
    blob: &BlobHash,
    replacing: &BlobHash,
) -> Result<()> {
    if let Some(id) = observation {
        tx.execute(
            "DELETE FROM observation_container WHERE observation_id = ?1",
            [id],
        )?;
        tx.execute("DELETE FROM observation WHERE id = ?1", [id])?;
    }
    if blob != replacing {
        tx.execute(
            "DELETE FROM blob WHERE hash = ?1 AND NOT EXISTS (SELECT 1 FROM extent WHERE blob = ?1)",
            [&blob.0[..]],
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;

/// The primary cache's persistent identity, independent of read cursor generations.
pub fn cache_identity(index: &Connection) -> Result<String> {
    index
        .query_row(
            "SELECT value FROM meta WHERE key='primary_cache'",
            [],
            |row| row.get(0),
        )
        .optional()?
        .filter(|value: &String| !value.is_empty())
        .ok_or_else(|| StoreError::Invalid("the primary cache has no identity".into()))
}
