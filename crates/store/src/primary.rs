//! Primary mode: Sinex holds the archive. Each step of an input's appended
//! bytes goes to Sinex as one part of that input, with the [`InputRows`]
//! that place it; every other change to the register (a look that found no
//! new bytes, a request a person or agent makes, an excision) goes as a
//! register material. The writer commits a register row only
//! after Sinex acknowledged what it describes, so the local register
//! (`sinex.db`) and `index.db` are caches: deleted, they are rebuilt from
//! what Sinex holds ([`load`]). `polylogue-sinex` implements [`Sinex`].
use crate::input_rows::InputRows;
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
    /// Sends a register change and the events it makes, and waits until
    /// Sinex acknowledged them. Returns the register material holding it.
    fn send_change(&self, change: &RegisterChange) -> Result<String>;
    /// Excises these materials, with everything read or derived from them,
    /// through Sinex's excision with Polylogue as the actor. Sinex refuses
    /// the whole request when it cannot remove one cleanly.
    fn excise(&self, materials: &[String]) -> Result<()>;
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
    pub rows: InputRows,
}

/// What an input is, in words Sinex keeps: the origin that wrote it, its
/// path (or `drive:<id>`, an import's file name, …) and the host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    /// An excision: what it removed, by identity, so a rebuild keeps
    /// refusing it.
    Excision(ExcisionRows),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExcisionRows {
    pub id: String,
    /// `session:<id>`, `document:<id>` or `input:<id>`.
    pub target: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub created_at: i64,
    /// The `excised` rows: what was removed, by kind and identity.
    pub excised: Vec<(String, String)>,
    /// What stays of the removed inputs: their observations, the last of
    /// them `excised`.
    pub observations: Vec<InputRows>,
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
        rows: Box<InputRows>,
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
            Held::Part { number, .. } | Held::Change { number, .. } => *number,
        }
    }
}

/// The Sinex material holding each part, by where its bytes sit in their
/// chain, and each register material, by the observation, request or
/// excision it holds. Only `sinex.db` has them.
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
  id            TEXT PRIMARY KEY,        -- the observation, request or excision ID
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
                .map_err(|_| StoreError::Invalid("too many parts in one chain".into()))?,
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
        RegisterChange::Excision(excision) => (excision.id.as_str(), None),
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
        .ok_or_else(|| StoreError::Invalid("a part needs an extent".into()))?;
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
    store: &hashed_blobs::Store,
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
                    index,
                    rows,
                    ..
                } => {
                    let stored_size = store.stored_size(&hashed_blobs::Hash(hash.0))?;
                    tx.prepare_cached(
                        "INSERT OR IGNORE INTO blob(hash, size, stored_size, place, encoding, state)
                         VALUES (?1, ?2, ?3, 'file', 'plain', 'present')",
                    )?
                    .execute(params![&hash.0[..], length as i64, stored_size as i64])?;
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
                Held::Change {
                    material, change, ..
                } => {
                    match &change {
                        RegisterChange::Observation { stored, .. } => stored.insert(&tx, seq)?,
                        RegisterChange::Request { request, .. } => {
                            crate::user_writes::apply(&tx, request, seq)?;
                        }
                        RegisterChange::Excision(excision) => insert_excision(&tx, excision, seq)?,
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
            "DELETE FROM observation_zip WHERE observation_id = ?1",
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

/// The Sinex materials holding `inputs`: their parts and the register
/// materials of looks at them.
pub(crate) fn materials_of_inputs(
    conn: &Connection,
    inputs: &std::collections::BTreeSet<String>,
) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for input in inputs {
        for sql in [
            "SELECT id FROM material WHERE input = ?1 ORDER BY chain, offset",
            "SELECT material FROM register_material WHERE input = ?1 ORDER BY id",
        ] {
            out.extend(
                conn.prepare_cached(sql)?
                    .query_map([input], |r| r.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<_>>>()?,
            );
        }
    }
    Ok(out)
}

/// The Sinex materials holding the requests that wrote `notes`.
pub(crate) fn materials_of_requests(conn: &Connection, ids: &[String]) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for id in ids {
        if let Some(material) = conn
            .prepare_cached("SELECT material FROM register_material WHERE id = ?1")?
            .query_row([id], |r| r.get::<_, String>(0))
            .optional()?
        {
            out.push(material);
        }
    }
    Ok(out)
}

/// Forgets the register materials of requests `ids` once Sinex excised them.
pub(crate) fn forget_requests(conn: &Connection, ids: &[String]) -> Result<()> {
    for id in ids {
        conn.execute("DELETE FROM register_material WHERE id = ?1", [id])?;
    }
    Ok(())
}

/// Forgets the parts and register materials of `inputs` once Sinex excised
/// them.
pub(crate) fn forget_inputs(
    conn: &Connection,
    inputs: &std::collections::BTreeSet<String>,
) -> Result<()> {
    for input in inputs {
        conn.execute("DELETE FROM material WHERE input = ?1", [input])?;
        conn.execute("DELETE FROM register_material WHERE input = ?1", [input])?;
    }
    Ok(())
}

/// The rows of a committed excision `id`, as Sinex keeps them.
pub(crate) fn excision_rows(conn: &Connection, id: &str) -> Result<ExcisionRows> {
    let (target, reason, created_at) = conn.query_row(
        "SELECT target, reason, created_at FROM excision WHERE id = ?1",
        [id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )?;
    let excised: Vec<(String, String)> = conn
        .prepare("SELECT kind, value FROM excised WHERE excision_id = ?1 ORDER BY kind, value")?
        .query_map([id], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let mut observations = Vec::new();
    for (kind, input) in &excised {
        if kind == "input" {
            observations.extend(crate::input_rows::observations(conn, input)?);
        }
    }
    Ok(ExcisionRows {
        id: id.to_owned(),
        target,
        reason,
        created_at,
        excised,
        observations,
    })
}

/// Puts an excision's rows into the register, when rebuilding from Sinex or
/// importing user state: its rows, and the note targets naming what it
/// removed (the notes it removed whole went with their requests).
pub(crate) fn insert_excision(tx: &Connection, excision: &ExcisionRows, seq: i64) -> Result<()> {
    tx.execute(
        "INSERT OR IGNORE INTO excision(id, target, reason, created_at, commit_seq)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            excision.id,
            excision.target,
            excision.reason,
            excision.created_at,
            seq
        ],
    )?;
    for (kind, value) in &excision.excised {
        tx.execute(
            "INSERT OR IGNORE INTO excised(excision_id, kind, value) VALUES (?1, ?2, ?3)",
            params![excision.id, kind, value],
        )?;
    }
    for observed in &excision.observations {
        observed.insert(tx, seq)?;
    }
    let of = |kind: &str| -> Vec<String> {
        excision
            .excised
            .iter()
            .filter(|(k, _)| k == kind)
            .map(|(_, v)| v.clone())
            .collect()
    };
    let sessions = of("session");
    let messages: std::collections::BTreeSet<String> = of("message").into_iter().collect();
    let mut references = crate::excise::index_rows::note_references(tx, &sessions, &messages)?;
    references.extend(of("document").into_iter().map(|d| format!("document:{d}")));
    crate::excise::index_rows::remove_notes_about(tx, &references)?;
    Ok(())
}

#[cfg(test)]
mod tests;
