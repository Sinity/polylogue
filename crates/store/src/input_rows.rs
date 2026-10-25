//! The register rows one stored step of an input writes: its revision, the
//! extent its new bytes fill and the observation that found them. The writer
//! builds one [`InputRows`] per step and inserts it; in primary mode Sinex
//! keeps it with the bytes it describes (or alone, when the step found no new
//! bytes), and a rebuild from Sinex inserts the same rows again.
use crate::Result;
use polylogue_model::BlobHash;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputRows {
    pub input: String,
    /// The origin that parses it (`claude-code`).
    pub format: String,
    /// A new revision, or the current one grown by a later chunk of the
    /// same read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<Revision>,
    /// Where the step's new bytes sit in the revision chain.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extent: Option<Extent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observation: Option<ObservationRow>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Revision {
    pub number: i64,
    pub produced: String,
    pub chain: i64,
    pub length: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Extent {
    pub chain: i64,
    pub offset: i64,
    pub length: i64,
    pub blob: BlobHash,
}

/// One look at a path that found something.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationRow {
    pub id: String,
    pub machine: String,
    pub host: String,
    pub path: String,
    pub class: String,
    pub revision: Option<i64>,
    pub length: Option<i64>,
    pub device: Option<i64>,
    pub inode: Option<i64>,
    pub size: Option<i64>,
    pub mtime_ns: Option<i64>,
    pub ctime_ns: Option<i64>,
    pub polylogue_version: String,
    pub seen_at: i64,
    pub details_json: Option<String>,
    /// For an `excised` observation, the excision that removed the input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub excision: Option<String>,
    /// The export ZIP it was read from, and its index there.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zip: Option<(String, i64)>,
}

impl InputRows {
    /// Inserts the rows at write counter `seq`. A revision or extent already
    /// present takes the recorded one: a later chunk grew the revision, or a
    /// write that stopped before its commit sent the bytes at that place
    /// again, longer. Everything else present is left as it is, so inserting
    /// the rows twice changes nothing.
    pub fn insert(&self, tx: &Connection, seq: i64) -> Result<()> {
        tx.prepare_cached("INSERT OR IGNORE INTO input(id, format) VALUES (?1, ?2)")?
            .execute(params![self.input, self.format])?;
        if let Some(r) = &self.revision {
            tx.prepare_cached(
                "INSERT INTO revision(input, number, produced, chain, length, commit_seq)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(input, number) DO UPDATE SET length = excluded.length",
            )?
            .execute(params![
                self.input, r.number, r.produced, r.chain, r.length, seq
            ])?;
        }
        if let Some(e) = &self.extent {
            tx.prepare_cached(
                "INSERT OR REPLACE INTO extent(input, chain, offset, length, blob, blob_offset)
                 VALUES (?1, ?2, ?3, ?4, ?5, 0)",
            )?
            .execute(params![
                self.input,
                e.chain,
                e.offset,
                e.length,
                &e.blob.0[..]
            ])?;
        }
        if let Some(o) = &self.observation {
            tx.prepare_cached(
                "INSERT OR IGNORE INTO machine(id, host, first_seen) VALUES (?1, ?2, ?3)",
            )?
            .execute(params![o.machine, o.host, o.seen_at])?;
            tx.prepare_cached(
                "INSERT OR IGNORE INTO observation(id, input, machine, path, class, revision, length,
                     device, inode, size, mtime_ns, ctime_ns, polylogue_version, seen_at, commit_seq,
                     details_json, excision)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            )?
            .execute(params![
                o.id,
                self.input,
                o.machine,
                o.path,
                o.class,
                o.revision,
                o.length,
                o.device,
                o.inode,
                o.size,
                o.mtime_ns,
                o.ctime_ns,
                o.polylogue_version,
                o.seen_at,
                seq,
                o.details_json,
                o.excision,
            ])?;
            if let Some((zip, index)) = &o.zip {
                tx.prepare_cached("INSERT OR IGNORE INTO observation_zip VALUES (?1, ?2, ?3)")?
                    .execute(params![o.id, zip, index])?;
            }
        }
        Ok(())
    }
}

/// Every observation of `input`, as input rows, in the order they were made.
pub(crate) fn observations(conn: &Connection, input: &str) -> Result<Vec<InputRows>> {
    let format: String =
        conn.query_row("SELECT format FROM input WHERE id = ?1", [input], |r| {
            r.get(0)
        })?;
    let mut statement = conn.prepare(
        "SELECT o.id, o.machine, coalesce((SELECT m.host FROM machine m WHERE m.id = o.machine
                 ORDER BY m.first_seen LIMIT 1), ''),
             o.path, o.class, o.revision, o.length, o.device, o.inode, o.size, o.mtime_ns,
             o.ctime_ns, o.polylogue_version, o.seen_at, o.details_json, o.excision,
             c.zip_input, c.member_index
         FROM observation o LEFT JOIN observation_zip c ON c.observation_id = o.id
         WHERE o.input = ?1 ORDER BY o.rowid",
    )?;
    let rows = statement.query_map([input], |r| {
        let zip: Option<String> = r.get(16)?;
        let index: Option<i64> = r.get(17)?;
        Ok(ObservationRow {
            id: r.get(0)?,
            machine: r.get(1)?,
            host: r.get(2)?,
            path: r.get(3)?,
            class: r.get(4)?,
            revision: r.get(5)?,
            length: r.get(6)?,
            device: r.get(7)?,
            inode: r.get(8)?,
            size: r.get(9)?,
            mtime_ns: r.get(10)?,
            ctime_ns: r.get(11)?,
            polylogue_version: r.get(12)?,
            seen_at: r.get(13)?,
            details_json: r.get(14)?,
            excision: r.get(15)?,
            zip: zip.zip(index),
        })
    })?;
    rows.map(|row| {
        Ok(InputRows {
            input: input.to_owned(),
            format: format.clone(),
            revision: None,
            extent: None,
            observation: Some(row?),
        })
    })
    .collect()
}
