//! The register rows one stored step of an input writes: its revision, the
//! extent its new bytes fill and the observation that found them. The writer
//! builds one [`InputRows`] per step and inserts it; in primary mode Sinex
//! keeps it with the bytes it describes (or alone, when the step found no new
//! bytes), and a rebuild from Sinex inserts the same rows again.
use crate::Result;
use polylogue_model::BlobHash;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Revision {
    pub number: i64,
    pub produced: String,
    pub chain: i64,
    pub length: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Extent {
    pub chain: i64,
    pub offset: i64,
    pub length: i64,
    pub blob: BlobHash,
    /// Offset into the original captured material after an exact prefix overlap.
    pub blob_offset: i64,
}

/// One look at a path that found something.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
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
    /// The exact container membership established by this observation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<crate::InContainer>,
}

impl InputRows {
    /// Inserts the rows at write counter `seq`. A revision or extent already
    /// present takes the recorded one: a later chunk grew the revision, or a
    /// write that stopped before its commit sent the bytes at that place
    /// again, longer. Everything else present is left as it is, so inserting
    /// the rows twice changes nothing.
    pub fn insert(&self, tx: &Connection, seq: i64) -> Result<()> {
        let source = crate::source::Source::of(tx)?;
        tx.prepare_cached(
            &source.sql("INSERT OR IGNORE INTO @source.input(id, format) VALUES (?1, ?2)"),
        )?
        .execute(params![self.input, self.format])?;
        if let Some(r) = &self.revision {
            tx.prepare_cached(&source.sql(
                "INSERT INTO @source.revision(input, number, produced, chain, length, commit_seq)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                 ON CONFLICT(input, number) DO UPDATE SET length = excluded.length",
            ))?
            .execute(params![
                self.input, r.number, r.produced, r.chain, r.length, seq
            ])?;
        }
        if let Some(e) = &self.extent {
            tx.prepare_cached(
                &source.sql("INSERT INTO @source.extent(input, chain, offset, length, blob, blob_offset, commit_seq)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                 ON CONFLICT(input,chain,offset) DO UPDATE SET
                     commit_seq=CASE WHEN length=excluded.length AND blob=excluded.blob AND blob_offset=excluded.blob_offset THEN min(commit_seq,excluded.commit_seq) ELSE excluded.commit_seq END,
                     length=excluded.length,blob=excluded.blob,blob_offset=excluded.blob_offset"),
            )?
            .execute(params![
                self.input,
                e.chain,
                e.offset,
                e.length,
                &e.blob.0[..],e.blob_offset,seq
            ])?;
        }
        if let Some(o) = &self.observation {
            tx.prepare_cached(&source.sql(
                "INSERT OR IGNORE INTO @source.machine(id, host, first_seen) VALUES (?1, ?2, ?3)",
            ))?
            .execute(params![o.machine, o.host, o.seen_at])?;
            tx.prepare_cached(
                &source.sql("INSERT OR IGNORE INTO @source.observation(id, input, machine, path, class, revision, length,
                     device, inode, size, mtime_ns, ctime_ns, polylogue_version, seen_at, commit_seq,
                     details_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)"),
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
            ])?;
            if let Some(c) = &o.container {
                tx.prepare_cached(
                    &source.sql("INSERT OR IGNORE INTO @source.observation_container VALUES (?1, ?2, ?3, ?4, ?5)"),
                )?
                .execute(params![
                    o.id,
                    c.input,
                    c.member_index as i64,
                    c.kind,
                    c.path_bytes
                ])?;
            }
        }
        Ok(())
    }
}
