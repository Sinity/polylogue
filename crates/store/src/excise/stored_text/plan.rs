//! Private staging for one text excision, discarded with its verified bytes.

use super::{Chain, Ranges};
use crate::{Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::{BTreeMap, BTreeSet};

mod edits;
mod frames;
mod matching;
pub(super) use edits::Edits;
mod rewrite;

pub(super) struct Plan {
    conn: Connection,
}

fn position(value: u64) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| StoreError::Invalid("text-excision position exceeds SQLite range".into()))
}

impl Plan {
    pub(super) fn new() -> Result<Self> {
        let conn = Connection::open("")?;
        conn.execute_batch(
            "PRAGMA cache_size=-1024; PRAGMA mmap_size=0; PRAGMA temp_store=FILE;
             PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
             CREATE TABLE matches(input TEXT,chain INTEGER,start INTEGER,end INTEGER,
                PRIMARY KEY(input,chain,start,end)) WITHOUT ROWID;
             CREATE TABLE raw_matches(input TEXT,chain INTEGER,start INTEGER,end INTEGER,
                PRIMARY KEY(input,chain,start,end)) WITHOUT ROWID;
             CREATE TABLE write_ranges(input TEXT,chain INTEGER,start INTEGER,end INTEGER,
                PRIMARY KEY(input,chain,start)) WITHOUT ROWID;
             CREATE TABLE patches(input TEXT,chain INTEGER,ordinal INTEGER,start INTEGER,
                end INTEGER,bytes BLOB,PRIMARY KEY(input,chain,ordinal)) WITHOUT ROWID;
             CREATE INDEX patches_start ON patches(input,chain,start);
             CREATE TABLE frames(input TEXT,chain INTEGER,ordinal INTEGER,start INTEGER,
                header BLOB,PRIMARY KEY(input,chain,ordinal)) WITHOUT ROWID;
             CREATE INDEX frames_start ON frames(input,chain,start);
             CREATE TABLE chains(input TEXT,chain INTEGER,framed INTEGER NOT NULL DEFAULT 0,
                patch_width INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(input,chain)) WITHOUT ROWID;
             CREATE TABLE places(hash BLOB PRIMARY KEY,start INTEGER,length INTEGER) WITHOUT ROWID;
             CREATE TABLE affected(hash BLOB PRIMARY KEY) WITHOUT ROWID;
             CREATE TABLE observations(ordinal INTEGER PRIMARY KEY,input TEXT,span TEXT);
             CREATE INDEX observations_input ON observations(input,ordinal);
             CREATE TABLE referring(ordinal INTEGER PRIMARY KEY,input TEXT,chain INTEGER,
                offset INTEGER,length INTEGER,blob BLOB,blob_offset INTEGER);
             BEGIN;",
        )?;
        Ok(Self { conn })
    }

    pub(super) fn matches(&self, input: &str, chain: i64, ranges: &Ranges) -> Result<()> {
        let mut insert = self.conn.prepare_cached(
            "INSERT OR IGNORE INTO matches(input,chain,start,end) VALUES (?1,?2,?3,?4)",
        )?;
        for &(start, end) in ranges {
            insert.execute(params![input, chain, position(start)?, position(end)?])?;
        }
        Ok(())
    }

    pub(super) fn patch(
        &self,
        input: &str,
        chain: i64,
        ordinal: i64,
        start: u64,
        bytes: &[u8],
    ) -> Result<()> {
        let end = start
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| StoreError::Invalid("text-excision patch end overflow".into()))?;
        self.conn.prepare_cached("INSERT INTO patches(input,chain,ordinal,start,end,bytes) VALUES (?1,?2,?3,?4,?5,?6)")?
            .execute(params![input,chain,ordinal,position(start)?,position(end)?,bytes])?;
        self.conn.execute("INSERT INTO chains(input,chain,patch_width) VALUES (?1,?2,?3)
            ON CONFLICT(input,chain) DO UPDATE SET patch_width=max(patch_width,excluded.patch_width)",params![input,chain,bytes.len() as i64])?;
        Ok(())
    }

    pub(super) fn placement(&self, hash: &BlobHash) -> Result<Option<(u64, u64)>> {
        Ok(self
            .conn
            .prepare_cached("SELECT start,length FROM places WHERE hash=?1")?
            .query_row([&hash.0[..]], |r| {
                Ok((r.get::<_, i64>(0)? as u64, r.get::<_, i64>(1)? as u64))
            })
            .optional()?)
    }

    pub(super) fn remember(&self, hash: &BlobHash, start: u64, length: u64) -> Result<()> {
        self.conn
            .prepare_cached("INSERT INTO places(hash,start,length) VALUES (?1,?2,?3)")?
            .execute(params![&hash.0[..], position(start)?, position(length)?])?;
        Ok(())
    }

    pub(super) fn inputs(&self) -> Result<BTreeSet<String>> {
        Ok(self
            .conn
            .prepare("SELECT DISTINCT input FROM matches ORDER BY input")?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?)
    }

    pub(super) fn counts(&self) -> Result<BTreeMap<String, u64>> {
        Ok(self
            .conn
            .prepare("SELECT input,count(*) FROM matches GROUP BY input ORDER BY input")?
            .query_map([], |r| Ok((r.get(0)?, r.get::<_, i64>(1)? as u64)))?
            .collect::<rusqlite::Result<_>>()?)
    }

    /// Keep matches before header patches within each chain. Overlapping
    /// unequal matches remain separate, including in observation counts.
    fn changed(&self, mut visit: impl FnMut(&str, i64, u64, u64) -> Result<()>) -> Result<()> {
        let mut statement = self.conn.prepare(
            "SELECT input,chain,start,end FROM (
                SELECT input,chain,start,end,0 AS kind,start AS ordinal FROM matches
                UNION ALL SELECT input,chain,start,end,1 AS kind,ordinal FROM patches)
             ORDER BY input,chain,kind,ordinal,end",
        )?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let input = row.get_ref(0)?;
            visit(
                input
                    .as_str()
                    .map_err(|e| StoreError::Invalid(e.to_string()))?,
                row.get(1)?,
                row.get::<_, i64>(2)? as u64,
                row.get::<_, i64>(3)? as u64,
            )?;
        }
        Ok(())
    }

    pub(super) fn edits<'a>(&'a self, key: &'a Chain, from: u64) -> Result<Edits<'a>> {
        Edits::new(self, key, from)
    }
}

#[cfg(test)]
mod tests;
