//! Private staging for one text excision, discarded with its verified bytes.

use super::{Chain, Ranges};
use crate::{Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::{Connection, OptionalExtension, params};
use std::collections::{BTreeMap, BTreeSet};

mod matching;
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

    pub(super) fn patches(
        &self,
        input: &str,
        chain: i64,
        patches: &[(u64, Vec<u8>)],
    ) -> Result<()> {
        let mut insert = self.conn.prepare_cached(
            "INSERT INTO patches(input,chain,ordinal,start,end,bytes) VALUES (?1,?2,?3,?4,?5,?6)",
        )?;
        for (ordinal, (start, bytes)) in patches.iter().enumerate() {
            let end = start
                .checked_add(bytes.len() as u64)
                .ok_or_else(|| StoreError::Invalid("text-excision patch end overflow".into()))?;
            insert.execute(params![
                input,
                chain,
                ordinal as i64,
                position(*start)?,
                position(end)?,
                bytes
            ])?;
        }
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

    pub(super) fn blank(&self, key: &Chain, at: u64, bytes: &mut [u8], fill: u8) -> Result<()> {
        let end = at
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| StoreError::Invalid("text-excision chunk end overflow".into()))?;
        let mut statement = self.conn.prepare_cached(
            "SELECT start,end FROM matches WHERE input=?1 AND chain=?2 AND start<?4 AND end>?3
             ORDER BY start,end",
        )?;
        let mut rows = statement.query(params![key.0, key.1, position(at)?, position(end)?])?;
        while let Some(row) = rows.next()? {
            let start = (row.get::<_, i64>(0)? as u64).max(at);
            let end = (row.get::<_, i64>(1)? as u64).min(end);
            for byte in &mut bytes[(start - at) as usize..(end - at) as usize] {
                if *byte != b'\n' {
                    *byte = fill;
                }
            }
        }
        Ok(())
    }

    pub(super) fn rewrite(&self, key: &Chain, at: u64, bytes: &mut [u8], fill: u8) -> Result<()> {
        self.blank(key, at, bytes, fill)?;
        let end = at
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| StoreError::Invalid("text-excision chunk end overflow".into()))?;
        let mut statement = self.conn.prepare_cached(
            "SELECT start,bytes FROM patches WHERE input=?1 AND chain=?2 AND start<?4 AND end>?3
             ORDER BY ordinal",
        )?;
        let mut rows = statement.query(params![key.0, key.1, position(at)?, position(end)?])?;
        while let Some(row) = rows.next()? {
            let start = row.get::<_, i64>(0)? as u64;
            let value = row.get_ref(1)?;
            let patch = value
                .as_blob()
                .map_err(|e| StoreError::Invalid(e.to_string()))?;
            let from = start.max(at);
            let end = (start + patch.len() as u64).min(end);
            bytes[(from - at) as usize..(end - at) as usize]
                .copy_from_slice(&patch[(from - start) as usize..(end - start) as usize]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
