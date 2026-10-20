//! Private, operation-scoped external ordering and ownership for export arrays.
use crate::{Result, StoreError};
use polylogue_model::SessionId;
use rusqlite::{Connection, OptionalExtension, params};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Range;

pub struct ExportArray {
    db: Connection,
    _index: tempfile::NamedTempFile,
    payload: File,
    ordinal: i64,
    cursor: Option<(i64, Vec<u8>, i64)>,
}

pub struct Item {
    pub range: Range<u64>,
    pub reader: Box<dyn Read>,
}

impl ExportArray {
    pub fn new() -> Result<Self> {
        let index = tempfile::NamedTempFile::new()?;
        let db = Connection::open(index.path())?;
        db.execute_batch("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-1024;
            PRAGMA mmap_size=0; PRAGMA temp_store=FILE;
            CREATE TABLE item(ordinal INTEGER PRIMARY KEY, missing INTEGER NOT NULL, sort BLOB NOT NULL,
                start INTEGER NOT NULL, end INTEGER NOT NULL, compressed_start INTEGER NOT NULL, compressed_end INTEGER NOT NULL);
            CREATE INDEX item_order ON item(missing,sort,ordinal);
            CREATE TABLE owner(message TEXT PRIMARY KEY,session TEXT NOT NULL) WITHOUT ROWID;")?;
        Ok(Self {
            db,
            _index: index,
            payload: tempfile::tempfile()?,
            ordinal: 0,
            cursor: None,
        })
    }

    pub fn push(&mut self, range: Range<u64>, key: Option<f64>, raw: &[u8]) -> Result<()> {
        let start = self.payload.stream_position()?;
        let mut encoder = zstd::stream::write::Encoder::new(&mut self.payload, 3)?;
        encoder.write_all(raw)?;
        encoder.finish()?;
        let end = self.payload.stream_position()?;
        // This byte key matches f64::total_cmp, including negative zero.
        let bits = key.unwrap_or(0.0).to_bits();
        let sort = if bits >> 63 != 0 {
            !bits
        } else {
            bits ^ (1 << 63)
        }
        .to_be_bytes();
        self.db.execute(
            "INSERT INTO item VALUES (?1,?2,?3,?4,?5,?6,?7)",
            params![
                self.ordinal,
                i64::from(key.is_none()),
                &sort[..],
                range.start as i64,
                range.end as i64,
                start as i64,
                end as i64
            ],
        )?;
        self.ordinal += 1;
        Ok(())
    }

    pub fn next_item(&mut self) -> Result<Option<Item>> {
        let (missing, sort, ordinal) = self.cursor.clone().unwrap_or((-1, vec![], -1));
        let row = self.db.query_row("SELECT missing,sort,ordinal,start,end,compressed_start,compressed_end
            FROM item WHERE (missing,sort,ordinal) > (?1,?2,?3) ORDER BY missing,sort,ordinal LIMIT 1",
            params![missing, sort, ordinal], |r| Ok((r.get::<_,i64>(0)?,r.get::<_,Vec<u8>>(1)?,r.get::<_,i64>(2)?,
                r.get::<_,i64>(3)?,r.get::<_,i64>(4)?,r.get::<_,i64>(5)?,r.get::<_,i64>(6)?))).optional()?;
        let Some((missing, sort, ordinal, start, end, compressed_start, compressed_end)) = row
        else {
            return Ok(None);
        };
        self.cursor = Some((missing, sort, ordinal));
        let mut file = self.payload.try_clone()?;
        file.seek(SeekFrom::Start(compressed_start as u64))?;
        let reader = zstd::stream::read::Decoder::new(
            file.take((compressed_end - compressed_start) as u64),
        )?;
        Ok(Some(Item {
            range: start as u64..end as u64,
            reader: Box::new(reader),
        }))
    }

    pub fn claim_message(&mut self, message: &str, session: &SessionId) -> Result<SessionId> {
        self.db.execute(
            "INSERT OR IGNORE INTO owner VALUES (?1,?2)",
            params![message, session.to_string()],
        )?;
        let owner: String = self.db.query_row(
            "SELECT session FROM owner WHERE message=?1",
            [message],
            |r| r.get(0),
        )?;
        SessionId::parse(&owner).map_err(|e| StoreError::Invalid(e.to_string()))
    }
}

#[cfg(test)]
mod tests;
