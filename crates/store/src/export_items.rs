//! Export items sorted in a temporary database, private to one import.
use crate::{Result, StoreError};
use polylogue_model::{MessageId, SessionId};
use rusqlite::{Connection, OptionalExtension, params};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::Path;

pub struct ExportItems {
    db: Connection,
    payload: File,
    ordinal: i64,
    cursor: Option<(Vec<u8>, i64)>,
    order: Order,
    scan_group: Option<(i64, u64)>,
    retained: bool,
}

/// Closed scan orders over the same staged payload bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    Primary,
    Alternate,
}

pub struct Item {
    pub range: Range<u64>,
    pub reader: Box<dyn Read>,
}

struct RangeReader {
    file: File,
    offset: u64,
    end: u64,
}

impl Read for RangeReader {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        let length = (self.end - self.offset).min(bytes.len() as u64) as usize;
        let n = self.file.read_at(&mut bytes[..length], self.offset)?;
        self.offset += n as u64;
        Ok(n)
    }
}

impl ExportItems {
    pub fn new() -> Result<Self> {
        Self::open(Connection::open("")?, tempfile::tempfile()?, false)
    }

    /// Retained scratch uses the same single compressed payload and ordering
    /// database. Opening it again resumes the staged records, never the cursor.
    pub fn in_directory(directory: &Path) -> Result<Self> {
        fs::create_dir_all(directory)?;
        let db = Connection::open(directory.join("items.db"))?;
        let payload = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(directory.join("payload.zst"))?;
        Self::open(db, payload, true)
    }

    fn open(db: Connection, payload: File, retained: bool) -> Result<Self> {
        db.execute_batch(if retained {
            "PRAGMA synchronous=FULL;"
        } else {
            "PRAGMA synchronous=OFF;"
        })?;
        db.execute_batch("PRAGMA journal_mode=DELETE; PRAGMA cache_size=-1024;
            PRAGMA mmap_size=0; PRAGMA temp_store=FILE;
            CREATE TABLE IF NOT EXISTS item(ordinal INTEGER PRIMARY KEY, source_key TEXT UNIQUE, sort BLOB NOT NULL, alternate_sort BLOB NOT NULL,
                start INTEGER NOT NULL, end INTEGER NOT NULL, compressed_start INTEGER NOT NULL, compressed_end INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS item_order ON item(sort,ordinal);
            CREATE INDEX IF NOT EXISTS item_alternate_order ON item(alternate_sort,ordinal);
            CREATE TABLE IF NOT EXISTS owner(message TEXT PRIMARY KEY,session TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS previous(session TEXT PRIMARY KEY,message TEXT NOT NULL) WITHOUT ROWID; BEGIN;")?;
        let ordinal = db.query_row("SELECT COALESCE(MAX(ordinal)+1,0) FROM item", [], |r| {
            r.get(0)
        })?;
        Ok(Self {
            db,
            payload,
            ordinal,
            cursor: None,
            order: Order::Primary,
            scan_group: None,
            retained,
        })
    }

    /// Commit ordering only after its referenced compressed bytes are durable.
    /// This is also used before retaining a failed client's working directory.
    pub fn checkpoint(&mut self) -> Result<()> {
        if self.scan_group.is_some() {
            return Err(StoreError::Invalid(
                "export scan group is incomplete".into(),
            ));
        }
        self.payload.sync_data()?;
        self.db.execute_batch("COMMIT; BEGIN;")?;
        Ok(())
    }

    pub fn close(mut self) -> Result<()> {
        self.checkpoint()?;
        self.retained = false;
        Ok(())
    }

    /// Stage original bytes and their range before reading ordered items.
    /// Reusing a source key replaces its bytes and ordering; `None` keeps each item.
    pub fn push(
        &mut self,
        range: Range<u64>,
        key: Vec<u8>,
        source_key: Option<&str>,
        raw: &[u8],
    ) -> Result<()> {
        self.push_ordered(range, key.clone(), key, source_key, raw)
    }

    /// Two traversal orders share one compressed payload and source identity.
    pub fn push_ordered(
        &mut self,
        range: Range<u64>,
        key: Vec<u8>,
        alternate_key: Vec<u8>,
        source_key: Option<&str>,
        raw: &[u8],
    ) -> Result<()> {
        let start = self.payload.seek(SeekFrom::End(0))?;
        let mut encoder = zstd::stream::write::Encoder::new(&mut self.payload, 3)?;
        encoder.write_all(raw)?;
        encoder.finish()?;
        let end = self.payload.stream_position()?;
        self.db.execute(
            "INSERT INTO item VALUES (?1,?2,?3,?4,?5,?6,?7,?8)
             ON CONFLICT(source_key) DO UPDATE SET ordinal=excluded.ordinal,sort=excluded.sort,alternate_sort=excluded.alternate_sort,
                start=excluded.start,end=excluded.end,compressed_start=excluded.compressed_start,
                compressed_end=excluded.compressed_end",
            params![
                self.ordinal,
                source_key,
                key,
                alternate_key,
                range.start as i64,
                range.end as i64,
                start as i64,
                end as i64
            ],
        )?;
        self.ordinal += 1;
        Ok(())
    }

    /// Restarts traversal without rereading or duplicating staged bytes.
    pub fn reset(&mut self, order: Order) -> Result<()> {
        if self.scan_group.is_some() {
            return Err(StoreError::Invalid(
                "export scan group is incomplete".into(),
            ));
        }
        self.order = order;
        self.cursor = None;
        Ok(())
    }

    pub fn source_item(&self, source_key: &str) -> Result<Option<Item>> {
        let row = self
            .db
            .query_row(
                "SELECT start,end,compressed_start,compressed_end FROM item WHERE source_key=?1",
                [source_key],
                |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, i64>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, i64>(3)?,
                    ))
                },
            )
            .optional()?;
        row.map(|(start, end, compressed_start, compressed_end)| {
            self.item(start, end, compressed_start, compressed_end)
        })
        .transpose()
    }

    pub fn next_item(&mut self) -> Result<Option<Item>> {
        if self.scan_group.is_some() {
            return Err(StoreError::Invalid(
                "export scan group is incomplete".into(),
            ));
        }
        let (sort, ordinal) = self.cursor.clone().unwrap_or((vec![], -1));
        let column = match self.order {
            Order::Primary => "sort",
            Order::Alternate => "alternate_sort",
        };
        let row = self
            .db
            .query_row(
                &format!(
                    "SELECT {column},ordinal,start,end,compressed_start,compressed_end
            FROM item WHERE ({column},ordinal)>(?1,?2) ORDER BY {column},ordinal LIMIT 1"
                ),
                params![sort, ordinal],
                |r| {
                    Ok((
                        r.get::<_, Vec<u8>>(0)?,
                        r.get::<_, i64>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, i64>(3)?,
                        r.get::<_, i64>(4)?,
                        r.get::<_, i64>(5)?,
                    ))
                },
            )
            .optional()?;
        let Some((sort, ordinal, start, end, compressed_start, compressed_end)) = row else {
            return Ok(None);
        };
        self.cursor = Some((sort, ordinal));
        Ok(Some(self.item(
            start,
            end,
            compressed_start,
            compressed_end,
        )?))
    }

    fn item(
        &self,
        start: i64,
        end: i64,
        compressed_start: i64,
        compressed_end: i64,
    ) -> Result<Item> {
        let file = RangeReader {
            file: self.payload.try_clone()?,
            offset: compressed_start as u64,
            end: compressed_end as u64,
        };
        let reader = zstd::stream::read::Decoder::new(file)?;
        Ok(Item {
            range: start as u64..end as u64,
            reader: Box::new(reader),
        })
    }

    /// One scan group can be accepted or discarded before ordered reading.
    pub fn begin_scan(&mut self) -> Result<()> {
        if self.scan_group.is_some() || self.cursor.is_some() {
            return Err(StoreError::Invalid(
                "an export scan group is already active or items are being read".into(),
            ));
        }
        let at = self.payload.stream_position()?;
        self.db.execute_batch("SAVEPOINT scan_group")?;
        self.scan_group = Some((self.ordinal, at));
        Ok(())
    }
    pub fn commit_scan(&mut self) -> Result<()> {
        if self.scan_group.is_none() {
            return Err(StoreError::Invalid("no export scan group".into()));
        }
        self.db.execute_batch("RELEASE scan_group")?;
        self.scan_group = None;
        Ok(())
    }
    pub fn rollback_scan(&mut self) -> Result<()> {
        let (ordinal, at) = self
            .scan_group
            .ok_or_else(|| StoreError::Invalid("no export scan group".into()))?;
        self.db
            .execute_batch("ROLLBACK TO scan_group; RELEASE scan_group")?;
        self.payload.set_len(at)?;
        self.payload.seek(SeekFrom::Start(at))?;
        self.ordinal = ordinal;
        self.scan_group = None;
        Ok(())
    }

    pub fn claim_owner(&mut self, message: &str, session: &SessionId) -> Result<Option<SessionId>> {
        let inserted = self.db.execute(
            "INSERT OR IGNORE INTO owner VALUES (?1,?2)",
            params![message, session.to_string()],
        )?;
        if inserted == 1 {
            return Ok(None);
        }
        let owner: String = self.db.query_row(
            "SELECT session FROM owner WHERE message=?1",
            [message],
            |r| r.get(0),
        )?;
        SessionId::parse(&owner)
            .map(Some)
            .map_err(|e| StoreError::Invalid(e.to_string()))
    }
    pub fn last_message(&self, session: &SessionId) -> Result<Option<MessageId>> {
        let message: Option<String> = self
            .db
            .query_row(
                "SELECT message FROM previous WHERE session=?1",
                [session.to_string()],
                |r| r.get(0),
            )
            .optional()?;
        message
            .map(|id| MessageId::parse(&id).map_err(|e| StoreError::Invalid(e.to_string())))
            .transpose()
    }
    pub fn set_last_message(&mut self, session: &SessionId, message: &MessageId) -> Result<()> {
        self.db.execute("INSERT INTO previous VALUES (?1,?2) ON CONFLICT(session) DO UPDATE SET message=excluded.message", params![session.to_string(),message.to_string()])?;
        Ok(())
    }
}

impl Drop for ExportItems {
    fn drop(&mut self) {
        if self.retained {
            if self.scan_group.is_some() {
                let _ = self.rollback_scan();
            }
            let _ = self.checkpoint();
        }
    }
}

#[cfg(test)]
#[path = "export_items/tests.rs"]
mod tests;
