//! Private, operation-scoped external ordering and ownership for export items.
use crate::{Result, StoreError};
use polylogue_model::{MessageId, SessionId};
use rusqlite::{Connection, OptionalExtension, params};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::ops::Range;

pub struct ExportItems {
    db: Connection,
    _index: tempfile::NamedTempFile,
    payload: File,
    ordinal: i64,
    cursor: Option<(Vec<u8>, i64)>,
    scan_group: Option<(i64, u64)>,
}

pub struct Item {
    pub range: Range<u64>,
    pub reader: Box<dyn Read>,
}

impl ExportItems {
    pub fn new() -> Result<Self> {
        let index = tempfile::NamedTempFile::new()?;
        let db = Connection::open(index.path())?;
        db.execute_batch("PRAGMA journal_mode=DELETE; PRAGMA synchronous=OFF; PRAGMA cache_size=-1024;
            PRAGMA mmap_size=0; PRAGMA temp_store=FILE;
            CREATE TABLE item(ordinal INTEGER PRIMARY KEY, source_key TEXT UNIQUE, sort BLOB NOT NULL,
                start INTEGER NOT NULL, end INTEGER NOT NULL, compressed_start INTEGER NOT NULL, compressed_end INTEGER NOT NULL);
            CREATE INDEX item_order ON item(sort,ordinal);
            CREATE TABLE owner(message TEXT PRIMARY KEY,session TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE previous(session TEXT PRIMARY KEY,message TEXT NOT NULL) WITHOUT ROWID; BEGIN;")?;
        Ok(Self {
            db,
            _index: index,
            payload: tempfile::tempfile()?,
            ordinal: 0,
            cursor: None,
            scan_group: None,
        })
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
        let start = self.payload.stream_position()?;
        let mut encoder = zstd::stream::write::Encoder::new(&mut self.payload, 3)?;
        encoder.write_all(raw)?;
        encoder.finish()?;
        let end = self.payload.stream_position()?;
        self.db.execute(
            "INSERT INTO item VALUES (?1,?2,?3,?4,?5,?6,?7)
             ON CONFLICT(source_key) DO UPDATE SET ordinal=excluded.ordinal,sort=excluded.sort,
                start=excluded.start,end=excluded.end,compressed_start=excluded.compressed_start,
                compressed_end=excluded.compressed_end",
            params![
                self.ordinal,
                source_key,
                key,
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
        if self.scan_group.is_some() {
            return Err(StoreError::Invalid(
                "export scan group is unfinished".into(),
            ));
        }
        let (sort, ordinal) = self.cursor.clone().unwrap_or((vec![], -1));
        let row = self
            .db
            .query_row(
                "SELECT sort,ordinal,start,end,compressed_start,compressed_end
            FROM item WHERE (sort,ordinal)>(?1,?2) ORDER BY sort,ordinal LIMIT 1",
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

#[cfg(test)]
mod tests;
