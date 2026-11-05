use super::*;
use rusqlite::{OptionalExtension, params};
use std::sync::Mutex;

#[derive(Debug)]
pub(super) struct ExtentList {
    conn: Mutex<Connection>,
}

pub(super) struct Extent {
    pub offset: u64,
    pub length: u64,
    pub hash: BlobHash,
    pub blob_offset: u64,
    pub size: u64,
}

impl ExtentList {
    pub fn capture(
        archive: &Connection,
        schema: &str,
        input: &str,
        chain: i64,
        start: u64,
        length: u64,
    ) -> Result<(Self, Option<BlobHash>)> {
        let target = Connection::open("")?;
        target.execute_batch("PRAGMA cache_size=-1024; PRAGMA mmap_size=0; PRAGMA temp_store=FILE;
            PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
            CREATE TABLE extent(offset INTEGER PRIMARY KEY,length INTEGER NOT NULL,hash BLOB NOT NULL,blob_offset INTEGER NOT NULL,size INTEGER NOT NULL);
            BEGIN;")?;
        let mut stmt = archive.prepare(&format!("SELECT e.offset,e.length,e.blob,e.blob_offset,b.size
            FROM {schema}extent e JOIN {schema}blob b ON b.hash=e.blob
            WHERE e.input=?1 AND e.chain=?2 AND e.offset<?3 AND e.offset+e.length>?4 ORDER BY e.offset"))?;
        let mut rows = stmt.query(params![input, chain, length as i64, start as i64])?;
        let mut count = 0u64;
        let mut previous_end = None;
        let mut single = None;
        while let Some(row) = rows.next()? {
            let offset = number(row.get(0)?)?;
            let len = number(row.get(1)?)?;
            let hash: Vec<u8> = row.get(2)?;
            let hash = BlobHash(
                hash.try_into()
                    .map_err(|_| invalid("blob hash is not 32 bytes"))?,
            );
            let blob_offset = number(row.get(3)?)?;
            let size = number(row.get(4)?)?;
            if len == 0
                || offset.checked_add(len).is_none()
                || blob_offset.checked_add(len).is_none_or(|end| end > size)
                || previous_end.is_some_and(|end| offset < end)
            {
                return Err(invalid("invalid stored extent bounds"));
            }
            previous_end = Some(offset + len);
            target.execute(
                "INSERT INTO extent VALUES (?1,?2,?3,?4,?5)",
                params![
                    offset as i64,
                    len as i64,
                    hash.0.as_slice(),
                    blob_offset as i64,
                    size as i64
                ],
            )?;
            count += 1;
            single =
                (count == 1 && offset == 0 && len == length && size == length && blob_offset == 0)
                    .then_some(hash);
        }
        target.execute_batch("COMMIT; PRAGMA cache_size=-64; PRAGMA shrink_memory;")?;
        Ok((
            Self {
                conn: Mutex::new(target),
            },
            single,
        ))
    }

    #[cfg(test)]
    pub(super) fn connection(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap()
    }

    pub fn select(&self, at: u64) -> Result<Extent> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| invalid("stored extents lock poisoned"))?;
        Self::extent(&conn, at)
    }

    pub fn extent(conn: &Connection, at: u64) -> Result<Extent> {
        let row = conn.prepare_cached("SELECT offset,length,hash,blob_offset,size FROM extent WHERE offset<=?1 ORDER BY offset DESC LIMIT 1")?
            .query_row([at as i64], |r| Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?,r.get::<_,Vec<u8>>(2)?,r.get::<_,i64>(3)?,r.get::<_,i64>(4)?))).optional()?;
        let (offset, length, hash, blob_offset, size) =
            row.ok_or_else(|| invalid("stored extent missing"))?;
        let extent = Extent {
            offset: number(offset)?,
            length: number(length)?,
            hash: BlobHash(
                hash.try_into()
                    .map_err(|_| invalid("blob hash is not 32 bytes"))?,
            ),
            blob_offset: number(blob_offset)?,
            size: number(size)?,
        };
        if extent
            .offset
            .checked_add(extent.length)
            .is_none_or(|end| end <= at)
        {
            return Err(invalid("stored extent gap"));
        }
        Ok(extent)
    }
}
fn number(n: i64) -> Result<u64> {
    u64::try_from(n).map_err(|_| invalid("negative stored extent metadata"))
}
fn invalid(message: &str) -> StoreError {
    StoreError::Invalid(message.into())
}
