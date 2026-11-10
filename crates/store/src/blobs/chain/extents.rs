use super::*;
use crate::encoding::Columns;
use rusqlite::{OpenFlags, OptionalExtension, params};
use std::sync::Mutex;

/// A chain's extent list, read where it is stored through the chain's own
/// read-only handle, inside one read transaction held for the chain's life:
/// every read sees the list as it was captured, without copying it.
#[derive(Debug)]
pub(super) struct ExtentList {
    conn: Mutex<Connection>,
    tables: &'static str,
    input: String,
    chain: i64,
}

pub(super) struct Extent {
    pub offset: u64,
    pub length: u64,
    pub hash: BlobHash,
    pub blob_offset: u64,
    pub size: u64,
}

impl ExtentList {
    /// Opens the handle and checks the extents covering `start..length` once:
    /// bounds within their blobs and no overlaps, so a read never selects a
    /// different source by its position.
    pub fn capture(
        source: &ChainSource,
        input: &str,
        chain: i64,
        start: u64,
        length: u64,
    ) -> Result<(Self, Option<BlobHash>)> {
        let conn = Connection::open_with_flags(
            &source.path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        conn.busy_timeout(std::time::Duration::from_secs(30))?;
        conn.execute_batch("PRAGMA cache_size=-256; PRAGMA mmap_size=0; BEGIN")?;
        let tables = source.tables;
        let mut single = None;
        {
            let mut stmt = conn.prepare(&format!(
                "SELECT e.offset,e.length,e.blob,e.blob_offset,b.size
                 FROM {tables}extent e JOIN {tables}blob b ON b.hash=e.blob
                 WHERE e.input=?1 AND e.chain=?2 AND e.offset<?3 AND e.offset+e.length>?4
                 ORDER BY e.offset"
            ))?;
            let mut rows = stmt.query(params![input, chain, length as i64, start as i64])?;
            let mut count = 0u64;
            let mut previous_end = None;
            while let Some(row) = rows.next()? {
                let offset: u64 = row.get(0)?;
                let len: u64 = row.get(1)?;
                let hash = BlobHash(row.hash(2)?);
                let blob_offset: u64 = row.get(3)?;
                let size: u64 = row.get(4)?;
                if len == 0
                    || offset.checked_add(len).is_none()
                    || blob_offset.checked_add(len).is_none_or(|end| end > size)
                    || previous_end.is_some_and(|end| offset < end)
                {
                    return Err(invalid("invalid stored extent bounds"));
                }
                previous_end = Some(offset + len);
                count += 1;
                single = (count == 1
                    && offset == 0
                    && len == length
                    && size == length
                    && blob_offset == 0)
                    .then_some(hash);
            }
        }
        Ok((
            Self {
                conn: Mutex::new(conn),
                tables,
                input: input.to_owned(),
                chain,
            },
            single,
        ))
    }

    #[cfg(test)]
    pub(super) fn connection(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap()
    }

    /// The extent holding byte `at`.
    pub fn select(&self, at: u64) -> Result<Extent> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| invalid("stored extents lock poisoned"))?;
        let tables = self.tables;
        let row = conn
            .prepare_cached(&format!(
                "SELECT e.offset,e.length,e.blob,e.blob_offset,b.size
                 FROM {tables}extent e JOIN {tables}blob b ON b.hash=e.blob
                 WHERE e.input=?1 AND e.chain=?2 AND e.offset<=?3
                 ORDER BY e.offset DESC LIMIT 1"
            ))?
            .query_row(params![self.input, self.chain, at as i64], |r| {
                Ok(Extent {
                    offset: r.get(0)?,
                    length: r.get(1)?,
                    hash: BlobHash(r.hash(2)?),
                    blob_offset: r.get(3)?,
                    size: r.get(4)?,
                })
            })
            .optional()?;
        let extent = row.ok_or_else(|| invalid("stored extent missing"))?;
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
fn invalid(message: &str) -> StoreError {
    StoreError::Invalid(message.into())
}
