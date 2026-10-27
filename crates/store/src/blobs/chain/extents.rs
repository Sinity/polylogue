use super::*;
use rusqlite::{MAIN_DB, OptionalExtension, params};
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
    pub inline: bool,
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
            CREATE TABLE extent(offset INTEGER PRIMARY KEY,length INTEGER NOT NULL,hash BLOB NOT NULL,blob_offset INTEGER NOT NULL);
            CREATE TABLE blob(hash BLOB PRIMARY KEY,place TEXT NOT NULL,encoding TEXT NOT NULL,size INTEGER NOT NULL,data BLOB);
            BEGIN;")?;
        let mut stmt = archive.prepare(&format!("SELECT e.offset,e.length,e.blob,e.blob_offset,b.rowid,b.place,b.encoding,b.size,length(b.data)
            FROM {schema}extent e JOIN {schema}blob b ON b.hash=e.blob
            WHERE e.input=?1 AND e.chain=?2 AND e.offset<?3 AND e.offset+e.length>?4 ORDER BY e.offset"))?;
        let mut rows = stmt.query(params![input, chain, length as i64, start as i64])?;
        let mut count = 0u64;
        let mut previous_end = None;
        let mut single = None;
        while let Some(row) = rows.next()? {
            let extent = capture_extent(archive, schema, &target, row)?;
            if previous_end.is_some_and(|end| extent.offset < end) {
                return Err(invalid("overlapping stored byte ranges"));
            }
            previous_end = Some(extent.offset + extent.length);
            count += 1;
            single = (count == 1
                && extent.offset == 0
                && extent.length == length
                && extent.size == length
                && extent.blob_offset == 0
                && !extent.inline)
                .then_some(extent.hash);
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

    pub fn select(
        &self,
        at: u64,
        store: &hashed_blobs::Store,
    ) -> Result<(Extent, Option<Box<dyn std::io::Read + Send>>)> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| invalid("stored byte ranges lock poisoned"))?;
        let extent = Self::extent(&conn, at)?;
        let inline = if extent.inline {
            Some(reader::inline(&conn, store, &extent)?)
        } else {
            None
        };
        // The owned inline reader needs no connection; file blobs open after unlock.
        Ok((extent, inline))
    }

    pub fn extent(conn: &Connection, at: u64) -> Result<Extent> {
        let row = conn.prepare_cached("SELECT e.offset,e.length,e.hash,e.blob_offset,b.size,b.place FROM extent e JOIN blob b ON b.hash=e.hash WHERE e.offset<=?1 ORDER BY e.offset DESC LIMIT 1")?
            .query_row([at as i64], |r| Ok((r.get::<_,i64>(0)?,r.get::<_,i64>(1)?,r.get::<_,Vec<u8>>(2)?,r.get::<_,i64>(3)?,r.get::<_,i64>(4)?,r.get::<_,String>(5)?))).optional()?;
        let (offset, length, hash, blob_offset, size, place) =
            row.ok_or_else(|| invalid("stored byte range missing"))?;
        let extent = Extent {
            offset: number(offset)?,
            length: number(length)?,
            hash: BlobHash(
                hash.try_into()
                    .map_err(|_| invalid("blob hash is not 32 bytes"))?,
            ),
            blob_offset: number(blob_offset)?,
            size: number(size)?,
            inline: place == "inline",
        };
        if extent
            .offset
            .checked_add(extent.length)
            .is_none_or(|end| end <= at)
        {
            return Err(invalid("stored byte range gap"));
        }
        Ok(extent)
    }
}

fn number(n: i64) -> Result<u64> {
    u64::try_from(n).map_err(|_| invalid("negative stored byte range metadata"))
}

fn invalid(message: &str) -> StoreError {
    StoreError::Invalid(message.into())
}

fn capture_extent(
    archive: &Connection,
    schema: &str,
    target: &Connection,
    row: &rusqlite::Row<'_>,
) -> Result<Extent> {
    let offset = number(row.get(0)?)?;
    let len = number(row.get(1)?)?;
    let hash: Vec<u8> = row.get(2)?;
    let hash = BlobHash(
        hash.try_into()
            .map_err(|_| invalid("blob hash is not 32 bytes"))?,
    );
    let blob_offset = number(row.get(3)?)?;
    let rowid: i64 = row.get(4)?;
    let place: String = row.get(5)?;
    let encoding: String = row.get(6)?;
    let size = number(row.get(7)?)?;
    if len == 0
        || offset.checked_add(len).is_none()
        || blob_offset.checked_add(len).is_none_or(|end| end > size)
    {
        return Err(invalid("invalid stored byte range bounds"));
    }
    if encoding != "plain" || !matches!(place.as_str(), "inline" | "file") {
        return Err(invalid(
            "stored input bytes need plain inline or file blobs",
        ));
    }
    let data_len: Option<i64> = row.get(8)?;
    if place == "inline" && data_len.is_none_or(|n| n < 0) {
        return Err(invalid("inline blob has no compressed bytes"));
    }
    let added = target.execute("INSERT OR IGNORE INTO blob(hash,place,encoding,size,data) VALUES (?1,?2,?3,?4,CASE WHEN ?2='inline' THEN zeroblob(?5) ELSE NULL END)", params![hash.0.as_slice(),place,encoding,size as i64,data_len])?;
    if added == 1 && place == "inline" {
        let database = if schema.is_empty() {
            "main"
        } else {
            schema.trim_end_matches('.')
        };
        let mut source = archive.blob_open(database, "blob", "data", rowid, true)?;
        let mut dest =
            target.blob_open(MAIN_DB, "blob", "data", target.last_insert_rowid(), false)?;
        let copied = std::io::copy(&mut source, &mut dest)?;
        if copied != data_len.unwrap() as u64 {
            return Err(invalid("inline blob changed during capture"));
        }
    }
    target.execute(
        "INSERT INTO extent VALUES (?1,?2,?3,?4)",
        params![
            offset as i64,
            len as i64,
            hash.0.as_slice(),
            blob_offset as i64
        ],
    )?;
    Ok(Extent {
        offset,
        length: len,
        hash,
        blob_offset,
        size,
        inline: place == "inline",
    })
}
