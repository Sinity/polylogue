use super::*;
use crate::read::limit::ReadLimit;
use crate::row::{Rows as _, Statement, row};
use polylogue_model::LockExt;
use rusqlite::{OpenFlags, params};
use std::ops::ControlFlow;
use std::sync::Mutex;

/// The page cache of one chain's handle. A chain looks its extents up by
/// primary key, a few pages per read; [`CHAIN_CACHE_KIB`] is counted in the
/// loop's reader memory once per parse worker and once per client slot.
pub const CHAIN_CACHE_KIB: u64 = 256;

/// A chain's extent list, read where it is stored through the chain's own
/// read-only handle. The handle holds a read transaction only while the
/// chain's bytes are being read: it opens at the first extent lookup and
/// closes when a read reaches its end or the reader is about to wait
/// ([`ExtentList::release`]), so a loaded chain that is not being read never
/// pins the write-ahead log. Every transaction first checks that the extents
/// covering the chain's range are the ones checked at load, so a read never
/// joins bytes from two versions of the list.
#[derive(Debug)]
pub(super) struct ExtentList {
    conn: Mutex<Option<Connection>>,
    source: ChainSource,
    input: String,
    chain: i64,
    start: u64,
    length: u64,
    loaded: blake3::Hash,
}

pub(super) struct Extent {
    pub offset: u64,
    pub length: u64,
    pub hash: BlobHash,
    pub blob_offset: u64,
    pub size: u64,
}

macro_rules! select_extents {
    (at $prefix:literal) => {
        select_extents!($prefix, "e.offset<=?3 ORDER BY e.offset DESC LIMIT 1")
    };
    (range $prefix:literal) => {
        select_extents!(
            $prefix,
            "e.offset<?3 AND e.offset+e.length>?4 ORDER BY e.offset"
        )
    };
    ($prefix:literal, $condition:literal) => {
        concat!(
            "SELECT e.offset,e.length,e.blob,e.blob_offset,b.size FROM ",
            $prefix,
            "extent e JOIN ",
            $prefix,
            "blob b ON b.hash=e.blob
             WHERE e.input=?1 AND e.chain=?2 AND ",
            $condition,
        )
    };
}

row! {
    struct ExtentRow {
        offset: u64,
        length: u64,
        blob: [u8; 32],
        blob_offset: u64,
        size: u64,
    }
    statements {
        const AT = select_extents!(at "");
        const SOURCE_AT = select_extents!(at "source_");
        const RANGE = select_extents!(range "");
        const SOURCE_RANGE = select_extents!(range "source_");
    }
}

/// What one pass over the extents covering a range found.
struct Checked {
    rows: blake3::Hash,
    single: Option<BlobHash>,
}

impl ExtentList {
    /// Checks the extents covering `start..length` once: bounds within their
    /// blobs and no overlaps, so a read never selects a different source by
    /// its position. The handle is closed again before this returns.
    pub fn capture(
        source: &ChainSource,
        input: &str,
        chain: i64,
        start: u64,
        length: u64,
    ) -> Result<(Self, Option<BlobHash>)> {
        let conn = open(source)?;
        conn.execute_batch("BEGIN")?;
        let checked = check(&conn, source.table_prefix, input, chain, start, length)?;
        conn.execute_batch("COMMIT")?;
        drop(conn);
        Ok((
            Self {
                conn: Mutex::new(None),
                source: source.clone(),
                input: input.to_owned(),
                chain,
                start,
                length,
                loaded: checked.rows,
            },
            checked.single,
        ))
    }

    #[cfg(test)]
    pub(super) fn connection(&self) -> std::sync::MutexGuard<'_, Option<Connection>> {
        self.conn.lock().unwrap()
    }

    /// Ends the read transaction, if one is open, and closes the handle with
    /// its page cache and descriptors. The next lookup opens a new one.
    pub fn release(&self) {
        let mut conn = self.conn.locked();
        if let Some(conn) = conn.take() {
            let _ = conn.execute_batch("COMMIT");
        }
    }

    /// The extent holding byte `at`.
    pub fn select(&self, at: u64) -> Result<Extent> {
        let mut conn = self
            .conn
            .lock()
            .map_err(|_| crate::StoreError::internal("stored extents lock poisoned"))?;
        if conn.is_none() {
            *conn = Some(self.reopen()?);
        }
        let conn = conn.as_ref().expect("opened above");
        let table_prefix = self.source.table_prefix;
        let prefixed;
        let statement = match table_prefix {
            "" => &ExtentRow::AT,
            "source_" => &ExtentRow::SOURCE_AT,
            prefix => {
                prefixed = Statement::built(format!(select_extents!(at "{0}"), prefix));
                &prefixed
            }
        };
        let row = conn
            .maybe(statement, params![self.input, self.chain, at as i64])?
            .ok_or_else(|| crate::StoreError::stored_data("stored extent missing"))?;
        let extent = Extent {
            offset: row.offset,
            length: row.length,
            hash: BlobHash(row.blob),
            blob_offset: row.blob_offset,
            size: row.size,
        };
        if extent
            .offset
            .checked_add(extent.length)
            .is_none_or(|end| end <= at)
        {
            return Err(crate::StoreError::stored_data("stored extent gap"));
        }
        Ok(extent)
    }

    /// A handle inside a new read transaction whose extents are the ones
    /// checked at load.
    fn reopen(&self) -> Result<Connection> {
        let conn = open(&self.source)?;
        conn.execute_batch("BEGIN")?;
        let checked = check(
            &conn,
            self.source.table_prefix,
            &self.input,
            self.chain,
            self.start,
            self.length,
        )?;
        if checked.rows != self.loaded {
            return Err(crate::StoreError::internal(
                "stored extents changed since the chain was loaded",
            ));
        }
        Ok(conn)
    }
}

/// A read-only handle on the extents' database, under the reading request's
/// limit when there is one.
fn open(source: &ChainSource) -> Result<Connection> {
    let conn = Connection::open_with_flags(
        &source.path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    conn.busy_timeout(std::time::Duration::from_secs(30))?;
    conn.execute_batch(&format!(
        "PRAGMA cache_size=-{CHAIN_CACHE_KIB}; PRAGMA mmap_size=0"
    ))?;
    ReadLimit::observe_current(&conn)?;
    Ok(conn)
}

/// Checks the extents covering `start..length` and hashes their rows.
fn check(
    conn: &Connection,
    table_prefix: &str,
    input: &str,
    chain: i64,
    start: u64,
    length: u64,
) -> Result<Checked> {
    let prefixed;
    let statement = match table_prefix {
        "" => &ExtentRow::RANGE,
        "source_" => &ExtentRow::SOURCE_RANGE,
        prefix => {
            prefixed = Statement::built(format!(select_extents!(range "{0}"), prefix));
            &prefixed
        }
    };
    let mut digest = blake3::Hasher::new();
    let mut single = None;
    let mut count = 0u64;
    let mut previous_end = None;
    let _ = conn.each(
        statement,
        params![input, chain, length as i64, start as i64],
        |ExtentRow {
             offset,
             length: len,
             blob,
             blob_offset,
             size,
         }| {
            if len == 0
                || offset.checked_add(len).is_none()
                || blob_offset.checked_add(len).is_none_or(|end| end > size)
                || previous_end.is_some_and(|end| offset < end)
            {
                return Err(crate::StoreError::stored_data(
                    "invalid stored extent bounds",
                ));
            }
            previous_end = Some(offset + len);
            for value in [offset, len, blob_offset, size] {
                digest.update(&value.to_le_bytes());
            }
            digest.update(&blob);
            count += 1;
            single =
                (count == 1 && offset == 0 && len == length && size == length && blob_offset == 0)
                    .then_some(BlobHash(blob));
            Ok(ControlFlow::Continue(()))
        },
    )?;
    Ok(Checked {
        rows: digest.finalize(),
        single,
    })
}

#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<ExtentRow>();
}
