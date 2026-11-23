//! Chain metadata, extents and planned copies stay on disk while one input is planned.
use super::{ChainLayout, StoredParseState};
use crate::ConnExt as _;
use crate::{Result, source::Source};
use rusqlite::{Connection, OptionalExtension, params};

fn scratch() -> Result<Connection> {
    let conn = Connection::open("")?;
    conn.execute_batch("PRAGMA cache_size=-256; PRAGMA mmap_size=0; PRAGMA temp_store=FILE; PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_spill=ON;")?;
    conn.set_prepared_statement_cache_capacity(8);
    Ok(conn)
}

/// One input's chain layouts, with fixed page-cache space and streamed extents.
pub struct ChainLayouts {
    conn: Connection,
    count: usize,
}

impl ChainLayouts {
    pub(crate) fn load(source: &Connection, input: &str) -> Result<Option<Self>> {
        let schema = Source::of(source)?;
        let multiple:bool = source.query_row(&schema.sql(
            "SELECT count(*)>1 FROM (SELECT DISTINCT chain FROM @source.revision WHERE input=?1 LIMIT 2)",
        ),[input],|row|row.get(0))?;
        if !multiple {
            return Ok(None);
        }
        let conn = scratch()?;
        conn.execute_batch("CREATE TABLE layout(ordinal INTEGER PRIMARY KEY,chain INTEGER UNIQUE,length INTEGER,version TEXT,status TEXT,parsed INTEGER,at_length INTEGER);
            CREATE TABLE extent(chain INTEGER,offset INTEGER,length INTEGER,blob BLOB,blob_offset INTEGER,PRIMARY KEY(chain,offset)) WITHOUT ROWID; BEGIN;")?;
        let mut statement = source.prepare(&schema.sql(
            "SELECT c.chain,c.length,p.reading_version,p.status,p.parsed_upto,p.at_length
             FROM (SELECT chain,max(length) AS length FROM @source.revision WHERE input=?1 GROUP BY chain) c
             LEFT JOIN input_ref i ON i.id=?1 LEFT JOIN parse_state p ON p.input_pk=i.pk AND p.chain=c.chain ORDER BY c.chain",
        ))?;
        let mut rows = statement.query([input])?;
        let mut count = 0;
        while let Some(row) = rows.next()? {
            conn.prepare_cached("INSERT INTO layout VALUES(?1,?2,?3,?4,?5,?6,?7)")?
                .execute(params![
                    count as u64,
                    row.get::<_, i64>(0)?,
                    row.get::<_, u64>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<u64>>(4)?,
                    row.get::<_, Option<u64>>(5)?,
                ])?;
            count += 1;
        }
        drop(rows);
        drop(statement);
        let unread = conn.exists(
            "SELECT EXISTS(SELECT 1 FROM layout WHERE version IS NULL)",
            [],
        )?;
        if unread {
            let mut statement = source.prepare(&schema.sql(
                "SELECT chain,offset,length,blob,blob_offset FROM @source.extent WHERE input=?1 ORDER BY chain,offset",
            ))?;
            let mut rows = statement.query([input])?;
            while let Some(row) = rows.next()? {
                conn.prepare_cached("INSERT INTO extent VALUES(?1,?2,?3,?4,?5)")?
                    .execute(params![
                        row.get::<_, i64>(0)?,
                        row.get::<_, u64>(1)?,
                        row.get::<_, u64>(2)?,
                        row.get::<_, Vec<u8>>(3)?,
                        row.get::<_, u64>(4)?,
                    ])?;
            }
        }
        conn.execute_batch("COMMIT")?;
        Ok(Some(Self { conn, count }))
    }
    pub fn len(&self) -> usize {
        self.count
    }
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    pub fn find(&self, chain: i64) -> Result<Option<usize>> {
        self.conn
            .query_row(
                "SELECT ordinal FROM layout WHERE chain=?1",
                [chain],
                |row| row.get::<_, u64>(0),
            )
            .optional()
            .map(|n| n.map(|n| n as usize))
            .map_err(Into::into)
    }
    pub fn get(&self, n: usize) -> Result<ChainLayout> {
        Ok(self.conn.query_row(
            "SELECT chain,length,version,status,parsed,at_length FROM layout WHERE ordinal=?1",
            [n as u64],
            |row| {
                let version: Option<String> = row.get(2)?;
                let reading = version
                    .map(|reading_version| -> rusqlite::Result<_> {
                        Ok(StoredParseState {
                            reading_version,
                            status: row.get(3)?,
                            parsed_upto: row.get(4)?,
                            at_length: row.get(5)?,
                        })
                    })
                    .transpose()?;
                Ok(ChainLayout {
                    chain: row.get(0)?,
                    length: row.get(1)?,
                    reading,
                })
            },
        )?)
    }
    pub fn unread_before(&self, n: usize) -> Result<bool> {
        self.conn.exists(
            "SELECT EXISTS(SELECT 1 FROM layout WHERE ordinal<?1 AND version IS NULL)",
            [n as u64],
        )
    }
    /// Extents are compared in offset order without loading either chain's list.
    pub fn shared_extents(&self, a: usize, b: usize) -> Result<u64> {
        let (a, b) = (self.get(a)?, self.get(b)?);
        let mut left = self.conn.prepare_cached(
            "SELECT offset,length,blob,blob_offset FROM extent WHERE chain=?1 ORDER BY offset",
        )?;
        let mut right = self.conn.prepare_cached(
            "SELECT offset,length,blob,blob_offset FROM extent WHERE chain=?1 ORDER BY offset",
        )?;
        let mut left = left.query([a.chain])?;
        let mut right = right.query([b.chain])?;
        let mut x = left.next()?;
        let mut y = right.next()?;
        let mut at = 0;
        while let (Some(l), Some(r)) = (x, y) {
            crate::cancellation::check_current()?;
            let (lo, ll, lb, lp): (u64, u64, Vec<u8>, u64) =
                (l.get(0)?, l.get(1)?, l.get(2)?, l.get(3)?);
            let (ro, rl, rb, rp): (u64, u64, Vec<u8>, u64) =
                (r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?);
            if lo > at
                || at >= lo + ll
                || ro > at
                || at >= ro + rl
                || lb != rb
                || lp + (at - lo) != rp + (at - ro)
            {
                break;
            }
            at = (lo + ll).min(ro + rl);
            if at == lo + ll {
                x = left.next()?;
            }
            if at == ro + rl {
                y = right.next()?;
            }
        }
        Ok(at.min(a.length).min(b.length))
    }
}

/// One older chain's copy at a stop in a newer chain's parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixCopy {
    pub chain: i64,
    pub stop: u64,
    pub reaches_end: bool,
}

/// An ordered copy plan keeps one row in memory, regardless of history size.
pub struct PrefixCopies {
    conn: Connection,
    first: Option<PrefixCopy>,
}
impl PrefixCopies {
    pub fn new() -> Result<Self> {
        let conn = scratch()?;
        conn.execute_batch("CREATE TABLE copy(stop INTEGER,chain INTEGER,reaches_end INTEGER,PRIMARY KEY(stop,chain)) WITHOUT ROWID; BEGIN;")?;
        Ok(Self { conn, first: None })
    }
    pub fn push(&self, copy: PrefixCopy) -> Result<()> {
        self.conn
            .prepare_cached("INSERT INTO copy VALUES(?1,?2,?3)")?
            .execute(params![copy.stop, copy.chain, copy.reaches_end])?;
        Ok(())
    }
    pub fn finish(&mut self) -> Result<()> {
        self.conn.execute_batch("COMMIT")?;
        self.first = self.next(None)?;
        Ok(())
    }
    fn next(&self, after: Option<&PrefixCopy>) -> Result<Option<PrefixCopy>> {
        let read = |row: &rusqlite::Row<'_>| {
            Ok(PrefixCopy {
                chain: row.get(0)?,
                stop: row.get(1)?,
                reaches_end: row.get(2)?,
            })
        };
        Ok(match after {
            Some(after) => self.conn.prepare_cached(
                "SELECT chain,stop,reaches_end FROM copy WHERE (stop,chain)>(?1,?2) ORDER BY stop,chain LIMIT 1",
            )?.query_row(params![after.stop,after.chain],read).optional()?,
            None => self.conn.prepare_cached(
                "SELECT chain,stop,reaches_end FROM copy ORDER BY stop,chain LIMIT 1",
            )?.query_row([],read).optional()?,
        })
    }
    pub fn first(&self) -> Option<&PrefixCopy> {
        self.first.as_ref()
    }
    pub fn pop(&mut self) -> Result<Option<PrefixCopy>> {
        let first = self.first.take();
        if first.is_some() {
            self.first = self.next(first.as_ref())?;
        }
        Ok(first)
    }
}

/// Completed byte comparisons stay on disk so planning never repeats a long prefix.
#[derive(Default)]
pub struct SharedPrefixes {
    conn: Option<Connection>,
}
impl SharedPrefixes {
    pub fn clear(&mut self) {
        self.conn = None;
    }
    pub fn get(&self, input: &str, older: (i64, u64), newer: (i64, u64)) -> Result<Option<u64>> {
        let Some(conn) = &self.conn else {
            return Ok(None);
        };
        Ok(conn.prepare_cached("SELECT bytes FROM shared WHERE input=?1 AND older=?2 AND older_length=?3 AND newer=?4 AND newer_length=?5")?
            .query_row(params![input,older.0,older.1,newer.0,newer.1], |row| row.get(0)).optional()?)
    }
    pub fn insert(
        &mut self,
        input: &str,
        older: (i64, u64),
        newer: (i64, u64),
        bytes: u64,
    ) -> Result<()> {
        if self.conn.is_none() {
            let conn = scratch()?;
            conn.execute_batch("CREATE TABLE shared(input TEXT,older INTEGER,older_length INTEGER,newer INTEGER,newer_length INTEGER,bytes INTEGER,PRIMARY KEY(input,older,older_length,newer,newer_length)) WITHOUT ROWID; BEGIN;")?;
            self.conn = Some(conn);
        }
        self.conn
            .as_ref()
            .expect("comparison database")
            .prepare_cached("INSERT OR REPLACE INTO shared VALUES(?1,?2,?3,?4,?5,?6)")?
            .execute(params![input, older.0, older.1, newer.0, newer.1, bytes])?;
        Ok(())
    }
}
