//! One normalized interval and one pending output range for a verified blob.
use super::Erasures;
use crate::excise::record_ranges::{Error, Ranges, edge};
use rusqlite::{OptionalExtension, params};
use std::fs::File;

type Range = (usize, usize);

pub(super) struct Cursor<'a> {
    owner: &'a Erasures,
    hash: &'a [u8],
    file: &'a File,
    current: Option<Range>,
    seen: (bool, bool),
    output: Option<Range>,
    pub(super) error: Option<Error>,
}

impl<'a> Cursor<'a> {
    pub(super) fn new(owner: &'a Erasures, hash: &'a [u8], file: &'a File) -> Result<Self, Error> {
        for table in ["checked", "normalized", "ranges"] {
            owner
                .conn
                .execute(&format!("DELETE FROM {table} WHERE hash=?1"), [hash])?;
        }
        let length = usize::try_from(file.metadata()?.len())
            .map_err(|_| Error::Records("source blob exceeds addressable offsets".into()))?;
        let mut pending: Option<Range> = None;
        {
            let mut statement = owner
                .conn
                .prepare("SELECT start,end FROM requested WHERE hash=?1 ORDER BY start,end")?;
            let mut rows = statement.query([hash])?;
            while let Some(row) = rows.next()? {
                let start = row.get::<_, i64>(0)? as usize;
                let end = row.get::<_, i64>(1)? as usize;
                if start > end || end > length {
                    return Err(Error::Records("source range is outside its blob".into()));
                }
                let start = edge(file, start, end, false)?
                    .ok_or_else(|| Error::Records("source range contains no JSON record".into()))?;
                let end = edge(file, start, end, true)?.expect("nonempty range") + 1;
                // Trimming cannot reverse start order: a later nonempty range
                // cannot contain a non-whitespace byte skipped by an earlier one.
                if let Some(last) = pending.as_mut().filter(|last| last.1 >= start) {
                    last.1 = last.1.max(end);
                } else {
                    if let Some(range) = pending.take() {
                        Self::normalized(owner, hash, range)?;
                    }
                    pending = Some((start, end));
                }
            }
        }
        if let Some(range) = pending {
            Self::normalized(owner, hash, range)?;
        }
        let mut cursor = Self {
            owner,
            hash,
            file,
            current: None,
            seen: (false, false),
            output: None,
            error: None,
        };
        cursor.next(-1)?;
        Ok(cursor)
    }

    fn normalized(owner: &Erasures, hash: &[u8], (start, end): Range) -> Result<(), Error> {
        owner
            .conn
            .prepare_cached("INSERT INTO normalized VALUES (?1,?2,?3)")?
            .execute(params![hash, start as i64, end as i64])?;
        Ok(())
    }

    fn next(&mut self, after: i64) -> Result<(), Error> {
        crate::fault::check("excision_ranges.read", "").map_err(Error::Store)?;
        self.current = self.owner.conn.prepare_cached("SELECT start,end FROM normalized WHERE hash=?1 AND start>?2 ORDER BY start LIMIT 1")?
            .query_row(params![self.hash, after], |row| Ok((row.get::<_, i64>(0)? as usize, row.get::<_, i64>(1)? as usize))).optional()?;
        self.seen = (false, false);
        Ok(())
    }

    fn advance(&mut self, at: usize) -> Result<(), Error> {
        while let Some(range) = self.current.filter(|range| range.1 <= at) {
            self.complete_interval()?;
            self.next(range.0 as i64)?;
        }
        Ok(())
    }

    fn complete_interval(&self) -> Result<(), Error> {
        if self.seen != (true, true) {
            return Err(Error::Records(
                "source citation does not follow complete JSON value boundaries".into(),
            ));
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), Error> {
        if let Some((start, end)) = self.output.take() {
            crate::fault::check("excision_ranges.write", "").map_err(Error::Store)?;
            self.owner
                .conn
                .prepare_cached("INSERT OR IGNORE INTO ranges VALUES (?1,?2,?3)")?
                .execute(params![self.hash, start as i64, end as i64])?;
        }
        Ok(())
    }

    fn emit(&mut self, range: Range) -> Result<(), Error> {
        if let Some(last) = self.output.as_mut()
            && last.1 <= range.0
            && edge(self.file, last.1, range.0, false)?.is_none()
        {
            last.1 = range.1;
        } else {
            self.flush()?;
            self.output = Some(range);
        }
        Ok(())
    }

    pub(super) fn finish(&mut self) -> Result<(), Error> {
        while let Some(range) = self.current {
            self.complete_interval()?;
            self.next(range.0 as i64)?;
        }
        self.flush()
    }

    fn retain_error<T>(&mut self, result: Result<T, Error>) -> Result<T, String> {
        result.map_err(|error| {
            self.error = Some(error);
            "record range planning failed".into()
        })
    }
}

impl Ranges for Cursor<'_> {
    fn covered(&mut self, range: Range) -> Result<bool, String> {
        let result = self.advance(range.0);
        self.retain_error(result)?;
        Ok(self
            .current
            .is_some_and(|r| r.0 <= range.0 && r.1 >= range.1))
    }
    fn touched(&mut self, range: Range) -> Result<bool, String> {
        let result = self.advance(range.0);
        self.retain_error(result)?;
        Ok(self.current.is_some_and(|r| r.0 < range.1))
    }
    fn mark(&mut self, range: Range) {
        let current = self.current.expect("covered interval");
        self.seen.0 |= current.0 == range.0;
        self.seen.1 |= current.1 == range.1;
    }
    fn push(&mut self, range: Range) -> Result<(), String> {
        let result = self.emit(range);
        self.retain_error(result)
    }
}
