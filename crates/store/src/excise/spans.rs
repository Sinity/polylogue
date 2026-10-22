use super::{Result, Span};
use rusqlite::{Connection, params};
use std::collections::BTreeSet;

struct State {
    conn: Connection,
    _file: tempfile::NamedTempFile,
}

/// Logical citations survive selected index-row deletion until rewrite completes.
#[derive(Default)]
pub(super) struct Spans(Option<State>);

impl Spans {
    pub fn insert(&mut self, input: &str, span: Span) -> Result<()> {
        if self.0.is_none() {
            let file = tempfile::NamedTempFile::new()?;
            let conn = Connection::open(file.path())?;
            conn.execute_batch("PRAGMA cache_size=-1024; PRAGMA mmap_size=0; PRAGMA temp_store=FILE;
                PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
                CREATE TABLE span(input TEXT NOT NULL,chain INTEGER NOT NULL,start INTEGER NOT NULL,end INTEGER NOT NULL);
                CREATE INDEX span_order ON span(input,chain,start,end);
                BEGIN;")?;
            self.0 = Some(State { conn, _file: file });
        }
        crate::fault::check("excision_spans.insert", input)?;
        self.0
            .as_ref()
            .unwrap()
            .conn
            .prepare_cached("INSERT INTO span VALUES (?1,?2,?3,?4)")?
            .execute(params![input, span.chain, span.start, span.end])?;
        Ok(())
    }

    pub fn remove_whole(&self, whole: &BTreeSet<String>) -> Result<()> {
        if let Some(state) = &self.0 {
            let mut delete = state
                .conn
                .prepare_cached("DELETE FROM span WHERE input=?1")?;
            for input in whole {
                delete.execute([input])?;
            }
        }
        Ok(())
    }

    pub fn visit_inputs(&self, mut f: impl FnMut(&str, u64) -> Result<()>) -> Result<()> {
        if let Some(state) = &self.0 {
            let mut stmt = state.conn.prepare_cached(
                "SELECT input,count(*) FROM span GROUP BY input ORDER BY input COLLATE BINARY",
            )?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let input: String = row.get(0)?;
                f(&input, row.get::<_, i64>(1)? as u64)?;
            }
        }
        Ok(())
    }

    pub fn visit(&self, input: &str, mut f: impl FnMut(Span) -> Result<()>) -> Result<()> {
        if let Some(state) = &self.0 {
            crate::fault::check("excision_spans.read", input)?;
            let mut stmt = state.conn.prepare_cached(
                "SELECT chain,start,end FROM span WHERE input=?1 ORDER BY chain,start,end,rowid",
            )?;
            let mut rows = stmt.query([input])?;
            while let Some(row) = rows.next()? {
                f(Span {
                    chain: row.get(0)?,
                    start: row.get(1)?,
                    end: row.get(2)?,
                })?;
            }
        }
        Ok(())
    }

    pub fn visit_chains(&self, input: &str, mut f: impl FnMut(i64) -> Result<()>) -> Result<()> {
        if let Some(state) = &self.0 {
            let mut stmt = state
                .conn
                .prepare_cached("SELECT DISTINCT chain FROM span WHERE input=?1 ORDER BY chain")?;
            let mut rows = stmt.query([input])?;
            while let Some(row) = rows.next()? {
                f(row.get(0)?)?;
            }
        }
        Ok(())
    }

    pub fn has_chain(&self, input: &str, chain: i64) -> Result<bool> {
        match &self.0 {
            Some(state) => Ok(state
                .conn
                .prepare_cached("SELECT EXISTS(SELECT 1 FROM span WHERE input=?1 AND chain=?2)")?
                .query_row(params![input, chain], |r| r.get(0))?),
            None => Ok(false),
        }
    }

    pub fn invalid_bounds(&self, input: &str, chain: i64, length: i64) -> Result<bool> {
        match &self.0 {
            Some(state) => Ok(state.conn.prepare_cached("SELECT EXISTS(SELECT 1 FROM span WHERE input=?1 AND chain=?2 AND (start<0 OR end>?3 OR end<=start))")?.query_row(params![input,chain,length], |r|r.get(0))?),
            None => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests;
