//! Frozen session selection belongs to the index transaction, without foreign keys.
use crate::Result;
use rusqlite::{Connection, params};

pub(super) struct SelectedSessions<'a>(&'a Connection);

impl<'a> SelectedSessions<'a> {
    pub(super) fn new(index: &'a Connection) -> Result<Self> {
        index.execute_batch(
            "CREATE TEMP TABLE excision_sessions(
                ordinal INTEGER PRIMARY KEY,
                session_pk INTEGER NOT NULL,
                session_id TEXT NOT NULL
             );
             CREATE INDEX temp.excision_sessions_pk ON excision_sessions(session_pk);",
        )?;
        Ok(Self(index))
    }

    pub(super) fn insert(&self, pk: i64, id: &str) -> Result<()> {
        self.0
            .prepare_cached(
                "INSERT INTO temp.excision_sessions(session_pk,session_id) VALUES (?1,?2)",
            )?
            .execute(params![pk, id])?;
        Ok(())
    }

    pub(super) fn is_empty(&self) -> Result<bool> {
        Ok(self.0.query_row(
            "SELECT NOT EXISTS(SELECT 1 FROM temp.excision_sessions)",
            [],
            |r| r.get(0),
        )?)
    }

    pub(super) fn ids(&self) -> Result<Vec<String>> {
        let mut ids = Vec::new();
        self.visit(|_, id| {
            ids.push(id.to_owned());
            Ok(())
        })?;
        Ok(ids)
    }

    pub(super) fn visit(&self, mut visit: impl FnMut(i64, &str) -> Result<()>) -> Result<()> {
        let mut statement = self
            .0
            .prepare("SELECT session_pk,session_id FROM temp.excision_sessions ORDER BY ordinal")?;
        let mut rows = statement.query([])?;
        while let Some(row) = rows.next()? {
            let id = row.get_ref(1)?.as_str().map_err(|error| {
                rusqlite::Error::FromSqlConversionFailure(
                    1,
                    rusqlite::types::Type::Text,
                    Box::new(error),
                )
            })?;
            visit(row.get(0)?, id)?;
        }
        Ok(())
    }

    pub(super) fn finish(self) -> Result<()> {
        self.0.execute_batch("DROP TABLE temp.excision_sessions")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
