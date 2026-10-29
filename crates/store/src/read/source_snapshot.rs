use super::Reader;
use crate::Result;
use crate::primary::{ConfirmedSource, SourceSelector, SourceStatus, SourceSubject};
use rusqlite::OptionalExtension;

impl Reader<'_> {
    /// Existing source authority in this read snapshot, before provider I/O.
    /// Missing full primary coverage remains unknown.
    pub fn source_boundary(&self) -> Result<Option<crate::primary::SourceBoundary>> {
        let conn = self.conn()?;
        crate::source::SourceBoundary::read(conn, conn, &self.pool.archive)
    }

    /// The exact retained input head at the pre-acquisition cut. Missing
    /// observation evidence cannot identify a native alias predecessor.
    pub fn source_subject_at(
        &self,
        boundary: &crate::primary::SourceBoundary,
        input: &str,
    ) -> Result<Option<SourceSubject>> {
        let conn = self.conn()?;
        let through = boundary.through(conn, &self.pool.archive)?;
        let through = i64::try_from(through).map_err(|_| {
            crate::StoreError::Invalid("source admission fence exceeds archive address space".into())
        })?;
        Ok(conn.query_row(
            "SELECT id FROM source_observation WHERE input=?1 AND commit_seq<=?2 ORDER BY rowid DESC LIMIT 1",
            rusqlite::params![input, through],
            |row| Ok(SourceSubject { input: input.into(), observation: row.get(0)? }),
        ).optional()?)
    }

    /// Exact retained observation identity. In primary mode, absent cache rows
    /// prove absence only after full owner scope coverage is certified.
    pub fn source_observation(
        &self,
        source: &SourceSelector,
    ) -> Result<(SourceStatus, Option<ConfirmedSource>)> {
        let conn = self.conn()?;
        let (mut status, confirmed) = crate::source::Source::confirmed_source(conn, source)?;
        if self.source_boundary()?.is_none() && status == SourceStatus::Ready {
            status = SourceStatus::MissingEvidence;
        }
        Ok((status, confirmed))
    }
}
