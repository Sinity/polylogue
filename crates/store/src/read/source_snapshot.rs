use super::Reader;
use crate::Result;
use crate::primary::{ConfirmedSource, SourceSelector, SourceStatus, SourceSubject};

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
        crate::source::Source::subject_at(conn, input, through)
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
