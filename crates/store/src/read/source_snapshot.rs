use super::Reader;
use crate::Result;
use crate::primary::{ConfirmedSource, SourceSelector, SourceStatus};

impl Reader<'_> {
    /// Exact retained observation identity. In primary mode, absent cache rows
    /// prove absence only after a confirmed checkpoint covers the source scope.
    pub fn source_observation(
        &self,
        source: &SourceSelector,
    ) -> Result<(SourceStatus, Option<ConfirmedSource>)> {
        let conn = self.conn()?;
        let (mut status, confirmed) = crate::source::Source::confirmed_source(conn, source)?;
        if self.pool.archive.is_primary() && crate::primary::projection::checkpoint(conn)?.is_none()
        {
            status = SourceStatus::MissingEvidence;
        }
        Ok((status, confirmed))
    }
}
