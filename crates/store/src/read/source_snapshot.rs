use super::Reader;
use crate::Result;
use crate::primary::{ConfirmedSource, SourceSelector, SourceStatus, SourceSubject};

impl Reader<'_> {
    /// A hint for whether an explicit import at this receiving location may
    /// already be stored. The writer checks the verified bytes and membership.
    pub fn has_import_at(&self, machine: &str, path: &[u8]) -> Result<bool> {
        use crate::ConnExt as _;
        self.conn()?.exists(
            "SELECT EXISTS(SELECT 1 FROM source_observation WHERE machine=?1 AND path_bytes=?2 AND class='imported')",
            rusqlite::params![machine, path],
        )
    }
    /// The source boundary this read snapshot already has, before provider I/O.
    /// Missing full primary coverage remains unknown.
    pub fn source_boundary(&self) -> Result<Option<crate::primary::SourceBoundary>> {
        let conn = self.conn()?;
        crate::source::SourceBoundary::read(conn, conn, &self.pool.archive)
    }

    /// The stored input's latest revision at the position fixed before the capture
    /// read. Without an observation it cannot identify the previous revision of
    /// an alias path.
    pub fn source_subject_at(
        &self,
        boundary: &crate::primary::SourceBoundary,
        input: &str,
    ) -> Result<Option<SourceSubject>> {
        let conn = self.conn()?;
        let through = boundary.through(conn, &self.pool.archive)?;
        crate::source::Source::subject_at(conn, input, through)
    }

    /// Stored observation identity. In primary mode, absent cache rows show
    /// absence only once the cache covers the whole Sinex scope.
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
