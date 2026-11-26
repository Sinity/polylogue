//! Reads of the stored source observation of a location.
use super::Reader;
use crate::Result;
use crate::capture_read::{ConfirmedSource, SourceSelector, SourceSnapshot, SourceStatus};

impl Reader<'_> {
    /// The source snapshot a capture read samples before provider I/O, from
    /// this read's committed state.
    pub fn source_snapshot(&self, source: SourceSelector) -> Result<SourceSnapshot> {
        crate::source::snapshot(self.conn()?, &source)
    }

    /// Stored observation identity.
    pub fn source_observation(
        &self,
        source: &SourceSelector,
    ) -> Result<(SourceStatus, Option<ConfirmedSource>)> {
        let snapshot = crate::source::snapshot(self.conn()?, source)?;
        Ok((snapshot.status, snapshot.confirmed))
    }
}
