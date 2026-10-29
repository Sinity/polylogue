use super::Inner;
use crate::Result;
use crate::primary::{PendingSource, SourceBasis, SourceSelector, SourceSnapshot, SourceStatus};

impl Inner {
    pub(super) fn source_snapshot(
        &mut self,
        source: SourceSelector,
        reading_version: &str,
    ) -> Result<SourceSnapshot> {
        self.location.scope_token()?.check()?;
        let conn = if self.location.is_primary() {
            &self.index
        } else {
            &self.archive
        };
        let (mut status, confirmed) = crate::source::Source::confirmed_source(conn, &source)?;
        if self.location.is_primary() && crate::primary::projection::checkpoint(conn)?.is_none() {
            status = SourceStatus::MissingEvidence;
        }
        let pending = if let Some(sinex) = &self.sinex {
            sinex.pending_source(&source, reading_version)?
        } else {
            PendingSource {
                status: SourceStatus::Ready,
                ..Default::default()
            }
        };
        status = match (status, pending.status) {
            (SourceStatus::Ambiguous, _) | (_, SourceStatus::Ambiguous) => SourceStatus::Ambiguous,
            (SourceStatus::MissingEvidence, _) | (_, SourceStatus::MissingEvidence) => {
                SourceStatus::MissingEvidence
            }
            _ => SourceStatus::Ready,
        };
        let basis = SourceBasis {
            source,
            status,
            confirmed: confirmed.as_ref().map(|value| value.observation.id.clone()),
            pending: pending.head.as_ref().map(|value| value.identity.clone()),
        };
        Ok(SourceSnapshot {
            basis,
            confirmed,
            pending,
        })
    }
}
