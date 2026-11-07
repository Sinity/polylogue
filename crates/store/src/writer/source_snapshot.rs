use super::Inner;
use crate::Result;
use crate::primary::{PendingSource, SourceBasis, SourceSelector, SourceSnapshot, SourceStatus};

impl Inner {
    pub(super) fn source_snapshot(
        &mut self,
        source: SourceSelector,
        reading_version: &str,
    ) -> Result<SourceSnapshot> {
        let token = self.location.scope_token()?;
        token.check()?;
        let conn = if self.location.is_primary() {
            &self.index
        } else {
            &self.archive
        };
        let (mut status, confirmed) = crate::source::Source::confirmed_source(conn, &source)?;
        let boundary = crate::source::SourceBoundary::read(&self.index, conn, &self.location)?;
        if boundary.is_none() && status == SourceStatus::Ready {
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
        token.check()?;
        let basis = SourceBasis {
            boundary,
            resolved: None,
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
    /// Validate the claims frozen before provider I/O, without rebinding them to
    /// a later head or upgrading missing original evidence.
    pub(super) fn sampled_basis_matches(
        &self,
        basis: &SourceBasis,
        source: &SourceSelector,
        input: &str,
    ) -> Result<bool> {
        if &basis.source != source {
            return Ok(false);
        }
        let through = match &basis.boundary {
            Some(boundary) => match boundary.through(&self.index, &self.location) {
                Ok(through) => Some(through),
                Err(crate::StoreError::ScopeChanged) => return Ok(false),
                Err(error) => return Err(error),
            },
            None => None,
        };
        if basis.status == SourceStatus::Ready {
            let Some(through) = through else {
                return Ok(false);
            };
            let (status, confirmed) =
                crate::source::Source::confirmed_source_at(&self.index, source, Some(through))?;
            if status != SourceStatus::Ready
                || confirmed.as_ref().map(|value| &value.observation.id) != basis.confirmed.as_ref()
            {
                return Ok(false);
            }
        } else if let Some(observation) = &basis.confirmed {
            if !crate::source::Source::observation_matches(
                &self.index,
                source,
                observation,
                through,
            )? {
                return Ok(false);
            }
        }
        if let Some(subject) = &basis.resolved {
            let Some(through) = through else {
                return Ok(false);
            };
            if basis.status != SourceStatus::Ready
                || subject.input != input
                || crate::source::Source::subject_at(&self.index, &subject.input, through)?.as_ref()
                    != Some(subject)
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
