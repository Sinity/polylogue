use super::Inner;
use crate::Result;
use crate::primary::{PendingSource, SourceBase, SourceSelector, SourceSnapshot, SourceStatus};

impl Inner {
    pub(super) fn source_snapshot(&mut self, source: SourceSelector) -> Result<SourceSnapshot> {
        let token = self.location.scope_token()?;
        token.check()?;
        let conn = if self.location.is_primary() {
            &self.index
        } else {
            &self.archive
        };
        let pending = if let Some(sinex) = &self.sinex {
            sinex.pending_source(&source)?
        } else {
            PendingSource {
                status: SourceStatus::Ready,
                ..Default::default()
            }
        };
        let snapshot = crate::source::snapshot(&self.index, conn, &self.location, source, pending)?;
        token.check()?;
        Ok(snapshot)
    }
    /// Validate what was stated before provider I/O, without rebinding it to a
    /// later position or treating a missing original record as present.
    pub(super) fn sampled_base_matches(
        &self,
        base: &SourceBase,
        source: &SourceSelector,
        input: &str,
    ) -> Result<bool> {
        if &base.source != source {
            return Ok(false);
        }
        let through = match &base.boundary {
            Some(boundary) => match boundary.through(&self.index, &self.location) {
                Ok(through) => Some(through),
                Err(crate::StoreError::ScopeChanged) => return Ok(false),
                Err(error) => return Err(error),
            },
            None => None,
        };
        if base.status == SourceStatus::Ready {
            let Some(through) = through else {
                return Ok(false);
            };
            let (status, confirmed) =
                crate::source::Source::confirmed_source_at(&self.index, source, Some(through))?;
            if status != SourceStatus::Ready
                || confirmed.as_ref().map(|value| &value.observation.id) != base.confirmed.as_ref()
            {
                return Ok(false);
            }
        } else if let Some(observation) = &base.confirmed
            && !crate::source::Source::observation_matches(
                &self.index,
                source,
                observation,
                through,
            )?
        {
            return Ok(false);
        }
        if let Some(subject) = &base.resolved {
            let Some(through) = through else {
                return Ok(false);
            };
            if base.status != SourceStatus::Ready
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
