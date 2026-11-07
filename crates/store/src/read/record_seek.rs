//! Record-family cardinalities in the same pinned physical session scope as
//! the canonical event and link ordinal visitors. Payloads are never selected.
use crate::{Reader, Result, StoreError};
use polylogue_model::SessionId;
use rusqlite::OptionalExtension;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalRecordCounts {
    pub events: u64,
    pub links: u64,
}
impl Reader<'_> {
    /// None names an absent session; a present empty session has zero counts.
    /// Message counts belong to MessageSelection because inheritance changes
    /// that sequence while events and links remain physical-session records.
    pub fn physical_record_counts(
        &self,
        session: &SessionId,
    ) -> Result<Option<PhysicalRecordCounts>> {
        let counts = self
            .conn()?
            .query_row(
                "SELECT (SELECT count(*) FROM event e WHERE e.session_pk=s.pk),
                    (SELECT count(*) FROM link l WHERE l.parent_id=s.id OR l.child_id=s.id)
             FROM session s WHERE s.id=?1",
                [session.to_string()],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
            )
            .optional()?;
        counts
            .map(|(events, links)| {
                let checked = |count| {
                    u64::try_from(count)
                        .map_err(|_| StoreError::Invalid("negative physical record count".into()))
                };
                Ok(PhysicalRecordCounts {
                    events: checked(events)?,
                    links: checked(links)?,
                })
            })
            .transpose()
    }
}

#[cfg(test)]
#[path = "record_seek/tests.rs"]
mod tests;
