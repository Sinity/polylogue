//! Record-family cardinalities in the same pinned physical session scope as
//! the event and link ordinal visitors. Payloads are never selected.
use crate::row::{Rows as _, row};
use crate::{Reader, Result};
use polylogue_model::SessionId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalRecordCounts {
    pub events: u64,
    pub links: u64,
}
row! {
    /// How many events and links a session has.
    struct RecordCounts {
        events: u64,
        links: u64,
    }
    statements {
        const OF_SESSION = "SELECT (SELECT count(*) FROM event e WHERE e.session_pk=s.pk) AS events,
            (SELECT count(*) FROM session_link_identity l WHERE l.parent_id=s.id OR l.child_id=s.id) AS links
            FROM session s WHERE s.id=?1";
    }
}

impl Reader<'_> {
    /// None names an absent session; a present empty session has zero counts.
    /// Message counts belong to MessageSelection because inheritance changes
    /// that sequence while events and links remain physical-session records.
    pub fn physical_record_counts(
        &self,
        session: &SessionId,
    ) -> Result<Option<PhysicalRecordCounts>> {
        Ok(self
            .conn()?
            .maybe(&RecordCounts::OF_SESSION, [session.to_string()])?
            .map(|counts| PhysicalRecordCounts {
                events: counts.events,
                links: counts.links,
            }))
    }
}

/// The rows this module declares statements for, for the every-statement test.
#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<RecordCounts>();
}

#[cfg(test)]
#[path = "record_seek/tests.rs"]
mod tests;
