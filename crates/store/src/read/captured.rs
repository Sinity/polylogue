//! What the index holds of conversations the browser extension captures:
//! enough for the extension to show a page's state and to skip, during a
//! backfill, conversations the provider has not changed since their last
//! capture.
use super::*;

/// One captured conversation's session in the index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedSession {
    pub messages: u64,
    /// When the latest stored capture of it was taken (ms): the provider's
    /// conversation at that moment is in the archive.
    pub last_captured_at: Option<i64>,
}

impl Reader<'_> {
    /// Each session's state, in the order given; `None` for a session the
    /// index does not hold.
    pub fn captured_sessions(
        &self,
        sessions: &[polylogue_model::SessionId],
    ) -> Result<Vec<Option<CapturedSession>>> {
        let mut stmt = self.conn()?.prepare_cached(
            "SELECT (SELECT count(*) FROM message m WHERE m.session_pk = s.pk),
                    (SELECT max(e.at) FROM event e
                     WHERE e.session_pk = s.pk AND e.kind = 'browser_capture')
             FROM session s WHERE s.id = ?1",
        )?;
        sessions
            .iter()
            .map(|id| {
                Ok(stmt
                    .query_row([id.to_string()], |r| {
                        Ok(CapturedSession {
                            messages: u64::try_from(r.get::<_, i64>(0)?).unwrap_or(0),
                            last_captured_at: r.get(1)?,
                        })
                    })
                    .optional()?)
            })
            .collect()
    }
}
