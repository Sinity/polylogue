//! User state as it moves between archives: every request a person or agent
//! made and every excision, in commit order. Replayed in that order through
//! the writer, they rebuild the notes, judgments, settings, kept results,
//! deliveries, chat actions and backfill jobs they wrote, and the excisions
//! keep refusing what they removed. This is the same sequence primary mode
//! keeps in Sinex ([`crate::primary::RegisterChange`]).
use crate::primary::RegisterChange;
use crate::user_writes::UserRequest;
use crate::{Reader, Result, StoreError};

impl Reader<'_> {
    /// Every request and excision in this archive, in the order they were
    /// written.
    pub fn user_state(&self) -> Result<Vec<RegisterChange>> {
        let conn = self.conn()?;
        // (commit_seq, created_at, id, change): one write counter orders
        // requests and excisions together.
        let mut entries: Vec<(i64, i64, String, RegisterChange)> = Vec::new();
        let mut requests =
            conn.prepare("SELECT id, request, commit_seq, created_at FROM a.client_request")?;
        let rows = requests.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<Vec<u8>>>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, i64>(3)?,
            ))
        })?;
        for row in rows {
            let (id, packed, seq, at) = row?;
            let packed = packed.ok_or_else(|| {
                StoreError::Invalid(format!(
                    "request {id} was written before requests were kept whole, so it cannot be exported"
                ))
            })?;
            let request: UserRequest =
                serde_json::from_str(&crate::encoding::unpack_text(&packed)?)?;
            entries.push((
                seq,
                at,
                id,
                RegisterChange::Request {
                    request: Box::new(request),
                    first_note: None,
                },
            ));
        }
        let excisions: Vec<(String, i64, i64)> = conn
            .prepare("SELECT id, commit_seq, created_at FROM a.excision")?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
            .collect::<rusqlite::Result<_>>()?;
        for (id, seq, at) in excisions {
            let rows = crate::primary::excision_rows(conn, &id)?;
            entries.push((seq, at, id, RegisterChange::Excision(rows)));
        }
        entries.sort_by(|a, b| (a.0, a.1, &a.2).cmp(&(b.0, b.1, &b.2)));
        Ok(entries.into_iter().map(|(.., change)| change).collect())
    }
}
