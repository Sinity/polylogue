//! Reading whole sessions back in the export format (architecture 7,
//! Export and import): the session, its own messages (not inherited ones,
//! which are exported with the session they came from), its events and
//! the links naming it. Importing the document into a fresh archive gives
//! the same sessions and IDs.

use crate::Result;
use crate::read::Reader;
use polylogue_model::export::SessionExport;
use polylogue_model::{Event, Json, Link, LinkKind, MessageId, SessionId, Timestamp};
use rusqlite::OptionalExtension;

impl Reader<'_> {
    /// One session as the export format holds it, or `None` when the
    /// archive has no such session.
    pub fn session_export(&self, id: &SessionId) -> Result<Option<SessionExport>> {
        let Some(session) = self.session(id)? else {
            return Ok(None);
        };
        let messages = self.messages(id, false)?;
        Ok(Some(SessionExport {
            session,
            messages,
            events: self.events(id)?,
            links: self.links(id)?,
        }))
    }

    /// The session's events in display order, payloads as written.
    pub fn events(&self, id: &SessionId) -> Result<Vec<Event>> {
        let pk: Option<i64> = self
            .conn()
            .prepare_cached("SELECT pk FROM session WHERE id = ?1")?
            .query_row([id.to_string()], |r| r.get(0))
            .optional()?;
        let Some(pk) = pk else {
            return Ok(Vec::new());
        };
        let mut stmt = self.conn().prepare_cached(
            "SELECT e.kind, e.at, m.id, e.source_start, e.source_end, e.data
             FROM current_event e LEFT JOIN current_message m ON m.pk = e.after_message_pk
             -- Events at the same position (each file's own byte 0, from
             -- different inputs) sort by their content-derived ID, not by
             -- which file was read first.
             WHERE e.session_pk = ?1 ORDER BY e.seq, e.id",
        )?;
        let rows = stmt.query_map([pk], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<i64>>(1)?,
                r.get::<_, Option<String>>(2)?,
                r.get::<_, Option<i64>>(3)?,
                r.get::<_, Option<i64>>(4)?,
                r.get::<_, Vec<u8>>(5)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let (kind, at, after, start, end, data) = row?;
            let data: Json = crate::encoding::unpack_json(&data)?;
            out.push(Event {
                session: id.clone(),
                kind,
                at: at.and_then(Timestamp::from_unix_ms),
                after: after.and_then(|a| MessageId::parse(&a).ok()),
                range: start.zip(end).map(|(s, e)| (s as u64, e as u64)),
                data,
            });
        }
        Ok(out)
    }

    /// Links where the session is the parent or the child, in an order fixed
    /// by their content: the order inputs arrived in never changes the view
    /// (law 4).
    pub fn links(&self, id: &SessionId) -> Result<Vec<Link>> {
        let mut stmt = self.conn().prepare_cached(
            "SELECT kind, how, parent_id, child_id, branch_point, spawning_call, evidence
             FROM current_link WHERE parent_id = ?1 OR child_id = ?1
             ORDER BY parent_id, child_id, kind, how",
        )?;
        let rows = stmt.query_map([id.to_string()], |r| {
            Ok((
                (r.get::<_, String>(0)?, r.get::<_, String>(1)?),
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, Option<String>>(4)?,
                r.get::<_, Option<String>>(5)?,
                r.get::<_, String>(6)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            let ((kind, how), parent, child, at, call_id, evidence) = row?;
            let (Some(kind), Ok(parent), Ok(child)) = (
                LinkKind::from_columns(&kind, &how),
                SessionId::parse(&parent),
                SessionId::parse(&child),
            ) else {
                continue;
            };
            out.push(Link {
                kind,
                parent,
                child,
                at: at.and_then(|a| MessageId::parse(&a).ok()),
                call_id,
                evidence,
            });
        }
        Ok(out)
    }
}
