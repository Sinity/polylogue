//! Reading whole sessions back in the export format: the session, its own
//! messages (not inherited ones, which are exported with the session they
//! came from), its events and the links naming it. Importing the document
//! into a fresh archive gives the same sessions and IDs.

use crate::Result;
use crate::read::Reader;
use polylogue_model::export::SessionExport;
use polylogue_model::{Event, Link, LinkKind, MessageId, SessionId};
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
            .conn()?
            .prepare_cached("SELECT pk FROM session WHERE id = ?1")?
            .query_row([id.to_string()], |r| r.get(0))
            .optional()?;
        let Some(pk) = pk else {
            return Ok(Vec::new());
        };
        let mut statement = self
            .conn()?
            .prepare_cached("SELECT fields,data FROM event WHERE session_pk=?1 ORDER BY seq,id")?;
        let rows = statement.query_map([pk], |row| {
            Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?;
        rows.map(|row| {
            let (fields, data) = row?;
            crate::events::canonical_event(&fields, &data)
        })
        .collect()
    }

    /// Links where the session is the parent or the child, in an order fixed
    /// by their content: the order inputs arrived in never changes the view
    ///.
    pub fn links(&self, id: &SessionId) -> Result<Vec<Link>> {
        let mut links = Vec::new();
        self.visit_links(id, |link| {
            links.push(link);
            std::ops::ControlFlow::Continue(())
        })?;
        Ok(links)
    }

    /// Visit links in content order, stopping before later payloads are decoded.
    pub fn visit_links(
        &self,
        id: &SessionId,
        mut visit: impl FnMut(Link) -> std::ops::ControlFlow<()>,
    ) -> Result<()> {
        let mut stmt = self.conn()?.prepare_cached(
            "SELECT kind, how, parent_id, child_id, branch_point, spawning_call, evidence
             FROM link WHERE parent_id = ?1 OR child_id = ?1
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
        for row in rows {
            let ((kind, how), parent, child, at, call_id, evidence) = row?;
            let (Some(kind), Ok(parent), Ok(child)) = (
                LinkKind::from_columns(&kind, &how),
                SessionId::parse(&parent),
                SessionId::parse(&child),
            ) else {
                continue;
            };
            let link = Link {
                kind,
                parent,
                child,
                at: at.and_then(|a| MessageId::parse(&a).ok()),
                call_id,
                evidence,
            };
            if visit(link).is_break() {
                break;
            }
        }
        Ok(())
    }
}
