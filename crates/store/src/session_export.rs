//! Reading whole sessions back in the export format: the session, its own
//! messages (not inherited ones, which are exported with the session they
//! came from), its events and the links naming it. Importing the document
//! into a fresh archive gives the same sessions and IDs.

use crate::Result;
use crate::encoding::Columns;
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

    /// Original copied-prefix membership beside its message. Imported explicit
    /// members never expand when other messages from their parent arrive later.
    pub fn copied_for(&self, id: &MessageId) -> Result<Vec<polylogue_model::bundle::CopiedFor>> {
        use polylogue_model::{PrefixSource, bundle::CopiedFor};
        let conn = self.conn()?;
        let mut result = Vec::new();
        let mut explicit = conn.prepare_cached("SELECT DISTINCT child_id,source_input,source_chain,source_version,sequence,position,parent_id FROM prefix_member WHERE message_id=?1 ORDER BY child_id,source_input,source_chain,source_version,sequence,position")?;
        for row in explicit.query_map([id.to_string()], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, String>(5)?,
                r.get::<_, String>(6)?,
            ))
        })? {
            let (child, input, chain, reading_version, sequence, position, parent) = row?;
            let coordinate = |s: String| {
                s.parse::<u64>()
                    .map_err(|e| crate::StoreError::Invalid(e.to_string()))
            };
            result.push(CopiedFor {
                parent: SessionId::parse(&parent)
                    .map_err(|e| crate::StoreError::Invalid(e.to_string()))?,
                child: SessionId::parse(&child)
                    .map_err(|e| crate::StoreError::Invalid(e.to_string()))?,
                source: PrefixSource {
                    input,
                    chain,
                    reading_version,
                },
                position: [coordinate(sequence)?, coordinate(position)?],
            });
        }
        // Bind the owner before expanding the UNION in session_prefix. A join
        // from the selected message leaves that view materializing every edge.
        // Use the exact stored ID here: explicit evidence above can also name a
        // message absent from the selected rows, and aliases do not own copies.
        let owner: Option<String> = conn
            .query_row(
                "SELECT s.id FROM message m JOIN session s ON s.pk=m.session_pk
                 WHERE m.id=?1 AND m.duplicate_of IS NULL",
                [id.to_string()],
                |row| row.get(0),
            )
            .optional()?;
        let mut inferred = conn.prepare_cached("SELECT DISTINCT p.child_id,ir.id,s.chain,ps.reading_version,s.seq,s.record_position,p.parent_id FROM message m JOIN session_prefix p ON p.parent_id=?2 AND p.copies_prefix=1 AND p.branch_point IS NULL JOIN message_source s ON s.message_pk=m.pk JOIN input_ref ir ON ir.pk=s.input_pk JOIN parse_state ps ON ps.input_pk=s.input_pk AND ps.chain=s.chain WHERE m.id=?1 AND m.duplicate_of IS NULL AND EXISTS(SELECT 1 FROM prefix_copy x WHERE x.input_pk=s.input_pk AND x.chain=s.chain AND x.child_id=p.child_id AND x.parent_id=p.parent_id AND x.explicit_members=0) AND NOT EXISTS(SELECT 1 FROM prefix_copy x WHERE x.input_pk=s.input_pk AND x.chain=s.chain AND x.child_id=p.child_id AND x.parent_id=p.parent_id AND x.explicit_members=1)")?;
        for row in inferred.query_map(rusqlite::params![id.to_string(), owner], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, u64>(4)?,
                r.get::<_, u64>(5)?,
                r.get::<_, String>(6)?,
            ))
        })? {
            let (child, input, chain, reading_version, sequence, position, parent) = row?;
            result.push(CopiedFor {
                parent: SessionId::parse(&parent)
                    .map_err(|e| crate::StoreError::Invalid(e.to_string()))?,
                child: SessionId::parse(&child)
                    .map_err(|e| crate::StoreError::Invalid(e.to_string()))?,
                source: PrefixSource {
                    input,
                    chain,
                    reading_version,
                },
                position: [sequence, position],
            });
        }
        result.sort();
        result.dedup();
        Ok(result)
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

    /// The directory a session resumes in: where it was launched, or where
    /// its tool says it was relocated (a Claude Code `relocated` record),
    /// never a directory it moved to while working. `None` when the session
    /// recorded no directory.
    pub fn resume_directory(&self, id: &SessionId) -> Result<Option<String>> {
        let found: Option<(i64, Option<String>)> = self
            .conn()?
            .prepare_cached("SELECT pk, cwd_json FROM session WHERE id = ?1")?
            .query_row([id.to_string()], |r| Ok((r.get(0)?, r.get(1)?)))
            .optional()?;
        let Some((pk, cwd_json)) = found else {
            return Ok(None);
        };
        let relocated: Option<(Vec<u8>, Vec<u8>)> = self
            .conn()?
            .prepare_cached(
                "SELECT fields,data FROM event WHERE session_pk=?1 AND kind='relocated'
                 ORDER BY seq DESC, id DESC LIMIT 1",
            )?
            .query_row([pk], |r| Ok((r.get(0)?, r.get(1)?)))
            .optional()?;
        if let Some((fields, data)) = relocated {
            let event = crate::events::canonical_event(&fields, &data)?;
            if let Some(dir) = event
                .data
                .get("relocatedCwd")
                .and_then(serde_json::Value::as_str)
                .filter(|dir| !dir.is_empty())
            {
                return Ok(Some(dir.to_owned()));
            }
        }
        let cwd: Vec<String> = cwd_json
            .map(|json| serde_json::from_str(&json))
            .transpose()?
            .unwrap_or_default();
        Ok(cwd.into_iter().next())
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
        visit: impl FnMut(Link) -> std::ops::ControlFlow<()>,
    ) -> Result<()> {
        self.visit_links_from_ordinal(id, 0, visit)
    }

    /// Seek within the physical session's canonical link order before decoding.
    pub fn visit_links_from_ordinal(
        &self,
        id: &SessionId,
        ordinal: u64,
        mut visit: impl FnMut(Link) -> std::ops::ControlFlow<()>,
    ) -> Result<()> {
        let Ok(ordinal) = i64::try_from(ordinal) else {
            return Ok(());
        };
        let mut stmt = self.conn()?.prepare_cached(
            "SELECT kind, how, parent_id, child_id, branch_point, spawning_call, evidence, copies_prefix
             FROM session_link WHERE parent_id = ?1 OR child_id = ?1
             ORDER BY parent_id, child_id, kind, how, branch_point, spawning_call LIMIT -1 OFFSET ?2",
        )?;
        let rows = stmt.query_map(rusqlite::params![id.to_string(), ordinal], |r| {
            Ok((
                (r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?),
                r.id::<SessionId>(2)?,
                r.id::<SessionId>(3)?,
                r.optional_id::<MessageId>(4)?,
                r.get::<_, Option<String>>(5)?,
                r.get::<_, String>(6)?,
                r.get::<_, bool>(7)?,
            ))
        })?;
        for row in rows {
            let ((kind, how), parent, child, at, call_id, evidence, copies_prefix) = row?;
            let kind = LinkKind::from_name(&kind)
                .ok_or_else(|| crate::StoreError::stored(0, format!("unknown link kind {kind}")))?;
            let how = how
                .map(|how| {
                    polylogue_model::ReferenceReason::from_name(&how).ok_or_else(|| {
                        crate::StoreError::stored(0, format!("unknown reference reason {how}"))
                    })
                })
                .transpose()?;
            let link = Link {
                kind,
                copies_prefix,
                how,
                parent,
                child,
                at,
                call_id,
                evidence,
            };
            link.validate()
                .map_err(|e| crate::StoreError::stored(0, e.to_owned()))?;
            if visit(link).is_break() {
                break;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
