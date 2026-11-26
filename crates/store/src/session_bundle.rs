//! Reading whole sessions back in the bundle format: the session, its own
//! messages (not inherited ones, which are bundled with the session they
//! came from), its events and the links naming it. Importing the document
//! into a fresh archive gives the same sessions and IDs.

use crate::ConnExt as _;
use crate::Result;
use crate::encoding::Columns;
use crate::read::Reader;
use polylogue_model::JsonExt;
use polylogue_model::{Event, Link, LinkKind, MessageId, SessionId};
use rusqlite::OptionalExtension;

impl Reader<'_> {
    /// Original copied-prefix membership beside its message. Imported explicit
    /// members never expand when other messages from their parent arrive later.
    pub fn copied_for(&self, id: &MessageId) -> Result<Vec<polylogue_model::bundle::CopiedFor>> {
        use polylogue_model::{PrefixSource, bundle::CopiedFor};
        let conn = self.conn()?;
        let mut result = Vec::new();
        let mut explicit = conn.prepare_cached("SELECT DISTINCT child_id,source_input,source_chain,source_version,sequence,position,parent_id FROM prefix_member WHERE message_id=?1 ORDER BY child_id,source_input,source_chain,source_version,sequence,position")?;
        for row in explicit.query_map([id.to_string()], |r| {
            <(String, String, i64, String, String, String, String)>::try_from(r)
        })? {
            let (child, input, chain, parse_version, sequence, position, parent) = row?;
            let coordinate = |s: String| {
                s.parse::<u64>()
                    .map_err(|error| crate::StoreError::stored(0, error.to_string()))
            };
            result.push(CopiedFor {
                parent: SessionId::parse(&parent)
                    .map_err(|error| crate::StoreError::stored(0, error.to_string()))?,
                child: SessionId::parse(&child)
                    .map_err(|error| crate::StoreError::stored(0, error.to_string()))?,
                source: PrefixSource {
                    input,
                    chain,
                    parse_version,
                },
                position: [coordinate(sequence)?, coordinate(position)?],
            });
        }
        // Bind the message's session before expanding the UNION in session_prefix. A join
        // from the selected message leaves that view computing every edge.
        // Use the stored ID here: explicit links above can also name a
        // message absent from the selected rows, and aliases hold no copies.
        let session: Option<String> = conn.scalar_optional(
            "SELECT s.id FROM message m JOIN session s ON s.pk=m.session_pk
                 WHERE m.id=?1 AND m.duplicate_of IS NULL",
            [id.to_string()],
        )?;
        let mut inferred = conn.prepare_cached("SELECT DISTINCT p.child_id,ir.id,s.chain,ps.parser_version,s.seq,s.record_position,p.parent_id FROM message m JOIN session_prefix p ON p.parent_id=?2 AND p.copies_prefix=1 AND p.branch_point IS NULL JOIN message_parse s ON s.message_id=m.id JOIN input_ref ir ON ir.pk=s.input_pk JOIN parse_state ps ON ps.input_pk=s.input_pk AND ps.chain=s.chain WHERE m.id=?1 AND m.duplicate_of IS NULL AND EXISTS(SELECT 1 FROM prefix_copy x WHERE x.input_pk=s.input_pk AND x.chain=s.chain AND x.child_id=p.child_id AND x.parent_id=p.parent_id AND x.explicit_members=0) AND NOT EXISTS(SELECT 1 FROM prefix_copy x WHERE x.input_pk=s.input_pk AND x.chain=s.chain AND x.child_id=p.child_id AND x.parent_id=p.parent_id AND x.explicit_members=1)")?;
        for row in inferred.query_map(rusqlite::params![id.to_string(), session], |r| {
            <(String, String, i64, String, u64, u64, String)>::try_from(r)
        })? {
            let (child, input, chain, parse_version, sequence, position, parent) = row?;
            result.push(CopiedFor {
                parent: SessionId::parse(&parent)
                    .map_err(|error| crate::StoreError::stored(0, error.to_string()))?,
                child: SessionId::parse(&child)
                    .map_err(|error| crate::StoreError::stored(0, error.to_string()))?,
                source: PrefixSource {
                    input,
                    chain,
                    parse_version,
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
        let conn = self.conn()?;
        let mut reader = crate::content::Reader::default();
        let mut statement = conn.prepare_cached(&format!(
            "SELECT {},i.id,e.source_chain FROM event e LEFT JOIN session s ON s.pk=e.session_pk
             JOIN input_ref i ON i.pk=e.input_pk
             WHERE e.session_pk=?1 ORDER BY e.seq,e.id",
            crate::events::SHOWN_EVENT_COLUMNS
        ))?;
        let events = statement
            .query_and_then([pk], |row| {
                Ok::<_, crate::StoreError>((
                    crate::events::shown_event(conn, &mut reader, row, 0)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, i64>(7)?,
                ))
            })?
            .collect::<Result<Vec<_>>>()?;
        // What a reading left in the stored bytes is shown from them.
        events
            .into_iter()
            .map(|(mut event, input, chain)| {
                self.resolve_stored_value(&input, chain, &mut event.data)?;
                Ok(event)
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
            .query_row([id.to_string()], |r| <(_, _)>::try_from(r))
            .optional()?;
        let Some((pk, cwd_json)) = found else {
            return Ok(None);
        };
        let conn = self.conn()?;
        let mut reader = crate::content::Reader::default();
        let relocated: Option<Event> = conn
            .prepare_cached(&format!(
                "SELECT {} FROM event e LEFT JOIN session s ON s.pk=e.session_pk
                 WHERE e.session_pk=?1 AND e.kind='relocated' ORDER BY e.seq DESC, e.id DESC LIMIT 1",
                crate::events::SHOWN_EVENT_COLUMNS
            ))?
            .query_and_then([pk], |row| {
                crate::events::shown_event(conn, &mut reader, row, 0)
            })?
            .next()
            .transpose()?;
        if let Some(event) = relocated
            && let Some(dir) = event.data.str("relocatedCwd").filter(|dir| !dir.is_empty())
        {
            return Ok(Some(dir.to_owned()));
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
        self.visit_links(id, 0, |link| {
            links.push(link);
            std::ops::ControlFlow::Continue(())
        })?;
        Ok(links)
    }

    /// Seek within the physical session's stored link order before decoding.
    pub fn visit_links(
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
                branch_point: at,
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
