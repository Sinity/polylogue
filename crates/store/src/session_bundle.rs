//! Reading whole sessions back in the bundle format: the session, its own
//! messages (not inherited ones, which are bundled with the session they
//! came from), its events and the links naming it. Importing the document
//! into a fresh archive gives the same sessions and IDs.

use crate::ConnExt as _;
use crate::Result;
use crate::read::Reader;
use crate::row::{GroupKey, Rows as _, SessionPk, row};
use polylogue_model::JsonExt;
use polylogue_model::{Event, Link, LinkKind, MessageId, SessionId};
use std::ops::ControlFlow;

row! {
    struct ExplicitCopy {
        child_id: String,
        source_input: String,
        source_chain: i64,
        source_version: String,
        position: String,
        record_position: String,
        parent_id: String,
    }
    statements {
        const OF_MESSAGE = "SELECT DISTINCT child_id,source_input,source_chain,source_version,position,record_position,parent_id FROM shared_prefix_message WHERE message_id=?1 ORDER BY child_id,source_input,source_chain,source_version,position,record_position";
    }
}

row! {
    struct InferredCopy {
        child_id: String,
        source_input: String,
        source_chain: i64,
        parser_version: String,
        position: u64,
        record_position: u64,
        parent_id: String,
    }
    statements {
        const OF_MESSAGE = "SELECT DISTINCT p.child_id,ir.id AS source_input,s.chain AS source_chain,ps.parser_version,s.position,s.record_position,p.parent_id FROM message m JOIN session_prefix p ON p.parent_id=?2 AND p.copies_prefix=1 AND p.branch_point IS NULL JOIN message_parse s ON s.message_id=m.id JOIN input_ref ir ON ir.pk=s.input_pk JOIN parse_state ps ON ps.input_pk=s.input_pk AND ps.chain=s.chain WHERE m.id=?1 AND m.duplicate_of IS NULL AND EXISTS(SELECT 1 FROM shared_prefix x WHERE x.input_pk=s.input_pk AND x.chain=s.chain AND x.child_id=p.child_id AND x.parent_id=p.parent_id AND x.explicit_members=0) AND NOT EXISTS(SELECT 1 FROM shared_prefix x WHERE x.input_pk=s.input_pk AND x.chain=s.chain AND x.child_id=p.child_id AND x.parent_id=p.parent_id AND x.explicit_members=1)";
    }
}

row! {
    struct SessionDirectory {
        pk: SessionPk,
        cwd_json: Option<String>,
    }
    statements {
        const OF_ID = "SELECT pk, cwd_json FROM session WHERE id = ?1";
    }
}

row! {
    struct BundleEvent {
        content_group: GroupKey,
        content_start: i64,
        content_length: i64,
        data_length: Option<i64>,
        session_id: Option<SessionId>,
        after_message_id: Option<MessageId>,
        input: String,
        source_chain: i64,
    }
    statements {
        const OF_SESSION = "SELECT e.content_group,e.content_start,e.content_length,e.data_length,
            s.id AS session_id,e.after_message_id,i.id AS input,e.source_chain
            FROM event e LEFT JOIN session s ON s.pk=e.session_pk
            JOIN input_ref i ON i.pk=e.input_pk
            WHERE e.session_pk=?1 ORDER BY e.position,e.id";
        const RELOCATED = "SELECT e.content_group,e.content_start,e.content_length,e.data_length,
            s.id AS session_id,e.after_message_id,'' AS input,0 AS source_chain
            FROM event e LEFT JOIN session s ON s.pk=e.session_pk
            WHERE e.session_pk=?1 AND e.kind='relocated' ORDER BY e.position DESC, e.id DESC LIMIT 1";
    }
}

impl BundleEvent {
    fn event(
        self,
        conn: &rusqlite::Connection,
        reader: &mut crate::content::Reader,
    ) -> Result<(Event, String, i64)> {
        let mut event = crate::writer::readings::read_event(
            conn,
            reader,
            crate::content::Piece {
                group: self.content_group.0,
                start: self.content_start,
                length: self.content_length,
                data_length: self.data_length,
            },
        )?;
        event.session = self.session_id;
        event.after = self.after_message_id;
        Ok((event, self.input, self.source_chain))
    }
}

row! {
    struct BundleLink {
        kind: String,
        reason: Option<String>,
        parent_id: SessionId,
        child_id: SessionId,
        branch_point: Option<MessageId>,
        spawning_call: Option<String>,
        evidence: String,
        copies_prefix: bool,
    }
    statements {
        const OF_SESSION = "SELECT kind, reason, parent_id, child_id, branch_point, spawning_call, evidence,
                 copies_prefix OR EXISTS(SELECT 1 FROM shared_prefix x WHERE x.parent_id = l.parent_id AND x.child_id = l.child_id) AS copies_prefix
             FROM session_link l WHERE parent_id = ?1 OR child_id = ?1
             ORDER BY parent_id, child_id, kind, reason, branch_point, spawning_call LIMIT -1 OFFSET ?2";
    }
}

#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<ExplicitCopy>();
    check.row::<InferredCopy>();
    check.row::<SessionDirectory>();
    check.row::<BundleEvent>();
    check.row::<BundleLink>();
}

impl Reader<'_> {
    /// Original copied-prefix membership beside its message. Imported explicit
    /// members never expand when other messages from their parent arrive later.
    pub fn copied_for(&self, id: &MessageId) -> Result<Vec<polylogue_model::bundle::CopiedFor>> {
        use polylogue_model::{PrefixSource, bundle::CopiedFor};
        let conn = self.conn()?;
        let mut result = Vec::new();
        let _ = conn.each(
            &ExplicitCopy::OF_MESSAGE,
            [id.to_string()],
            |ExplicitCopy {
                 child_id: child,
                 source_input: input,
                 source_chain: chain,
                 source_version: parse_version,
                 position,
                 record_position,
                 parent_id: parent,
             }| {
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
                    position: [coordinate(position)?, coordinate(record_position)?],
                });
                Ok(ControlFlow::Continue(()))
            },
        )?;
        // Bind the message's session before expanding the UNION in session_prefix. A join
        // from the selected message leaves that view computing every edge.
        // Use the stored ID here: explicit links above can also name a
        // message absent from the selected rows, and aliases hold no copies.
        let session: Option<String> = conn.scalar_optional(
            "SELECT s.id FROM message m JOIN session s ON s.pk=m.session_pk
                 WHERE m.id=?1 AND m.duplicate_of IS NULL",
            [id.to_string()],
        )?;
        let _ = conn.each(
            &InferredCopy::OF_MESSAGE,
            rusqlite::params![id.to_string(), session],
            |InferredCopy {
                 child_id: child,
                 source_input: input,
                 source_chain: chain,
                 parser_version: parse_version,
                 position,
                 record_position,
                 parent_id: parent,
             }| {
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
                    position: [position, record_position],
                });
                Ok(ControlFlow::Continue(()))
            },
        )?;
        result.sort();
        result.dedup();
        Ok(result)
    }

    /// The session's events in display order, payloads as written.
    pub fn events(&self, id: &SessionId) -> Result<Vec<Event>> {
        let Some(pk) = self.session_pk(id)? else {
            return Ok(Vec::new());
        };
        let mut reader = crate::content::Reader::default();
        let conn = self.conn()?;
        let mut events = Vec::new();
        let _ = conn.each(&BundleEvent::OF_SESSION, [pk], |row| {
            let (event, input, chain) = row.event(conn, &mut reader)?;
            events.push((event, input, chain));
            Ok(ControlFlow::Continue(()))
        })?;
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
        let conn = self.conn()?;
        let Some(SessionDirectory { pk, cwd_json }) =
            conn.maybe(&SessionDirectory::OF_ID, [id.to_string()])?
        else {
            return Ok(None);
        };
        let mut reader = crate::content::Reader::default();
        let relocated = conn
            .maybe(&BundleEvent::RELOCATED, [pk])?
            .map(|row| row.event(conn, &mut reader))
            .transpose()?;
        if let Some((event, _, _)) = relocated
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
        let _ = self.conn()?.each(
            &BundleLink::OF_SESSION,
            rusqlite::params![id.to_string(), ordinal],
            |BundleLink {
                 kind,
                 reason,
                 parent_id: parent,
                 child_id: child,
                 branch_point,
                 spawning_call: call_id,
                 evidence,
                 copies_prefix,
             }| {
                let kind = LinkKind::from_name(&kind).ok_or_else(|| {
                    crate::StoreError::stored(0, format!("unknown link kind {kind}"))
                })?;
                let reason = reason
                    .map(|reason| {
                        polylogue_model::ReferenceReason::from_name(&reason).ok_or_else(|| {
                            crate::StoreError::stored(
                                0,
                                format!("unknown reference reason {reason}"),
                            )
                        })
                    })
                    .transpose()?;
                let link = Link {
                    kind,
                    copies_prefix,
                    reason,
                    parent,
                    child,
                    branch_point,
                    call_id,
                    evidence,
                };
                link.validate()
                    .map_err(|e| crate::StoreError::stored(0, e.to_owned()))?;
                Ok(visit(link))
            },
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
