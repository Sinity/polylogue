//! Message selection in one pinned read, with incomplete parents kept explicit.
use super::*;
mod navigation;
pub use navigation::OutlineEntry;
use polylogue_model::masking::MessageMasker;
use std::collections::VecDeque;
use std::ops::ControlFlow;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, thiserror::Error)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ParentIssue {
    #[error("the parents loop back to {session}")]
    Cycle { session: SessionId },
    #[error("parent {parent} of {child} is not available")]
    MissingParent { child: SessionId, parent: SessionId },
    #[error("parent ID {parent:?} of {child} is invalid")]
    InvalidParent { child: SessionId, parent: String },
    #[error("fork from {parent} to {child} has no available branch anchor ({anchor:?})")]
    MissingAnchor {
        child: SessionId,
        parent: SessionId,
        anchor: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Parents {
    pub complete: bool,
    pub issues: Vec<ParentIssue>,
}

struct Key {
    session_pk: i64,
    session: SessionId,
    message_pk: i64,
    id: String,
}

struct Link {
    parent: String,
    how: String,
    anchor: Option<String>,
}

struct Frame {
    pk: i64,
    session: SessionId,
    links: VecDeque<Link>,
    keys: Vec<Key>,
    issues: Vec<ParentIssue>,
    /// The fork whose parent view this frame is constructing.
    via: Option<(SessionId, Option<String>)>,
}

/// Where a visit of a selection starts: a message's place in it, and the
/// message's key when the caller knows it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VisitStart {
    pub message: usize,
    pub key: Option<i64>,
}

pub(super) struct Visibility {
    pub(super) kinds: String,
    pub(super) prompts: bool,
}

/// A message selection belongs to the reader that made it. Bodies are read
/// lazily; the parents' history keeps keys for ordering, anchors and deduplication.
pub struct MessageSelection<'r, 'a> {
    reader: &'r Reader<'a>,
    own: Option<(i64, SessionId)>,
    with_parents: Option<Vec<Key>>,
    parents: Parents,
    /// Whether tool results carry their structured data; a text read leaves
    /// it unread ([`Self::for_text`]).
    structured: bool,
    visibility: Option<Visibility>,
}

impl<'a> Reader<'a> {
    pub fn select_messages<'r>(
        &'r self,
        id: &SessionId,
        with_parents: bool,
    ) -> Result<MessageSelection<'r, 'a>> {
        let own = self.session_pk(id)?.map(|pk| (pk, id.clone()));
        let has_parents = with_parents && self.conn()?.query_row(
            "SELECT EXISTS(SELECT 1 FROM link WHERE child_id=?1 AND kind='continuation' AND how IN ('fork','resume'))",
            [id.to_string()], |r| r.get::<_, bool>(0))?;
        let (keys, issues) = match own.as_ref().filter(|_| has_parents) {
            Some((pk, _)) => self.parent_keys(*pk, id)?,
            None => (Vec::new(), Vec::new()),
        };
        Ok(MessageSelection {
            reader: self,
            own,
            with_parents: has_parents.then_some(keys),
            parents: Parents {
                complete: issues.is_empty(),
                issues,
            },
            structured: true,
            visibility: None,
        })
    }

    fn parent_frame(
        &self,
        pk: i64,
        session: SessionId,
        via: Option<(SessionId, Option<String>)>,
    ) -> Result<Frame> {
        let links = self
            .conn()?
            .prepare_cached(
                "SELECT parent_id, how, branch_point FROM link
                 WHERE child_id = ?1 AND kind = 'continuation' AND how IN ('fork', 'resume')
                 ORDER BY pk",
            )?
            .query_map([session.to_string()], |r| {
                Ok(Link {
                    parent: r.get(0)?,
                    how: r.get(1)?,
                    anchor: r.get(2)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(Frame {
            pk,
            session,
            links,
            keys: Vec::new(),
            issues: Vec::new(),
            via,
        })
    }

    fn parent_keys(&self, pk: i64, id: &SessionId) -> Result<(Vec<Key>, Vec<ParentIssue>)> {
        let mut frames = vec![self.parent_frame(pk, id.clone(), None)?];
        let mut active = HashSet::from([pk]);
        loop {
            let frame = frames
                .last_mut()
                .expect("the root frame remains until return");
            if let Some(link) = frame.links.pop_front() {
                let parent = match SessionId::parse(&link.parent) {
                    Ok(parent) => parent,
                    Err(_) => {
                        frame.issues.push(ParentIssue::InvalidParent {
                            child: frame.session.clone(),
                            parent: link.parent,
                        });
                        continue;
                    }
                };
                let parent_pk = self
                    .conn()?
                    .prepare_cached("SELECT pk FROM session WHERE id=?1 AND stub=0")?
                    .query_row([parent.to_string()], |r| r.get::<_, i64>(0))
                    .optional()?;
                let Some(parent_pk) = parent_pk else {
                    frame.issues.push(ParentIssue::MissingParent {
                        child: frame.session.clone(),
                        parent,
                    });
                    continue;
                };
                if active.contains(&parent_pk) {
                    frame.issues.push(ParentIssue::Cycle { session: parent });
                    continue;
                }
                if link.how == "resume" {
                    let copied = self
                        .conn()?
                        .prepare_cached(
                            "SELECT DISTINCT m.pk, m.id, m.order_input_id, m.seq, m.record_position FROM message m
                             JOIN message_source s ON s.message_pk = m.pk
                             JOIN parsed p ON p.input_pk = s.input_pk AND p.session_pk = ?2
                             WHERE m.session_pk = ?1 AND m.duplicate_of IS NULL ORDER BY m.order_input_id, m.seq, m.record_position, m.id",
                        )?
                        .query_map(params![parent_pk, frame.pk], |r| {
                            Ok(Key {
                                session_pk: parent_pk,
                                session: parent.clone(),
                                message_pk: r.get(0)?,
                                id: r.get(1)?,
                            })
                        })?
                        .collect::<rusqlite::Result<Vec<_>>>()?;
                    frame.keys.extend(copied);
                } else if link.anchor.is_none() {
                    frame.issues.push(ParentIssue::MissingAnchor {
                        child: frame.session.clone(),
                        parent,
                        anchor: None,
                    });
                } else {
                    let child = frame.session.clone();
                    active.insert(parent_pk);
                    frames.push(self.parent_frame(
                        parent_pk,
                        parent,
                        Some((child, link.anchor)),
                    )?);
                }
                continue;
            }
            let mut done = frames.pop().expect("the current frame exists");
            for (message_pk, id) in self.own_keys(done.pk)? {
                done.keys.push(Key {
                    session_pk: done.pk,
                    session: done.session.clone(),
                    message_pk,
                    id,
                });
            }
            active.remove(&done.pk);
            let Some(parent) = frames.last_mut() else {
                let mut seen = HashSet::new();
                done.keys.retain(|key| seen.insert(key.message_pk));
                return Ok((done.keys, done.issues));
            };
            let (child, anchor) = done.via.expect("only the root has no incoming fork");
            match done
                .keys
                .iter()
                .position(|key| Some(&key.id) == anchor.as_ref())
            {
                Some(at) => done.keys.truncate(at + 1),
                None => {
                    done.keys.clear();
                    done.issues.push(ParentIssue::MissingAnchor {
                        child,
                        parent: done.session,
                        anchor,
                    });
                }
            }
            // Move the common single-parent prefix rather than copying it at
            // every ancestor of a long chain.
            if parent.keys.is_empty() {
                parent.keys = done.keys;
            } else {
                parent.keys.extend(done.keys);
            }
            parent.issues.extend(done.issues);
        }
    }
}

impl MessageSelection<'_, '_> {
    /// The same selection read for a text view, which never shows a tool
    /// result's structured data: it is left unread, so a large value costs
    /// nothing to page past.
    pub fn for_text(mut self) -> Self {
        self.structured = false;
        self.visibility = None;
        self
    }

    /// Hidden messages without PEM markers cannot change ordered masking
    /// state. Keep their metadata positions while leaving their bodies unread.
    pub fn for_blocks(mut self, kinds: &[polylogue_model::BlockKind], prompts: bool) -> Self {
        self.structured = false;
        self.visibility = Some(Visibility {
            kinds: kinds
                .iter()
                .map(|kind| format!("'{}'", kind.as_str()))
                .collect::<Vec<_>>()
                .join(","),
            prompts,
        });
        self
    }

    pub fn parents(&self) -> &Parents {
        &self.parents
    }

    pub(super) fn require_complete(&self) -> Result<()> {
        match self.parents.issues.first() {
            Some(issue) => Err(StoreError::IncompleteParents(issue.clone())),
            None => Ok(()),
        }
    }

    pub fn visit(&self, visit: impl FnMut(Message) -> ControlFlow<()>) -> Result<()> {
        self.visit_from(VisitStart::default(), visit).map(drop)
    }

    /// Skip source messages before hydrating their blocks. Rendering callers must
    /// restore masking state checked against this exact selection. With the
    /// start message's key, a session's own messages before it are not read
    /// at all. Returns the key of the last message visited.
    pub fn visit_from(
        &self,
        start: VisitStart,
        visit: impl FnMut(Message) -> ControlFlow<()>,
    ) -> Result<Option<i64>> {
        let mut remaining = start.message;
        let after_prefix = |_| {
            if remaining == 0 {
                true
            } else {
                remaining -= 1;
                false
            }
        };
        match (start.key, &self.with_parents) {
            (Some(key), None) => self.visit_kept(Some(key), |_| true, visit),
            _ => self.visit_kept(None, after_prefix, visit),
        }
    }

    /// Visits the selected messages `keep` takes, in order, from the own
    /// session's message `from` when given; `keep` sees each key before its
    /// blocks are read.
    fn visit_kept(
        &self,
        from: Option<i64>,
        mut keep: impl FnMut(i64) -> bool,
        mut visit: impl FnMut(Message) -> ControlFlow<()>,
    ) -> Result<Option<i64>> {
        let mut last = None;
        let mut visit = |pk, message| {
            last = Some(pk);
            visit(message)
        };
        if let Some(keys) = &self.with_parents {
            let mut start = 0;
            while start < keys.len() {
                let key = &keys[start];
                let end = start
                    + keys[start..]
                        .iter()
                        .take_while(|k| k.session_pk == key.session_pk)
                        .count();
                let set: HashSet<_> = keys[start..end].iter().map(|k| k.message_pk).collect();
                if self
                    .reader
                    .visit_messages_where(
                        key.session_pk,
                        &key.session,
                        None,
                        MessageContent {
                            structured: self.structured,
                            visibility: self.visibility.as_ref(),
                        },
                        |m| set.contains(&m) && keep(m),
                        &mut visit,
                    )?
                    .is_break()
                {
                    return Ok(last);
                }
                start = end;
            }
        } else if let Some((pk, id)) = &self.own {
            let _ = self.reader.visit_messages_where(
                *pk,
                id,
                from,
                MessageContent {
                    structured: self.structured,
                    visibility: self.visibility.as_ref(),
                },
                keep,
                &mut visit,
            )?;
        }
        Ok(last)
    }

    /// The sessions of the selection in order, each with the messages it
    /// contributes when that is not all of its own.
    fn sessions(&self) -> Vec<(i64, Option<Vec<i64>>)> {
        match (&self.with_parents, &self.own) {
            (Some(keys), _) => {
                let mut sessions: Vec<(i64, Option<Vec<i64>>)> = Vec::new();
                for key in keys {
                    match sessions.last_mut() {
                        Some((pk, Some(members))) if *pk == key.session_pk => {
                            members.push(key.message_pk);
                        }
                        _ => sessions.push((key.session_pk, Some(vec![key.message_pk]))),
                    }
                }
                sessions
            }
            (None, Some((pk, _))) => vec![(*pk, None)],
            (None, None) => Vec::new(),
        }
    }

    /// The masking state before the selection's message `message` when a
    /// private key block may be open there; `None` when masking starts
    /// afresh. An open key is the only masking state that crosses a block
    /// boundary, and only messages holding a PEM header or END line
    /// (`message.pem`) open or close one. So the state is the masker after
    /// those messages alone, in order from the first that leaves a key open.
    pub fn masking_before(&self, start: VisitStart) -> Result<Option<MessageMasker>> {
        let marked = self.marked_before(start)?;
        let Some(first) = marked.iter().position(|(_, pem)| *pem == 2) else {
            return Ok(None);
        };
        let mut masker = MessageMasker::default();
        for (pk, _) in &marked[first..] {
            for mut block in self.reader.message_blocks(*pk, false)? {
                masker.block_text(&mut block);
            }
        }
        Ok(Some(masker))
    }

    /// Only marked rows before the requested message, in reading order.
    /// A cursor's key avoids reading the ordinary message keys as well.
    fn marked_before(&self, start: VisitStart) -> Result<Vec<(i64, i64)>> {
        let conn = self.reader.conn()?;
        if self.with_parents.is_none()
            && let (Some((session, _)), Some(key)) = (&self.own, start.key)
        {
            let from = self.reader.message_order(key, *session)?.ok_or_else(|| {
                StoreError::Invalid("the read starts at a message the session does not hold".into())
            })?;
            let mut statement = conn.prepare_cached(
                "SELECT pk, pem, order_input_id, seq, record_position, id
                 FROM message WHERE session_pk = ?1 AND pem <> 0 AND duplicate_of IS NULL",
            )?;
            let mut marked: Vec<(MessageOrder, i64, i64)> = statement
                .query_map([session], |r| {
                    Ok((
                        (r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?),
                        r.get(0)?,
                        r.get(1)?,
                    ))
                })?
                .collect::<rusqlite::Result<_>>()?;
            marked.retain(|(order, _, _)| order < &from);
            marked.sort_unstable_by(|a, b| a.0.cmp(&b.0));
            return Ok(marked.into_iter().map(|(_, pk, pem)| (pk, pem)).collect());
        }
        let mut marked = HashMap::new();
        let mut statement =
            conn.prepare_cached("SELECT pk, pem FROM message WHERE session_pk = ?1 AND pem <> 0")?;
        for (session, _) in self.sessions() {
            for row in statement.query_map([session], |r| Ok((r.get::<_, i64>(0)?, r.get(1)?)))? {
                let (pk, pem): (i64, i64) = row?;
                marked.insert(pk, pem);
            }
        }
        if !marked.values().any(|pem| *pem == 2) {
            return Ok(Vec::new());
        }
        Ok(self
            .keys()?
            .into_iter()
            .take(start.message)
            .filter_map(|pk| marked.get(&pk).map(|pem| (pk, *pem)))
            .collect())
    }

    pub(super) fn keys(&self) -> Result<Vec<i64>> {
        match &self.with_parents {
            Some(keys) => Ok(keys.iter().map(|key| key.message_pk).collect()),
            None => match &self.own {
                Some((pk, _)) => Ok(self
                    .reader
                    .own_keys(*pk)?
                    .into_iter()
                    .map(|(pk, _)| pk)
                    .collect()),
                None => Ok(Vec::new()),
            },
        }
    }

    /// What the selected content is: parents status; for the session read,
    /// the write counter of its last change (`session.changed`, which every
    /// change to its messages, their versions and blocks, or its derived rows
    /// moves); for each parent, the row and shown version of each message it
    /// contributes, since a parent's later messages are not selected. No
    /// block or text is read: blocks are written only with a new shown
    /// version, whose content hash names them.
    pub fn fingerprint(&self) -> Result<String> {
        let mut hash = blake3::Hasher::new();
        if self.with_parents.is_some() {
            hash.update(&serde_json::to_vec(&self.parents)?);
        }
        let conn = self.reader.conn()?;
        let mut changed = conn.prepare_cached("SELECT changed FROM session WHERE pk = ?1")?;
        let mut facts = conn.prepare_cached("SELECT * FROM message WHERE pk = ?1")?;
        let mut version = conn.prepare_cached(
            "SELECT version_hash FROM message_source WHERE message_pk = ?1 AND shown = 1",
        )?;
        let own = self.own.as_ref().map(|(pk, _)| *pk);
        for (session, members) in self.sessions() {
            hash.update(b"session");
            hash.update(&session.to_le_bytes());
            if Some(session) == own {
                let counter: i64 = changed.query_row([session], |r| r.get(0))?;
                hash.update(&counter.to_le_bytes());
                continue;
            }
            for pk in members.iter().flatten() {
                for statement in [&mut facts, &mut version] {
                    let mut rows = statement.query([pk])?;
                    while let Some(row) = rows.next()? {
                        hash_row(&mut hash, row)?;
                    }
                }
            }
        }
        Ok(hash.finalize().to_hex().to_string())
    }

    /// Fold only import metadata and current byte catalogs associated
    /// with the selected message versions. No message or asset body is loaded.
    pub fn attachment_fingerprint(&self) -> Result<String> {
        let conn = self.reader.conn()?;
        let mut scopes = std::collections::BTreeSet::new();
        // The imports of every input parsed into the selected sessions: a
        // superset of those holding the selected versions, read without
        // visiting each message.
        let mut selected = conn.prepare_cached(
            "SELECT DISTINCT own.import_id FROM parsed p
            JOIN input_ref ir ON ir.pk=p.input_pk JOIN a.input i ON i.id=ir.id
            JOIN import_member own ON own.input=ir.id
            WHERE p.session_pk=?1",
        )?;
        for (session, _) in self.sessions() {
            for scope in selected.query_map([session], |row| row.get::<_, String>(0))? {
                scopes.insert(scope?);
            }
        }
        let mut hash = blake3::Hasher::new();
        let mut stmt=conn.prepare_cached("SELECT m.observation,m.input,m.member,m.source_id,r.chain,r.length,e.offset,e.length,e.blob,e.blob_offset,b.state,
            metadata.ordinal,metadata.source_id,metadata.name,metadata.media_type,metadata.size,s.version,s.error
            FROM import_member m
            LEFT JOIN a.revision r ON r.input=m.input AND r.number=(SELECT max(latest.number) FROM a.revision latest WHERE latest.input=m.input)
            LEFT JOIN a.extent e ON e.input=r.input AND e.chain=r.chain LEFT JOIN a.blob b ON b.hash=e.blob
            LEFT JOIN import_asset_metadata metadata ON metadata.input=m.input AND metadata.sidecar=m.sidecar
            LEFT JOIN import_asset_sidecar s ON s.input=m.input AND s.sidecar=m.sidecar
            WHERE m.import_id=?1 ORDER BY m.observation,e.offset,metadata.ordinal")?;
        for scope in scopes {
            hash.update(&(scope.len() as u64).to_le_bytes());
            hash.update(scope.as_bytes());
            let mut rows = stmt.query([scope])?;
            while let Some(row) = rows.next()? {
                hash_row(&mut hash, row)?;
            }
        }
        Ok(hash.finalize().to_hex().to_string())
    }
}

/// Folds every column of a row into `hash`, with its type.
fn hash_row(hash: &mut blake3::Hasher, row: &rusqlite::Row<'_>) -> Result<()> {
    use rusqlite::types::ValueRef;
    hash.update(b"row");
    for column in 0..row.as_ref().column_count() {
        match row.get_ref(column)? {
            ValueRef::Null => {
                hash.update(b"n");
            }
            ValueRef::Integer(value) => {
                hash.update(b"i");
                hash.update(&value.to_le_bytes());
            }
            ValueRef::Real(value) => {
                hash.update(b"r");
                hash.update(&value.to_bits().to_le_bytes());
            }
            ValueRef::Text(bytes) | ValueRef::Blob(bytes) => {
                hash.update(b"b");
                hash.update(&(bytes.len() as u64).to_le_bytes());
                hash.update(bytes);
            }
        }
    }
    Ok(())
}
