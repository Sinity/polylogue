//! Message selection in one pinned read, with incomplete parents kept explicit.
use super::*;
mod navigation;
use std::collections::VecDeque;
use std::ops::ControlFlow;

pub use polylogue_model::read::{ParentIssue, Parents};

struct Key {
    session_pk: i64,
    session: SessionId,
    message_pk: i64,
    id: String,
    order: i64,
}

struct Prefix {
    parent: String,
    copies_prefix: bool,
    anchor: Option<String>,
}

struct CopiedKey {
    source: String,
    chain: i64,
    version: String,
    sequence: u64,
    position: u64,
    key: Key,
}

struct Frame {
    pk: i64,
    session: SessionId,
    prefixes: VecDeque<Prefix>,
    keys: Vec<Key>,
    copied: Vec<CopiedKey>,
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
        let has_parents = with_parents
            && self.conn()?.query_row(
                "SELECT EXISTS(SELECT 1 FROM session_prefix WHERE child_id=?1)",
                [id.to_string()],
                |r| r.get::<_, bool>(0),
            )?;
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
        let mut prefixes: VecDeque<Prefix> = self.conn()?
            .prepare_cached("SELECT parent_id,branch_point,copies_prefix FROM session_prefix WHERE child_id=?1 ORDER BY parent_id,branch_point")?
            .query_map([session.to_string()], |r| Ok(Prefix {
                parent: r.get(0)?, anchor: r.get(1)?, copies_prefix: r.get(2)?,
            }))?.collect::<rusqlite::Result<_>>()?;
        let forks: std::collections::BTreeSet<_> = prefixes
            .iter()
            .filter_map(|link| {
                Some((
                    SessionId::parse(&link.parent).ok()?,
                    link.anchor.as_ref()?.clone(),
                ))
            })
            .collect();
        let mut issues = Vec::new();
        if forks.len() > 1 {
            let mut recorded = forks.into_iter();
            let (first, first_anchor) = recorded.next().expect("conflicting recorded forks");
            for (second, second_anchor) in recorded {
                issues.push(ParentIssue::ConflictingParents {
                    child: session.clone(),
                    first: first.clone(),
                    first_anchor: first_anchor.clone(),
                    second,
                    second_anchor,
                });
            }
            // No branch is selected from incompatible recorded fork points.
            prefixes.retain(|prefix| {
                prefix.anchor.is_none() || SessionId::parse(&prefix.parent).is_err()
            });
            // Conflicting provider anchors do not erase independently recorded
            // physical copies. Their exact members choose no provider anchor.
            let copies=self.conn()?.prepare_cached("SELECT DISTINCT parent_id FROM prefix_copy WHERE child_id=?1 AND explicit_members=1 ORDER BY parent_id")?
                .query_map([session.to_string()], |r| r.get::<_,String>(0))?.collect::<rusqlite::Result<Vec<_>>>()?;
            for parent in copies {
                if !prefixes
                    .iter()
                    .any(|prefix| prefix.parent == parent && prefix.anchor.is_none())
                {
                    prefixes.push_back(Prefix {
                        parent,
                        copies_prefix: true,
                        anchor: None,
                    });
                }
            }
        }
        Ok(Frame {
            pk,
            session,
            prefixes,
            keys: Vec::new(),
            copied: Vec::new(),
            issues,
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
            if let Some(link) = frame.prefixes.pop_front() {
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
                let explicit: bool = self.conn()?.query_row("SELECT EXISTS(SELECT 1 FROM prefix_copy WHERE parent_id=?1 AND child_id=?2 AND explicit_members=1)",params![parent.to_string(),frame.session.to_string()],|r|r.get(0))?;
                if explicit || link.copies_prefix && link.anchor.is_none() {
                    let copied = self
                        .conn()?
                        .prepare_cached(&format!(
                            "WITH canonical_order AS ({}) SELECT DISTINCT m.pk,m.id,c.record_order,ir.id,s.chain,s.seq,s.record_position FROM message m
                             JOIN canonical_order c ON c.pk=m.pk
                             JOIN message_source s ON s.message_pk = m.pk
                             JOIN input_ref ir ON ir.pk=s.input_pk
                             WHERE m.session_pk = ?1 AND m.duplicate_of IS NULL
                               AND EXISTS(SELECT 1 FROM prefix_copy p WHERE p.input_pk=s.input_pk AND p.chain=s.chain AND p.child_id=?2 AND p.parent_id=?3 AND p.explicit_members=0)
                               AND NOT EXISTS(SELECT 1 FROM prefix_copy p WHERE p.input_pk=s.input_pk AND p.chain=s.chain AND p.child_id=?2 AND p.parent_id=?3 AND p.explicit_members=1)
                             ORDER BY m.session_order, m.id",
                             order::canonical_rows("?1")))?
                        .query_map(params![parent_pk, frame.session.to_string(), parent.to_string()], |r| {
                            Ok(CopiedKey {
                                source: r.get(3)?,
                                chain: r.get(4)?,
                                version: String::new(),
                                sequence: r.get(5)?,
                                position: r.get(6)?,
                                key: Key {
                                    session_pk: parent_pk,
                                    session: parent.clone(),
                                    message_pk: r.get(0)?,
                                    id: r.get(1)?,
                                    order: r.get(2)?,
                                },
                            })
                        })?
                        .collect::<rusqlite::Result<Vec<_>>>()?;
                    frame.copied.extend(copied);
                    let missing = self.conn()?.prepare_cached("SELECT DISTINCT p.message_id FROM prefix_member p WHERE p.child_id=?1 AND p.parent_id=?2 AND NOT EXISTS(SELECT 1 FROM message m WHERE m.id=p.message_id AND m.session_pk=?3 AND m.duplicate_of IS NULL) ORDER BY p.message_id")?
                        .query_map(params![frame.session.to_string(),parent.to_string(),parent_pk], |r| r.id::<MessageId>(0))?.collect::<rusqlite::Result<Vec<_>>>()?;
                    frame.issues.extend(missing.into_iter().map(|message| {
                        ParentIssue::MissingCopiedMessage {
                            child: frame.session.clone(),
                            parent: parent.clone(),
                            message,
                        }
                    }));
                    let explicit = self.conn()?.prepare_cached(&format!(
                        "WITH canonical_order AS ({}) SELECT DISTINCT m.pk,m.id,c.record_order,p.source_input,p.source_chain,p.source_version,p.sequence,p.position FROM prefix_member p JOIN message m ON m.id=p.message_id JOIN canonical_order c ON c.pk=m.pk WHERE p.child_id=?2 AND p.parent_id=?3 AND m.session_pk=?1 AND m.duplicate_of IS NULL", order::canonical_rows("?1")))?
                        .query_map(params![parent_pk,frame.session.to_string(),parent.to_string()], |r| {
                            let coordinate = |index| -> rusqlite::Result<u64> {
                                let value: String = r.get(index)?;
                                value.parse().map_err(|error| rusqlite::Error::FromSqlConversionFailure(index, rusqlite::types::Type::Text, Box::new(error)))
                            };
                            Ok(CopiedKey { source:r.get(3)?,chain:r.get(4)?,version:r.get(5)?,sequence:coordinate(6)?,position:coordinate(7)?,key:Key {session_pk:parent_pk,session:parent.clone(),message_pk:r.get(0)?,id:r.get(1)?,order:r.get(2)?} })
                        })?.collect::<rusqlite::Result<Vec<_>>>()?;
                    frame.copied.extend(explicit);
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
            let (copied, issue) = copied_order(&done.session, std::mem::take(&mut done.copied))?;
            done.keys.extend(copied);
            done.issues.extend(issue);
            for (message_pk, id, order) in self.own_keys(done.pk)? {
                done.keys.push(Key {
                    session_pk: done.pk,
                    session: done.session.clone(),
                    message_pk,
                    id,
                    order,
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

/// Each source contributes only its recorded ordering constraints. A unique
/// topological order composes copied families without choosing a source by ID.
fn copied_order(
    child: &SessionId,
    copies: Vec<CopiedKey>,
) -> Result<(Vec<Key>, Option<ParentIssue>)> {
    let mut sources =
        std::collections::BTreeMap::<(String, i64, String), Vec<(u64, u64, i64)>>::new();
    let mut keys = std::collections::BTreeMap::new();
    for copy in copies {
        sources
            .entry((copy.source, copy.chain, copy.version))
            .or_default()
            .push((copy.sequence, copy.position, copy.key.message_pk));
        keys.entry(copy.key.message_pk).or_insert(copy.key);
    }
    let mut edges = std::collections::BTreeMap::<i64, std::collections::BTreeSet<i64>>::new();
    let mut incoming: std::collections::BTreeMap<i64, usize> =
        keys.keys().map(|key| (*key, 0)).collect();
    let message = |key: i64| -> Result<MessageId> {
        MessageId::parse(&keys[&key].id).map_err(|e| StoreError::Invalid(e.to_string()))
    };
    for positions in sources.values_mut() {
        positions.sort_unstable_by_key(|(sequence, position, _)| (*sequence, *position));
        positions.dedup();
        for pair in positions.windows(2) {
            let (left, right) = (pair[0].2, pair[1].2);
            if left != right && (pair[0].0, pair[0].1) == (pair[1].0, pair[1].1) {
                return Ok((
                    Vec::new(),
                    Some(ParentIssue::UnknownOrder {
                        child: child.clone(),
                        first: message(left)?,
                        second: message(right)?,
                    }),
                ));
            }
            if left != right && edges.entry(left).or_default().insert(right) {
                *incoming
                    .get_mut(&right)
                    .expect("every source member has a key") += 1;
            }
        }
    }
    let mut ready: std::collections::BTreeSet<_> = incoming
        .iter()
        .filter(|(_, count)| **count == 0)
        .map(|(key, _)| *key)
        .collect();
    let mut order = Vec::new();
    while !incoming.is_empty() {
        let candidates: Vec<_> = ready.iter().copied().take(2).collect();
        let issue = match candidates.as_slice() {
            [] => {
                let (left, right) = edges
                    .iter()
                    .find_map(|(left, next)| {
                        incoming
                            .contains_key(left)
                            .then(|| {
                                next.iter()
                                    .find(|right| incoming.contains_key(right))
                                    .map(|right| (*left, *right))
                            })
                            .flatten()
                    })
                    .expect("a remaining cycle has an edge");
                Some(ParentIssue::ConflictingOrder {
                    child: child.clone(),
                    first: message(left)?,
                    second: message(right)?,
                })
            }
            [left, right] => Some(ParentIssue::UnknownOrder {
                child: child.clone(),
                first: message(*left)?,
                second: message(*right)?,
            }),
            _ => None,
        };
        if let Some(issue) = issue {
            return Ok((Vec::new(), Some(issue)));
        }
        let key = ready.pop_first().expect("one uniquely ordered member");
        incoming.remove(&key);
        order.push(key);
        for next in edges.get(&key).into_iter().flatten() {
            let count = incoming.get_mut(next).expect("unvisited source successor");
            *count -= 1;
            if *count == 0 {
                ready.insert(*next);
            }
        }
    }
    Ok((
        order
            .into_iter()
            .map(|key| keys.remove(&key).expect("each member emitted once"))
            .collect(),
        None,
    ))
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

    /// Keep hidden messages' metadata positions while leaving their bodies unread.
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

    /// Require the complete requested parent history before using its metadata
    /// as a strict content or evidence binding. No record bodies are loaded.
    pub fn require_complete(&self) -> Result<()> {
        match self.parents.issues.first() {
            Some(issue) => Err(StoreError::IncompleteParents(Box::new(issue.clone()))),
            None => Ok(()),
        }
    }

    pub fn visit(&self, visit: impl FnMut(Message) -> ControlFlow<()>) -> Result<()> {
        self.visit_from(VisitStart::default(), visit).map(drop)
    }

    pub fn visit_with_projection(
        &self,
        visit: impl FnMut(Message, MessageProjection) -> ControlFlow<()>,
    ) -> Result<()> {
        self.visit_from_with_projection(VisitStart::default(), visit)
            .map(drop)
    }

    /// Visit canonical facts, original block positions and their admission in
    /// the same selected row. Raw-source retention does not affect the receipt.
    pub fn visit_with_evidence(
        &self,
        visit: impl FnMut(Message, MessageProjection, Option<u64>) -> ControlFlow<()>,
    ) -> Result<()> {
        self.visit_from_with_evidence(VisitStart::default(), visit)
            .map(drop)
    }

    /// Canonical facts with the full own-session snapshot ordinal, before
    /// inherited prefixes or caller subsets change the visited sequence.
    pub fn visit_with_record_evidence(
        &self,
        visit: impl FnMut(Message, MessageProjection, Option<u64>, u64) -> ControlFlow<()>,
    ) -> Result<()> {
        self.visit_from_with_record_evidence(VisitStart::default(), visit)
            .map(drop)
    }

    pub fn visit_from_with_record_evidence(
        &self,
        start: VisitStart,
        mut visit: impl FnMut(Message, MessageProjection, Option<u64>, u64) -> ControlFlow<()>,
    ) -> Result<Option<i64>> {
        let mut error = None;
        let last = self.visit_from_projected(start, true, |message, projection, evidence| {
            match self.reader.conn().and_then(|conn| evidence.record(conn)) {
                Ok((admitted, order)) => visit(message, projection, admitted, order),
                Err(err) => {
                    error = Some(err);
                    ControlFlow::Break(())
                }
            }
        })?;
        match error {
            Some(error) => Err(error),
            None => Ok(last),
        }
    }

    /// Skip source messages before hydrating their blocks. With the start
    /// message's key, a session's own messages before it are not read
    /// at all. Returns the key of the last message visited.
    pub fn visit_from(
        &self,
        start: VisitStart,
        mut visit: impl FnMut(Message) -> ControlFlow<()>,
    ) -> Result<Option<i64>> {
        self.visit_from_rows(start, false, |message, _, _| visit(message))
    }

    pub fn visit_from_with_projection(
        &self,
        start: VisitStart,
        mut visit: impl FnMut(Message, MessageProjection) -> ControlFlow<()>,
    ) -> Result<Option<i64>> {
        self.visit_from_projected(start, false, |message, projection, _| {
            visit(message, projection)
        })
    }

    pub fn visit_from_with_evidence(
        &self,
        start: VisitStart,
        mut visit: impl FnMut(Message, MessageProjection, Option<u64>) -> ControlFlow<()>,
    ) -> Result<Option<i64>> {
        let mut error = None;
        let last = self.visit_from_projected(start, false, |message, projection, admission| {
            match self.reader.conn().and_then(|conn| admission.resolve(conn)) {
                Ok(admitted) => visit(message, projection, admitted),
                Err(err) => {
                    error = Some(err);
                    ControlFlow::Break(())
                }
            }
        })?;
        match error {
            Some(error) => Err(error),
            None => Ok(last),
        }
    }

    fn visit_from_projected(
        &self,
        start: VisitStart,
        ordered: bool,
        mut visit: impl FnMut(Message, MessageProjection, MessageEvidence) -> ControlFlow<()>,
    ) -> Result<Option<i64>> {
        let mut missing = false;
        let last =
            self.visit_from_rows(
                start,
                ordered,
                |message, projection, admitted| match projection {
                    Some(projection) => visit(message, projection, admitted),
                    None => {
                        missing = true;
                        ControlFlow::Break(())
                    }
                },
            )?;
        if missing {
            return Err(StoreError::Invalid(
                "the message has no canonical source reading".into(),
            ));
        }
        Ok(last)
    }

    fn visit_from_rows(
        &self,
        start: VisitStart,
        ordered: bool,
        mut visit: impl FnMut(Message, Option<MessageProjection>, MessageEvidence) -> ControlFlow<()>,
    ) -> Result<Option<i64>> {
        if let Some(keys) = &self.with_parents {
            let mut last = None;
            for key in keys.iter().skip(start.message) {
                let mut flow = ControlFlow::Continue(());
                let _ = self.reader.visit_messages_where(
                    key.session_pk,
                    &key.session,
                    Some(key.message_pk),
                    MessageContent {
                        order: if ordered {
                            RecordOrder::Known(key.order)
                        } else {
                            RecordOrder::Skip
                        },
                        structured: self.structured,
                        visibility: self.visibility.as_ref(),
                    },
                    |pk| pk == key.message_pk,
                    &mut |pk, message, projection, admitted| {
                        last = Some(pk);
                        flow = visit(message, projection, admitted);
                        ControlFlow::Break(())
                    },
                )?;
                if flow.is_break() {
                    break;
                }
            }
            return Ok(last);
        }
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
            (Some(key), None) => self.visit_kept_projection(Some(key), 0, ordered, |_| true, visit),
            _ => self.visit_kept_projection(None, 0, ordered, after_prefix, visit),
        }
    }

    /// Visits the selected messages `keep` takes, in order, from the own
    /// session's message `from` when given; `keep` sees each key before its
    /// blocks are read.
    fn visit_kept_projection(
        &self,
        from: Option<i64>,
        skip: usize,
        ordered: bool,
        mut keep: impl FnMut(i64) -> bool,
        mut visit: impl FnMut(Message, Option<MessageProjection>, MessageEvidence) -> ControlFlow<()>,
    ) -> Result<Option<i64>> {
        let mut last = None;
        let mut visit = |pk, message, projection, admitted| {
            last = Some(pk);
            visit(message, projection, admitted)
        };
        if let Some(keys) = &self.with_parents {
            let mut start = skip;
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
                        Some(key.message_pk),
                        MessageContent {
                            order: if ordered {
                                RecordOrder::OwnSession
                            } else {
                                RecordOrder::Skip
                            },
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
                    order: if ordered {
                        RecordOrder::OwnSession
                    } else {
                        RecordOrder::Skip
                    },
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

    pub(super) fn keys(&self) -> Result<Vec<i64>> {
        match &self.with_parents {
            Some(keys) => Ok(keys.iter().map(|key| key.message_pk).collect()),
            None => match &self.own {
                Some((pk, _)) => Ok(self
                    .reader
                    .own_keys(*pk)?
                    .into_iter()
                    .map(|(pk, _, _)| pk)
                    .collect()),
                None => Ok(Vec::new()),
            },
        }
    }
}
