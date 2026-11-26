//! Message selection in one pinned read, with incomplete parents kept explicit.
use super::*;
mod navigation;
use std::collections::VecDeque;
use std::ops::ControlFlow;

pub use polylogue_model::read::{ParentIssue, Parents};

struct Key {
    session_pk: SessionPk,
    session: SessionId,
    message_pk: MessagePk,
    id: String,
    order: i64,
}

struct Prefix {
    parent: String,
    copies_prefix: bool,
    branch_point: Option<String>,
}

row! {
    /// One of a session's own messages at its place in the reading order.
    pub(super) struct OwnKey {
        pub(super) pk: MessagePk,
        pub(super) id: String,
        pub(super) record_order: i64,
    }
    statements {
        const OF_SESSION = concat!(
            super::order::record_ranks!(),
            "SELECT m.pk, m.id, c.record_order FROM record_ranks c JOIN message m ON m.pk=c.pk
            ORDER BY c.record_order"
        );
    }
}

row! {
    /// A parent's message that a session's file copied, at its place in the
    /// copying reading.
    struct CopiedMessage {
        pk: MessagePk,
        id: String,
        record_order: i64,
        input: String,
        chain: i64,
        seq: u64,
        record_position: u64,
    }
    statements {
        const OF_COPY = concat!(
            super::order::record_ranks!(),
            "SELECT DISTINCT m.pk, m.id, c.record_order, ir.id AS input, s.chain, s.seq, s.record_position
            FROM message m
            JOIN record_ranks c ON c.pk=m.pk
            JOIN message_parse s ON s.message_id = m.id
            JOIN input_ref ir ON ir.pk=s.input_pk
            WHERE m.session_pk = ?1 AND m.duplicate_of IS NULL
              AND EXISTS(SELECT 1 FROM prefix_copy p WHERE p.input_pk=s.input_pk AND p.chain=s.chain AND p.child_id=?2 AND p.parent_id=?3 AND p.explicit_members=0)
              AND NOT EXISTS(SELECT 1 FROM prefix_copy p WHERE p.input_pk=s.input_pk AND p.chain=s.chain AND p.child_id=?2 AND p.parent_id=?3 AND p.explicit_members=1)
            ORDER BY m.session_order, m.id"
        );
    }
}

row! {
    /// A parent's message that a copy lists by name, at its stated place in
    /// the copying reading.
    struct ListedMessage {
        pk: MessagePk,
        id: String,
        record_order: i64,
        source_input: String,
        source_chain: i64,
        source_version: String,
        sequence: String,
        position: String,
    }
    statements {
        const OF_COPY = concat!(
            super::order::record_ranks!(),
            "SELECT DISTINCT m.pk, m.id, c.record_order, p.source_input, p.source_chain, p.source_version, p.sequence, p.position
            FROM prefix_member p JOIN message m ON m.id=p.message_id JOIN record_ranks c ON c.pk=m.pk
            WHERE p.child_id=?2 AND p.parent_id=?3 AND m.session_pk=?1 AND m.duplicate_of IS NULL"
        );
    }
}

row! {
    /// A message's session and its place there.
    struct MessagePlace {
        session_pk: SessionPk,
        session_order: String,
    }
    statements {
        const OF_MESSAGE = "SELECT session_pk, session_order FROM message WHERE id=?1 AND duplicate_of IS NULL";
    }
}

struct CopiedKey {
    input: String,
    chain: i64,
    version: String,
    sequence: u64,
    position: u64,
    key: Key,
}

struct Frame {
    pk: SessionPk,
    session: SessionId,
    prefixes: VecDeque<Prefix>,
    keys: Vec<Key>,
    copied: Vec<CopiedKey>,
    issues: Vec<ParentIssue>,
    /// The fork whose parent view this frame is constructing.
    via: Option<(SessionId, Option<String>)>,
}

/// A message's place in a selection, and its key when the caller knows it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MessagePosition {
    pub message: usize,
    pub key: Option<MessagePk>,
}

pub(super) struct Visibility {
    pub(super) kinds: String,
    pub(super) prompts: bool,
}

/// Block kinds as the comma-separated SQL list of an `IN (...)` test.
fn block_kind_list(kinds: &[polylogue_model::BlockKind]) -> String {
    kinds
        .iter()
        .map(|kind| format!("'{}'", kind.as_str()))
        .collect::<Vec<_>>()
        .join(",")
}

/// A message selection belongs to the reader that made it. Bodies are read
/// lazily; the parents' history keeps keys for ordering, branch points and deduplication.
pub struct MessageSelection<'r, 'a> {
    reader: &'r Reader<'a>,
    own: Option<(SessionPk, SessionId)>,
    with_parents: Option<Vec<Key>>,
    parents: Parents,
    /// Whether tool results carry their structured data; a text read leaves
    /// it unread ([`Self::for_text`]).
    structured: bool,
    visibility: Option<Visibility>,
}

impl<'a> Reader<'a> {
    /// Parent transcript ranges use the same copied members and order as message reads.
    pub fn parent_parts(
        &self,
        id: &SessionId,
    ) -> Result<(Vec<polylogue_model::read::Part>, Parents, bool)> {
        let Some(pk) = self.session_pk(id)? else {
            return Ok((Vec::new(), Parents { issues: Vec::new() }, false));
        };
        let (keys, issues) = parent_keys(self.conn()?, pk, id)?;
        let mut parts: Vec<polylogue_model::read::Part> = Vec::new();
        for key in keys.into_iter().filter(|key| key.session_pk != pk) {
            let message = MessageId::parse(&key.id)
                .map_err(|error| StoreError::stored_data(error.to_string()))?;
            let Some((start, end)) = self.item_message_span(key.session_pk.0, &message)? else {
                continue;
            };
            if let Some(last) = parts.last_mut()
                && last.session == key.session
                && last.end == start as usize
            {
                last.end = end as usize;
            } else {
                parts.push(polylogue_model::read::Part {
                    session: key.session,
                    start: start as usize,
                    end: end as usize,
                    inherited: true,
                    joined_by: None,
                });
            }
        }
        let mut unstored = false;
        for (parent, _, copied) in
            crate::lineage::inherited_prefixes(self.conn()?, &id.to_string())?
        {
            if copied
                && self
                    .conn()?
                    .value::<i64>("SELECT pk FROM session WHERE id=?1 AND stub=0", [parent])?
                    .is_none()
            {
                unstored = true;
            }
        }
        Ok((parts, Parents { issues }, unstored))
    }

    pub fn select_messages<'r>(
        &'r self,
        id: &SessionId,
        with_parents: bool,
    ) -> Result<MessageSelection<'r, 'a>> {
        let own = self.session_pk(id)?.map(|pk| (pk, id.clone()));
        let has_parents = with_parents
            && !crate::lineage::inherited_prefixes(self.conn()?, &id.to_string())?.is_empty();
        let (keys, issues) = match own.as_ref().filter(|_| has_parents) {
            Some((pk, _)) => parent_keys(self.conn()?, *pk, id)?,
            None => (Vec::new(), Vec::new()),
        };
        self.check()?;
        Ok(MessageSelection {
            reader: self,
            own,
            with_parents: has_parents.then_some(keys),
            parents: Parents { issues },
            structured: true,
            visibility: None,
        })
    }
}

fn parent_frame(
    conn: &Connection,
    pk: SessionPk,
    session: SessionId,
    via: Option<(SessionId, Option<String>)>,
) -> Result<Frame> {
    let mut prefixes: VecDeque<Prefix> =
        crate::lineage::inherited_prefixes(conn, &session.to_string())?
            .into_iter()
            .map(|(parent, branch_point, copies_prefix)| Prefix {
                parent,
                branch_point,
                copies_prefix,
            })
            .collect();
    let forks: std::collections::BTreeSet<_> = prefixes
        .iter()
        .filter_map(|link| {
            Some((
                SessionId::parse(&link.parent).ok()?,
                link.branch_point.as_ref()?.clone(),
            ))
        })
        .collect();
    let mut issues = Vec::new();
    if forks.len() > 1 {
        let mut recorded = forks.into_iter();
        let (first, first_branch_point) = recorded.next().expect("conflicting recorded forks");
        for (second, second_branch_point) in recorded {
            issues.push(ParentIssue::ConflictingParents {
                child: session.clone(),
                first: first.clone(),
                first_branch_point: first_branch_point.clone(),
                second,
                second_branch_point,
            });
        }
        // No branch is selected from incompatible recorded fork points.
        prefixes.retain(|prefix| {
            prefix.branch_point.is_none() || SessionId::parse(&prefix.parent).is_err()
        });
        // Conflicting provider branch points do not erase separately recorded
        // byte copies. Their listed members choose no provider branch_point.
        let copies: Vec<String> = conn.values(
            "SELECT DISTINCT parent_id FROM prefix_copy WHERE child_id=?1 AND explicit_members=1 ORDER BY parent_id",
            [session.to_string()],
        )?;
        for parent in copies {
            if !prefixes
                .iter()
                .any(|prefix| prefix.parent == parent && prefix.branch_point.is_none())
            {
                prefixes.push_back(Prefix {
                    parent,
                    copies_prefix: true,
                    branch_point: None,
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

fn parent_keys(
    conn: &Connection,
    pk: SessionPk,
    id: &SessionId,
) -> Result<(Vec<Key>, Vec<ParentIssue>)> {
    let mut frames = vec![parent_frame(conn, pk, id.clone(), None)?];
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
            let explicit: bool = conn.exists("SELECT EXISTS(SELECT 1 FROM prefix_copy WHERE parent_id=?1 AND child_id=?2 AND explicit_members=1)", params![parent.to_string(),frame.session.to_string()])?;
            if !explicit && !link.copies_prefix && link.branch_point.is_none() {
                frame.issues.push(ParentIssue::MissingBranchPoint {
                    child: frame.session.clone(),
                    parent,
                    branch_point: None,
                });
                continue;
            }
            let parent_pk: Option<SessionPk> = conn.value(
                "SELECT pk FROM session WHERE id=?1 AND stub=0",
                [parent.to_string()],
            )?;
            let Some(parent_pk) = parent_pk else {
                // A copied prefix whose source is not stored leaves the
                // child's messages its own; it is no broken link.
                if !link.copies_prefix {
                    frame.issues.push(ParentIssue::MissingParent {
                        child: frame.session.clone(),
                        parent,
                    });
                }
                continue;
            };
            if active.contains(&parent_pk) {
                frame.issues.push(ParentIssue::Cycle { session: parent });
                continue;
            }

            if explicit || link.copies_prefix && link.branch_point.is_none() {
                let (child_text, parent_text) = (frame.session.to_string(), parent.to_string());
                let params = params![parent_pk, child_text, parent_text];
                let key = |message_pk, id, order| Key {
                    session_pk: parent_pk,
                    session: parent.clone(),
                    message_pk,
                    id,
                    order,
                };
                for copied in conn.all(&CopiedMessage::OF_COPY, params)? {
                    frame.copied.push(CopiedKey {
                        input: copied.input,
                        chain: copied.chain,
                        version: String::new(),
                        sequence: copied.seq,
                        position: copied.record_position,
                        key: key(copied.pk, copied.id, copied.record_order),
                    });
                }
                let missing: Vec<MessageId> = conn.values("SELECT DISTINCT p.message_id FROM prefix_member p WHERE p.child_id=?1 AND p.parent_id=?2 AND NOT EXISTS(SELECT 1 FROM message m WHERE m.id=p.message_id AND m.session_pk=?3 AND m.duplicate_of IS NULL) ORDER BY p.message_id",
                    params![child_text, parent_text, parent_pk])?;
                frame.issues.extend(missing.into_iter().map(|message| {
                    ParentIssue::MissingCopiedMessage {
                        child: frame.session.clone(),
                        parent: parent.clone(),
                        message,
                    }
                }));
                let coordinate = |value: &str| -> Result<u64> {
                    value.parse().map_err(|_| {
                        StoreError::stored_data(format!(
                            "a listed copy's coordinate {value:?} is not a number"
                        ))
                    })
                };
                for listed in conn.all(&ListedMessage::OF_COPY, params)? {
                    frame.copied.push(CopiedKey {
                        input: listed.source_input,
                        chain: listed.source_chain,
                        version: listed.source_version,
                        sequence: coordinate(&listed.sequence)?,
                        position: coordinate(&listed.position)?,
                        key: key(listed.pk, listed.id, listed.record_order),
                    });
                }
            } else {
                let child = frame.session.clone();
                active.insert(parent_pk);
                frames.push(parent_frame(
                    conn,
                    parent_pk,
                    parent,
                    Some((child, link.branch_point)),
                )?);
            }
            continue;
        }
        let mut done = frames.pop().expect("the current frame exists");
        let (copied, issue) = copied_order(&done.session, std::mem::take(&mut done.copied))?;
        done.keys.extend(copied);
        done.issues.extend(issue);
        for own in own_keys(conn, done.pk)? {
            done.keys.push(Key {
                session_pk: done.pk,
                session: done.session.clone(),
                message_pk: own.pk,
                id: own.id,
                order: own.record_order,
            });
        }
        active.remove(&done.pk);
        let Some(child_frame) = frames.last_mut() else {
            let mut seen = HashSet::new();
            done.keys.retain(|key| seen.insert(key.message_pk));
            return Ok((done.keys, done.issues));
        };
        let (child, branch_point) = done.via.expect("only the root has no incoming fork");
        match done
            .keys
            .iter()
            .position(|key| Some(&key.id) == branch_point.as_ref())
        {
            Some(at) => done.keys.truncate(at + 1),
            None => {
                done.keys.clear();
                done.issues.push(ParentIssue::MissingBranchPoint {
                    child,
                    parent: done.session,
                    branch_point,
                });
            }
        }
        // Move the common single-parent prefix rather than copying it at
        // every ancestor of a long chain.
        if child_frame.keys.is_empty() {
            child_frame.keys = done.keys;
        } else {
            child_frame.keys.extend(done.keys);
        }
        child_frame.issues.extend(done.issues);
    }
}

/// A session's own message keys and IDs in reading order.
pub(super) fn own_keys(conn: &Connection, session: SessionPk) -> Result<Vec<OwnKey>> {
    conn.all(&OwnKey::OF_SESSION, [session])
}

/// Compare messages in the same target view used by reading. Own-session
/// coordinates are indexed; inherited items use that target lineage's metadata
/// order, including branch truncation and copied-prefix ambiguity. The fallback
/// costs O(messages in that lineage), and never loads message bodies.
pub(crate) fn compare_session_messages(
    conn: &Connection,
    target: &SessionId,
    left: &MessageId,
    right: &MessageId,
) -> Result<std::cmp::Ordering> {
    let target_pk: SessionPk = conn
        .scalar_optional(
            "SELECT pk FROM session WHERE id=?1 AND stub=0",
            [target.to_string()],
        )?
        .ok_or_else(|| StoreError::NotFound(format!("session:{target}")))?;
    let coordinate = |id: &MessageId| -> Result<(String, SessionPk, String)> {
        let id = crate::chain_names::resolved_message(conn, id)?.to_string();
        let place = conn
            .maybe(&MessagePlace::OF_MESSAGE, [&id])?
            .ok_or_else(|| StoreError::NotFound(format!("message:{id}")))?;
        Ok((id, place.session_pk, place.session_order))
    };
    let left = coordinate(left)?;
    let right = coordinate(right)?;
    if left.1 == target_pk && right.1 == target_pk {
        return Ok((&left.2, &left.0).cmp(&(&right.2, &right.0)));
    }
    if crate::lineage::inherited_prefixes(conn, &target.to_string())?.is_empty() {
        return Err(polylogue_model::Reason::NotInSession {
            item: "the seen items".into(),
            session: format!("session:{target}"),
        }
        .into());
    }
    let (keys, issues) = parent_keys(conn, target_pk, target)?;
    let locate = |id: &str| -> Result<usize> {
        keys.iter().position(|key| key.id == id).ok_or_else(|| {
            issues.first().map_or_else(
                || {
                    StoreError::from(polylogue_model::Reason::NotInSession {
                        item: format!("message:{id}"),
                        session: format!("session:{target}"),
                    })
                },
                |issue| StoreError::IncompleteParents(Box::new(issue.clone())),
            )
        })
    };
    Ok(locate(&left.0)?.cmp(&locate(&right.0)?))
}

/// Each source contributes only its recorded ordering constraints. A unique
/// topological order composes copied families without choosing a source by ID.
fn copied_order(
    child: &SessionId,
    copies: Vec<CopiedKey>,
) -> Result<(Vec<Key>, Option<ParentIssue>)> {
    let mut readings =
        std::collections::BTreeMap::<(String, i64, String), Vec<(u64, u64, MessagePk)>>::new();
    let mut keys = std::collections::BTreeMap::new();
    for copy in copies {
        readings
            .entry((copy.input, copy.chain, copy.version))
            .or_default()
            .push((copy.sequence, copy.position, copy.key.message_pk));
        keys.entry(copy.key.message_pk).or_insert(copy.key);
    }
    let mut edges =
        std::collections::BTreeMap::<MessagePk, std::collections::BTreeSet<MessagePk>>::new();
    let mut incoming: std::collections::BTreeMap<MessagePk, usize> =
        keys.keys().map(|key| (*key, 0)).collect();
    let message = |key: MessagePk| -> Result<MessageId> {
        MessageId::parse(&keys[&key].id).map_err(|error| StoreError::stored(0, error.to_string()))
    };
    for positions in readings.values_mut() {
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
            kinds: block_kind_list(kinds),
            prompts,
        });
        self
    }

    pub fn parents(&self) -> &Parents {
        &self.parents
    }

    /// Require the complete requested parent history before using its metadata
    /// as a strict content or source binding. No record bodies are loaded.
    pub fn require_complete(&self) -> Result<()> {
        match self.parents.issues.first() {
            Some(issue) => Err(StoreError::IncompleteParents(Box::new(issue.clone()))),
            None => Ok(()),
        }
    }

    pub fn visit_with_sources(
        &self,
        start: MessagePosition,
        mut visit: impl FnMut(Message, MessageParse) -> ControlFlow<()>,
    ) -> Result<Option<MessagePk>> {
        let mut error = None;
        let last =
            self.visit_from_selected_blocks(start, true, |message, block_selection, arrival| {
                match self.reader.conn().and_then(|conn| arrival.record(conn)) {
                    Ok((committed, ordinal)) => visit(
                        message,
                        MessageParse {
                            blocks: block_selection,
                            committed,
                            ordinal,
                        },
                    ),
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
    pub fn visit(
        &self,
        start: MessagePosition,
        mut visit: impl FnMut(Message) -> ControlFlow<()>,
    ) -> Result<Option<MessagePk>> {
        self.visit_from_rows(start, false, |message, _, _| visit(message))
    }

    fn visit_from_selected_blocks(
        &self,
        start: MessagePosition,
        ordered: bool,
        mut visit: impl FnMut(Message, ShownBlocks, MessageArrival) -> ControlFlow<()>,
    ) -> Result<Option<MessagePk>> {
        let mut missing = false;
        let last =
            self.visit_from_rows(start, ordered, |message, block_selection, committed| {
                match block_selection {
                    Some(block_selection) => visit(message, block_selection, committed),
                    None => {
                        missing = true;
                        ControlFlow::Break(())
                    }
                }
            })?;
        if missing {
            return Err(StoreError::stored_data("the message has no source reading"));
        }
        Ok(last)
    }

    fn visit_from_rows(
        &self,
        start: MessagePosition,
        ordered: bool,
        mut visit: impl FnMut(Message, Option<ShownBlocks>, MessageArrival) -> ControlFlow<()>,
    ) -> Result<Option<MessagePk>> {
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
                    &mut |pk, message, block_selection, committed| {
                        last = Some(pk);
                        flow = visit(message, block_selection, committed);
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
            (Some(key), None) => {
                self.visit_kept_block_selection(Some(key), 0, ordered, |_| true, visit)
            }
            _ => self.visit_kept_block_selection(None, 0, ordered, after_prefix, visit),
        }
    }

    /// Visits the selected messages `keep` takes, in order, from the own
    /// session's message `from` when given; `keep` sees each key before its
    /// blocks are read.
    fn visit_kept_block_selection(
        &self,
        from: Option<MessagePk>,
        skip: usize,
        ordered: bool,
        mut keep: impl FnMut(MessagePk) -> bool,
        mut visit: impl FnMut(Message, Option<ShownBlocks>, MessageArrival) -> ControlFlow<()>,
    ) -> Result<Option<MessagePk>> {
        let mut last = None;
        let mut visit = |pk, message, block_selection, committed| {
            last = Some(pk);
            visit(message, block_selection, committed)
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
    fn sessions(&self) -> Vec<(SessionPk, Option<Vec<MessagePk>>)> {
        match (&self.with_parents, &self.own) {
            (Some(keys), _) => {
                let mut sessions: Vec<(SessionPk, Option<Vec<MessagePk>>)> = Vec::new();
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

    pub(super) fn keys(&self) -> Result<Vec<MessagePk>> {
        match &self.with_parents {
            Some(keys) => Ok(keys.iter().map(|key| key.message_pk).collect()),
            None => match &self.own {
                Some((pk, _)) => Ok(self
                    .reader
                    .own_keys(*pk)?
                    .into_iter()
                    .map(|own| own.pk)
                    .collect()),
                None => Ok(Vec::new()),
            },
        }
    }
}

/// The rows this module declares statements for, for the every-statement test.
#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<OwnKey>();
    check.row::<CopiedMessage>();
    check.row::<ListedMessage>();
    check.row::<MessagePlace>();
}
