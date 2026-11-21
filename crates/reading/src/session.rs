//! Paged session reads over the session item index. A page is a run of
//! shown entries read in windows from its first entry; where it sits among the
//! shown entries, and the counts a reading shows, come from the counts the
//! writer stores per entry class, so a page reads its own entries and the
//! bodies of its messages and events, not the entries before it. A
//! continuation names its first entry by key and binds the rewrite version
//! of every composed session, so appends never stale it and rewrites always
//! do. Nothing is written.
use super::ReadRequest;
use super::records::{SelectedRecord, message_record};
use super::{layout_facts, output};
use crate::{Error, ReadServices};
use polylogue_model::{MessageId, Reference, SessionId, bundle::Record, read::*};
use polylogue_store::Reader;
use polylogue_store::items::{ItemLookup, ItemVersions};
use polylogue_store::read::limit::ReadLimit;
use std::collections::HashSet;
use std::ops::ControlFlow;
use std::sync::Arc;

mod entries;
use entries::{PageEntry, ShownEntries};

/// What a session read covers.
pub(super) struct Scope {
    pub session: SessionId,
    /// For a message or message range: the first and last message read.
    pub range: Option<(MessageId, MessageId)>,
    /// `root:SESSION`: the logical session the root names.
    pub logical: bool,
}

/// The session read a reference names, when it names one. `root:REF` names
/// the logical session whose root is REF.
pub(super) fn scope(reader: &Reader<'_>, target: &str) -> Result<Option<Scope>, Error> {
    if let Some(root) = target.strip_prefix("root:") {
        let reference = Reference::parse(&format!("session:{root}")).map_err(Error::bad_request)?;
        let Reference::Session(session) = reference else {
            return Err(Error::bad_request(
                "root: names a session, the root of a logical session",
            ));
        };
        return Ok(Some(Scope {
            session,
            range: None,
            logical: true,
        }));
    }
    let Ok(reference) = Reference::parse(target) else {
        return Ok(None);
    };
    let session_of = |message: &MessageId| -> Result<SessionId, Error> {
        reader
            .message_session(message)?
            .ok_or_else(|| Error::not_found(format!("no message {message}")))
    };
    Ok(match reference {
        Reference::Session(session) => Some(Scope {
            session,
            range: None,
            logical: false,
        }),
        Reference::Message(message) => Some(Scope {
            session: session_of(&message)?,
            range: Some((message.clone(), message)),
            logical: false,
        }),
        Reference::MessageRange { first, last } => {
            let session = session_of(&first)?;
            if session_of(&last)? != session {
                return Err(Error::bad_request(
                    "a message range is read inside one session",
                ));
            }
            Some(Scope {
                session,
                range: Some((first, last)),
                logical: false,
            })
        }
        _ => None,
    })
}

/// What a reading needs of one composed session before reading its
/// entries: where its index is, its counters, and its counts.
pub(super) struct SessionIndex {
    pub pk: i64,
    pub versions: ItemVersions,
    pub entries: u64,
    pub counts: ItemCounts,
}

/// A session's index, or `None` when the session is not stored.
fn session_index(
    reader: &Reader<'_>,
    session: &SessionId,
) -> Result<Option<Arc<SessionIndex>>, Error> {
    Ok(reader.item_state(session)?.map(|state| {
        Arc::new(SessionIndex {
            pk: state.session_pk,
            versions: state.versions,
            entries: state.entries,
            counts: state.counts,
        })
    }))
}

/// Cards shown after one entry: a fork at its branch point, a spawned
/// session at its call.
struct CardsAfterEntry {
    session: SessionId,
    seq: usize,
    cards: Vec<ReadItem>,
}

/// One composed session reading: its parts, each with the index of its
/// session, and the link problems found while composing.
struct Composed {
    parts: Vec<Part>,
    indexes: Vec<Arc<SessionIndex>>,
    issues: Vec<ParentIssue>,
    /// Lines shown before an entry (a `continues` line at a continuation).
    before: std::collections::HashMap<EntryKey, Vec<ReadItem>>,
    after: Vec<CardsAfterEntry>,
}
/// A part that reads a spawned session inline.
fn spawned(part: &Part) -> bool {
    part.joined_by.as_deref().is_some_and(|how| how == "spawn")
}

impl Composed {
    /// The counts of a part's session before `seq`.
    fn counts_at(&self, reader: &Reader<'_>, part: usize, seq: usize) -> Result<ItemCounts, Error> {
        entries::counts_before(reader, &self.indexes[part], seq)
    }
    /// The entries of a part's session in `start..end` whose class `which`
    /// accepts.
    fn count(
        &self,
        reader: &Reader<'_>,
        part: usize,
        (start, end): (usize, usize),
        which: impl Fn(EntryClass) -> bool,
    ) -> Result<u64, Error> {
        if start >= end {
            return Ok(0);
        }
        Ok(self
            .counts_at(reader, part, end)?
            .since(&self.counts_at(reader, part, start)?)
            .count(which))
    }
    /// The entry of a part's session at `seq`.
    fn entry(
        &self,
        reader: &Reader<'_>,
        part: usize,
        seq: usize,
    ) -> Result<Option<IndexEntry>, Error> {
        Ok(reader
            .item_window(
                self.indexes[part].pk,
                seq as u64..seq as u64 + 1,
                true,
                1,
                |_| true,
            )?
            .pop()
            .map(|stored| stored.entry))
    }
    /// Which parts share a context with `part`: an inline spawned session
    /// has its own; the reading's other parts share one.
    fn same_context(&self, part: usize) -> impl Fn(&Part) -> bool + '_ {
        let own = &self.parts[part];
        move |other: &Part| {
            if spawned(own) {
                other.session == own.session
            } else {
                !spawned(other)
            }
        }
    }
    /// The parts with every part outside `part`'s context made empty, so
    /// places keep their part numbers.
    fn context_parts(&self, part: usize) -> Vec<Part> {
        let same = self.same_context(part);
        self.parts
            .iter()
            .map(|other| {
                if same(other) {
                    other.clone()
                } else {
                    Part {
                        end: other.start,
                        ..other.clone()
                    }
                }
            })
            .collect()
    }
    fn versions(&self, reader: &Reader<'_>) -> Result<Vec<SessionVersion>, Error> {
        let mut seen = HashSet::new();
        self.parts
            .iter()
            .zip(&self.indexes)
            .filter(|(part, _)| seen.insert(part.session.clone()))
            .map(|(part, index)| {
                Ok(SessionVersion {
                    session: part.session.clone(),
                    rewrite_version: index.versions.rewrite_version,
                    attachments: reader.session_attachment_version(&part.session)?,
                })
            })
            .collect()
    }
}

/// A session's call entry with a provider call ID, and where it sits.
fn spawning_call(
    reader: &Reader<'_>,
    index: &SessionIndex,
    call: &str,
) -> Result<Option<(usize, IndexEntry)>, Error> {
    Ok(reader
        .item_calls(index.pk, call)?
        .into_iter()
        .find(|stored| stored.entry.kind == EntryKind::Call)
        .map(|stored| (stored.seq as usize, stored.entry)))
}

/// The parent prefix of `session` through its branch point, as parts of
/// parent sessions, all marked inherited. The walk is iterative over
/// sessions already on the path; problems are reported, never resolved by
/// choosing one link.
fn prefix(reader: &Reader<'_>, session: &SessionId, composed: &mut Composed) -> Result<(), Error> {
    let mut chain: Vec<(SessionId, Option<String>)> = Vec::new();
    let mut on_path: HashSet<SessionId> = HashSet::from([session.clone()]);
    let mut current = session.clone();
    loop {
        let links = reader.inherited_prefixes(&current)?;
        let forks: std::collections::BTreeSet<(String, String)> = links
            .iter()
            .filter_map(|link| Some((link.parent.clone(), link.branch_point.clone()?)))
            .collect();
        if forks.len() > 1 {
            let mut recorded = forks.into_iter();
            let (first, first_branch_point) = recorded.next().expect("two forks");
            let (second, second_branch_point) = recorded.next().expect("two forks");
            if let (Ok(first), Ok(second)) = (SessionId::parse(&first), SessionId::parse(&second)) {
                composed.issues.push(ParentIssue::ConflictingParents {
                    child: current.clone(),
                    first,
                    first_branch_point,
                    second,
                    second_branch_point,
                });
            }
            break;
        }
        // More than one composing parent (several resumes, or a fork and a
        // resume) is reported, never resolved by choosing one.
        let parents: std::collections::BTreeSet<(&str, &str)> = links
            .iter()
            .map(|link| {
                (
                    link.parent.as_str(),
                    link.branch_point.as_deref().unwrap_or(""),
                )
            })
            .collect();
        if parents.len() > 1 {
            let mut recorded = parents.into_iter();
            let (first, first_branch_point) = recorded.next().expect("two parents");
            let (second, second_branch_point) = recorded.next().expect("two parents");
            if let (Ok(first), Ok(second)) = (SessionId::parse(first), SessionId::parse(second)) {
                composed.issues.push(ParentIssue::ConflictingParents {
                    child: current.clone(),
                    first,
                    first_branch_point: first_branch_point.to_owned(),
                    second,
                    second_branch_point: second_branch_point.to_owned(),
                });
            }
            break;
        }
        let Some(link) = links.first() else { break };
        let Ok(parent) = SessionId::parse(&link.parent) else {
            composed.issues.push(ParentIssue::InvalidParent {
                child: current.clone(),
                parent: link.parent.clone(),
            });
            break;
        };
        if !on_path.insert(parent.clone()) {
            composed.issues.push(ParentIssue::Cycle { session: parent });
            break;
        }
        if !link.copies_prefix && link.branch_point.is_none() {
            composed.issues.push(ParentIssue::MissingBranchPoint {
                child: current.clone(),
                parent,
                branch_point: None,
            });
            break;
        }
        chain.push((parent.clone(), link.branch_point.clone()));
        current = parent;
    }
    // Compose from the oldest ancestor down, cutting each at the branch
    // point its child continues from.
    let mut parts: Vec<(Part, Arc<SessionIndex>)> = Vec::new();
    for (at, (parent, branch_point)) in chain.iter().enumerate().rev() {
        let child = if at == 0 {
            session.clone()
        } else {
            chain[at - 1].0.clone()
        };
        let Some(index) = session_index(reader, parent)? else {
            composed.issues.push(ParentIssue::MissingParent {
                child,
                parent: parent.clone(),
            });
            parts.clear();
            continue;
        };
        parts.push((
            Part {
                session: parent.clone(),
                start: 0,
                end: index.entries as usize,
                inherited: true,
                joined_by: None,
            },
            index,
        ));
        if let Some(point) = branch_point {
            let mut cut = None;
            if let Ok(message) = MessageId::parse(point) {
                for (which, (part, index)) in parts.iter().enumerate().rev() {
                    if let Some((header, end)) = reader.item_message_span(index.pk, &message)?
                        && (part.start..part.end).contains(&(header as usize))
                    {
                        cut = Some((which, (end as usize).min(part.end)));
                        break;
                    }
                }
            }
            match cut {
                Some((which, end)) => {
                    parts.truncate(which + 1);
                    parts[which].0.end = end;
                }
                None => {
                    composed.issues.push(ParentIssue::MissingBranchPoint {
                        child,
                        parent: parent.clone(),
                        branch_point: Some(point.clone()),
                    });
                    parts.clear();
                }
            }
        }
    }
    for (part, index) in parts {
        composed.parts.push(part);
        composed.indexes.push(index);
    }
    Ok(())
}

fn compose(
    d: &impl ReadServices,
    reader: &Reader<'_>,
    scope: &Scope,
    options: &ReadOptions,
) -> Result<Composed, Error> {
    let around = options.around.unwrap_or(0) as usize;
    let own = session_index(reader, &scope.session)?
        .ok_or_else(|| Error::not_found(format!("no session {}", scope.session)))?;
    let mut composed = Composed {
        parts: Vec::new(),
        indexes: Vec::new(),
        issues: Vec::new(),
        before: Default::default(),
        after: Vec::new(),
    };
    // The prefix is composed always: hidden, it is one `continues` line.
    if scope.range.is_none() {
        prefix(reader, &scope.session, &mut composed)?;
    }
    let (start, end) = match &scope.range {
        None => (0, own.entries as usize),
        Some((first, last)) => {
            let header = |message: &MessageId| -> Result<usize, Error> {
                reader
                    .item_place(own.pk, &EntryKey::header(message))?
                    .map(|seq| seq as usize)
                    .ok_or_else(|| {
                        Error::not_found(format!("message {message} is not in the reading"))
                    })
            };
            let start = header(first)?;
            let last_at = header(last)?;
            if last_at < start {
                return Err(Error::bad_request("a message range ends before it starts"));
            }
            // `--around N`: N messages either side, as far as there are.
            let (start, last_at) = if around > 0 {
                let headers = |range: std::ops::Range<u64>, forward: bool| {
                    reader.item_window(own.pk, range, forward, around, EntryClass::is_header)
                };
                let earlier = headers(0..start as u64, false)?;
                let later = headers(last_at as u64 + 1..own.entries, true)?;
                (
                    earlier.last().map_or(start, |stored| stored.seq as usize),
                    later.last().map_or(last_at, |stored| stored.seq as usize),
                )
            } else {
                (start, last_at)
            };
            (
                start,
                reader.item_unit_end(own.pk, last_at as u64)? as usize,
            )
        }
    };
    composed.parts.push(Part {
        session: scope.session.clone(),
        start,
        end,
        inherited: false,
        joined_by: None,
    });
    composed.indexes.push(own);
    children(d, reader, scope, &mut composed, options.view)?;
    if let SpawnFold::Inline { depth } = options.spawns {
        inline_spawns(d, reader, &mut composed, depth, options.view)?;
    }
    Ok(composed)
}

/// Continuations and resumes follow their parent in a logical reading,
/// each after a `continues` line; forks and spawned sessions appear as cards
/// at their branch point or spawning call. The walk is iterative and visits
/// each session once.
fn children(
    d: &impl ReadServices,
    reader: &Reader<'_>,
    scope: &Scope,
    composed: &mut Composed,
    view: View,
) -> Result<(), Error> {
    let mut visited: HashSet<SessionId> = HashSet::from([scope.session.clone()]);
    let mut at = composed.parts.len() - 1;
    while at < composed.parts.len() {
        let session = composed.parts[at].session.clone();
        let index = Arc::clone(&composed.indexes[at]);
        at += 1;
        for link in reader.child_links(&session)? {
            still_wanted()?;
            if link.kind == "reference" {
                continue;
            }
            let Ok(child) = SessionId::parse(&link.child) else {
                continue;
            };
            let joins = matches!(link.kind.as_str(), "continuation" | "resume");
            if joins && scope.logical {
                if !visited.insert(child.clone()) {
                    composed.issues.push(ParentIssue::Cycle { session: child });
                    continue;
                }
                let Some(child_index) = session_index(reader, &child)? else {
                    composed.issues.push(ParentIssue::MissingParent {
                        child,
                        parent: session.clone(),
                    });
                    continue;
                };
                let first = reader
                    .item_window(child_index.pk, 0..child_index.entries, true, 1, |_| true)?
                    .pop();
                if let Some(first) = first {
                    composed
                        .before
                        .entry(first.entry.key)
                        .or_default()
                        .push(ReadItem {
                            source: Source::record(format!("session:{child}")),
                            kind: ItemKind::Indicator {
                                indicator: IndicatorKind::Continues,
                                text: format!("continues {session} ({})", link.kind),
                                expand: None,
                            },
                        });
                }
                composed.parts.push(Part {
                    session: child,
                    start: 0,
                    end: child_index.entries as usize,
                    inherited: false,
                    joined_by: Some(link.kind.clone()),
                });
                composed.indexes.push(child_index);
                continue;
            }
            if joins {
                continue;
            }
            // A card at the branch point or the spawning call.
            let (slot, call) = match link.kind.as_str() {
                "spawn" => match link.spawning_call.as_deref() {
                    Some(call) => match spawning_call(reader, &index, call)? {
                        Some((seq, entry)) => (Some(seq), Some(entry)),
                        None => (None, None),
                    },
                    None => (None, None),
                },
                _ => {
                    let point = link
                        .branch_point
                        .as_deref()
                        .and_then(|point| MessageId::parse(point).ok());
                    let end = match point {
                        Some(point) => reader.item_message_span(index.pk, &point)?,
                        None => None,
                    };
                    (
                        end.and_then(|(_, end)| end.checked_sub(1))
                            .map(|seq| seq as usize),
                        None,
                    )
                }
            };
            let Some(slot) = slot else { continue };
            let card = card(d, reader, &link, &child, call.as_ref(), view)?;
            composed.after.push(CardsAfterEntry {
                session: session.clone(),
                seq: slot,
                cards: vec![card],
            });
        }
    }
    Ok(())
}

/// Spawned sessions inline (`--show spawns:full` or `spawns:N`): each
/// spawned child's reading follows its card at the spawning call, by
/// splitting the spawning part there. Nested spawns are expanded to `depth`
/// levels and are cards below it; a session already in the reading stays a
/// card, so the walk ends.
fn inline_spawns(
    d: &impl ReadServices,
    reader: &Reader<'_>,
    composed: &mut Composed,
    depth: Option<u32>,
    view: View,
) -> Result<(), Error> {
    let deepest = depth.unwrap_or(u32::MAX);
    let mut levels: Vec<u32> = vec![0; composed.parts.len()];
    // Parts whose spawns are already placed: the pieces of a split part.
    let mut done: Vec<bool> = vec![false; composed.parts.len()];
    let mut seen: HashSet<SessionId> = composed
        .parts
        .iter()
        .map(|part| part.session.clone())
        .collect();
    let mut at = 0;
    while at < composed.parts.len() {
        if done[at] || composed.parts[at].inherited {
            at += 1;
            continue;
        }
        let level = levels[at];
        let part = composed.parts[at].clone();
        let index = Arc::clone(&composed.indexes[at]);
        let mut spawned = Vec::new();
        for link in reader.child_links(&part.session)? {
            still_wanted()?;
            if link.kind != "spawn" {
                continue;
            }
            let Ok(child) = SessionId::parse(&link.child) else {
                continue;
            };
            let call = match link.spawning_call.as_deref() {
                Some(call) => spawning_call(reader, &index, call)?,
                None => None,
            };
            let Some((entry, call)) = call.filter(|(seq, _)| (part.start..part.end).contains(seq))
            else {
                continue;
            };
            spawned.push((entry, call, child, link));
        }
        spawned.sort_by_key(|(entry, _, _, _)| *entry);
        // (part, its index, its spawn level, whether its spawns are placed)
        let mut pieces: Vec<(Part, Arc<SessionIndex>, u32, bool)> = Vec::new();
        let mut cut = part.start;
        for (entry, call, child, link) in spawned {
            still_wanted()?;
            // The first level's cards come from composition; deeper levels
            // place their own.
            if level > 0 {
                let card = card(d, reader, &link, &child, Some(&call), view)?;
                composed.after.push(CardsAfterEntry {
                    session: part.session.clone(),
                    seq: entry,
                    cards: vec![card],
                });
            }
            if level >= deepest || !seen.insert(child.clone()) {
                continue;
            }
            let Some(child_index) = session_index(reader, &child)? else {
                continue;
            };
            pieces.push((
                Part {
                    start: cut,
                    end: entry + 1,
                    ..part.clone()
                },
                Arc::clone(&index),
                level,
                true,
            ));
            let end = child_index.entries as usize;
            pieces.push((
                Part {
                    session: child,
                    start: 0,
                    end,
                    inherited: false,
                    joined_by: Some(link.kind.clone()),
                },
                child_index,
                level + 1,
                false,
            ));
            cut = entry + 1;
        }
        if pieces.is_empty() {
            done[at] = true;
            at += 1;
            continue;
        }
        if cut < part.end {
            pieces.push((
                Part {
                    start: cut,
                    ..part.clone()
                },
                Arc::clone(&index),
                level,
                true,
            ));
        }
        let mut new_parts = Vec::new();
        let mut new_indexes = Vec::new();
        let mut new_levels = Vec::new();
        let mut new_done = Vec::new();
        for (piece, index, level, placed) in pieces {
            new_parts.push(piece);
            new_indexes.push(index);
            new_levels.push(level);
            new_done.push(placed);
        }
        composed.parts.splice(at..=at, new_parts);
        composed.indexes.splice(at..=at, new_indexes);
        levels.splice(at..=at, new_levels);
        done.splice(at..=at, new_done);
        at += 1;
    }
    Ok(())
}

/// The logical header of `root:REF`: the root, the physical sessions the
/// reading joins in order, and its own and inherited messages. Spawned
/// sessions read inline are not members.
fn logical_header(
    reader: &Reader<'_>,
    root: &SessionId,
    composed: &Composed,
) -> Result<HeaderKind, Error> {
    let messages = |inherited: bool| -> Result<u64, Error> {
        let mut total = 0;
        for (at, part) in composed.parts.iter().enumerate() {
            if part.inherited == inherited && !spawned(part) {
                total +=
                    composed.count(reader, at, (part.start, part.end), EntryClass::is_header)?;
            }
        }
        Ok(total)
    };
    let mut physical: Vec<SessionId> = Vec::new();
    for part in &composed.parts {
        if !part.inherited && !spawned(part) && !physical.contains(&part.session) {
            physical.push(part.session.clone());
        }
    }
    Ok(HeaderKind::Logical {
        root: root.clone(),
        physical_sessions: physical,
        own_messages: ValueState::known(messages(false)?),
        inherited_messages: ValueState::known(messages(true)?),
    })
}

/// The card for a fork or a spawned session: a delegation card for a
/// spawn, a session card for a fork, each with the call that reads it in
/// the reading's view.
fn card(
    d: &impl ReadServices,
    reader: &Reader<'_>,
    link: &polylogue_store::lineage::SessionLink,
    child: &SessionId,
    call: Option<&IndexEntry>,
    view: View,
) -> Result<ReadItem, Error> {
    let reference = format!("session:{child}");
    let open = d.continuation(
        &ReadRequest {
            references: None,
            target: reference.clone(),
            options: ReadOptions {
                view,
                ..ReadOptions::default()
            },
            delivery: ReadDelivery::Page,
            format: Format::Json.name().into(),
            budget: None,
            with_blobs: false,
            cursor: None,
        },
        None,
        command_surface::Deadline::Default,
    )?;
    let kind = if link.kind == "spawn" {
        let objective = call
            .and_then(|entry| entry.target.clone())
            .map_or(ValueState::NotRecorded, ValueState::known);
        Card::Delegation {
            card: Box::new(DelegationCard {
                child: reference.clone(),
                objective,
                result: ValueState::Unknown {
                    reason: Some(format!("read {reference}")),
                },
                duration_ms: ValueState::Unknown { reason: None },
                cost: ValueState::Unknown { reason: None },
                retries: Vec::new(),
                result_use: ValueState::Unknown { reason: None },
                open: Some(open),
            }),
        }
    } else {
        match reader.session(child)? {
            Some(session) => Card::Session {
                assertions: Vec::new(),
                session: Box::new(session),
                lineage_complete: ValueState::NotApplicable,
                open: Some(Box::new(open)),
            },
            None => Card::Fields {
                title: format!("{} {reference} (not in the archive)", link.kind),
                figures: Vec::new(),
            },
        }
    };
    Ok(ReadItem {
        source: Source::record(reference),
        kind: ItemKind::Card {
            card: Box::new(kind),
        },
    })
}

/// The runs of entries a reading shows, as segments over the parts.
struct Runs {
    /// (part, first entry, end entry) for each segment.
    spans: Vec<(usize, usize, usize)>,
    evicted: Vec<EntryKey>,
    context: Option<(bool, Option<String>)>,
    /// A context difference: `evicted` is what left between the two places
    /// and the spans are what entered.
    left: bool,
}

fn runs(reader: &Reader<'_>, composed: &Composed, options: &ReadOptions) -> Result<Runs, Error> {
    if options.view != View::Context || composed.parts.is_empty() {
        return Ok(Runs {
            spans: composed
                .parts
                .iter()
                .enumerate()
                .filter(|(_, part)| options.inherited || !part.inherited)
                .map(|(at, part)| (at, part.start, part.end))
                .collect(),
            evicted: Vec::new(),
            context: None,
            left: false,
        });
    }
    // The context view shows what the model had up to its place: it reads
    // the composed entries up to there.
    let loaded: Vec<Vec<IndexEntry>> = composed
        .parts
        .iter()
        .zip(&composed.indexes)
        .map(|(part, index)| {
            Ok(reader
                .item_window(index.pk, 0..part.end as u64, true, usize::MAX, |_| true)?
                .into_iter()
                .map(|stored| stored.entry)
                .collect())
        })
        .collect::<Result<_, Error>>()?;
    let entries = |part: usize| loaded[part].as_slice();
    let keys = |places: &[Place]| -> Vec<EntryKey> {
        places
            .iter()
            .map(|place| loaded[place.part][place.entry].key.clone())
            .collect()
    };
    let last = composed.parts.len() - 1;
    let latest = Place {
        part: last,
        entry: composed.parts[last].end.saturating_sub(1),
    };
    let place_of = |message: &MessageId| -> Result<Place, Error> {
        find(&composed.parts, entries, &EntryKey::header(message))
            .ok_or_else(|| Error::not_found(format!("message {message} is not in the reading")))
    };
    let at = match options.context_message().map_err(Error::bad_request)? {
        Some(message) => place_of(&message)?,
        None => latest,
    };
    let context = context_at(&composed.parts, entries, at);
    let Some(diff) = &options.diff else {
        return Ok(Runs {
            spans: spans(&context.members),
            evicted: keys(&context.evicted),
            context: Some((context.complete, context.reason)),
            left: false,
        });
    };
    // The other place: a message, or just before the Nth-last boundary.
    let other = if let Some(rest) = diff.strip_prefix("compaction") {
        let back: usize = match rest.strip_prefix(':') {
            Some(n) => n
                .parse()
                .map_err(|_| Error::bad_request("compaction:N takes a count"))?,
            None if rest.is_empty() => 1,
            None => {
                return Err(Error::bad_request("diff takes a message or compaction[:N]"));
            }
        };
        let mut boundaries = Vec::new();
        for (part_at, part) in composed.parts.iter().enumerate() {
            for entry in part.start..part.end {
                let place = Place {
                    part: part_at,
                    entry,
                };
                if place <= at && entries(part_at)[entry].boundary == Some(Boundary::Compaction) {
                    boundaries.push(place);
                }
            }
        }
        let boundary = back
            .checked_sub(1)
            .and_then(|from_end| boundaries.iter().rev().nth(from_end))
            .ok_or_else(|| Error::not_found("the reading has no such compaction"))?;
        if boundary.entry == composed.parts[boundary.part].start {
            return Err(Error::not_found("the compaction is the first entry"));
        }
        Place {
            part: boundary.part,
            entry: boundary.entry - 1,
        }
    } else {
        place_of(&MessageId::parse(diff).map_err(Error::bad_request)?)?
    };
    let (earlier, later) = if other < at {
        (context_at(&composed.parts, entries, other), context)
    } else {
        let later = context_at(&composed.parts, entries, other);
        (context, later)
    };
    let difference = context_difference(&earlier, &later);
    Ok(Runs {
        spans: spans(&difference.entered),
        evicted: keys(&difference.left),
        context: Some((later.complete, later.reason)),
        left: true,
    })
}

/// Places as runs of consecutive entries of one part.
fn spans(places: &[Place]) -> Vec<(usize, usize, usize)> {
    let mut spans: Vec<(usize, usize, usize)> = Vec::new();
    for place in places {
        match spans.last_mut() {
            Some((part, _, end)) if *part == place.part && *end == place.entry => *end += 1,
            _ => spans.push((place.part, place.entry, place.entry + 1)),
        }
    }
    spans
}

/// Stops a read whose client has gone or whose deadline has passed. SQLite
/// steps are interrupted by the store; this covers the work between them.
fn still_wanted() -> Result<(), Error> {
    if let Some(limit) = ReadLimit::current() {
        limit.check()?;
    }
    Ok(())
}

pub(super) fn page(
    reader: &Reader<'_>,
    d: &impl ReadServices,
    request: &ReadRequest,
    operation: &Arc<crate::operation::Context>,
    scope: Scope,
) -> Result<ReadPage, Error> {
    let ReadRequest {
        target,
        options,
        format,
        budget,
        cursor,
        ..
    } = request;
    let format = Format::parse(format).ok_or_else(|| Error::bad_request("unknown read format"))?;
    let mut limits = PageBudget::default();
    if let Some(tokens) = budget {
        limits.tokens = std::num::NonZeroU64::new(*tokens as u64)
            .ok_or_else(|| Error::bad_request("read budget must be positive"))?;
    }
    let digest = options_digest(&(options, format));
    let key = super::cursor_seal(reader)?;
    let opened = cursor
        .as_deref()
        .map(|text| {
            let opened = PageCursor::open(text, &key).map_err(cursor_failure)?;
            opened.check(target, &digest).map_err(cursor_failure)?;
            Ok::<_, Error>(opened)
        })
        .transpose()?;
    if options.physical && !target.starts_with("root:") {
        return Err(Error::bad_request(
            "--physical applies to root:REF; a session reference is already one physical session",
        ));
    }
    let scope = Scope {
        logical: scope.logical && !options.physical,
        ..scope
    };
    let composed = compose(d, reader, &scope, options)?;
    still_wanted()?;
    let read_session = reader
        .session(&scope.session)?
        .ok_or_else(|| Error::not_found(format!("no session {}", scope.session)))?;
    let versions = PageCursor::version(&composed.versions(reader)?);
    if let Some(opened) = &opened
        && opened.rewritten(&versions)
    {
        return Err(Error::stale_cursor(
            "this session or one of its linked sessions changed since this page was issued",
        ));
    }
    still_wanted()?;
    let runs = runs(reader, &composed, options)?;
    still_wanted()?;
    let shown = ShownEntries::new(reader, &composed, &runs, ShowProfile::of(options))?;
    still_wanted()?;
    // Where a key sits in the reading: the first part that holds it.
    let locate = |key: &EntryKey| -> Result<Option<(usize, usize)>, Error> {
        for (at, (part, index)) in composed.parts.iter().zip(&composed.indexes).enumerate() {
            if let Some(seq) = reader.item_place(index.pk, key)?
                && (part.start..part.end).contains(&(seq as usize))
            {
                return Ok(Some((at, seq as usize)));
            }
        }
        Ok(None)
    };
    let inherited_messages = if !options.inherited && options.view != View::Context {
        let mut total = 0;
        for (at, part) in composed.parts.iter().enumerate() {
            if part.inherited {
                total +=
                    composed.count(reader, at, (part.start, part.end), EntryClass::is_header)?;
            }
        }
        total
    } else {
        0
    };
    let lead = lead_items(
        &composed,
        &runs,
        options,
        target,
        &read_session,
        inherited_messages,
        if options.assertions {
            layout_facts::annotations(
                reader,
                &polylogue_model::Reference::Session(read_session.id.clone()),
                operation.ran_at,
                options.annotation_position(),
            )?
        } else {
            vec![]
        },
    );
    let tokens = limits.tokens.get();
    let items = limits.items.get();
    let (page_entries, next_entry): (Vec<PageEntry>, Option<PageEntry>) = match &opened {
        None if options.latest => (shown.page_before(shown.end(), tokens, items)?, None),
        // The first page's lead lines spend its budget first; when they
        // use all of it, the page is those lines alone.
        None => {
            let lead_tokens: u64 = lead
                .iter()
                .map(|item| {
                    serde_json::to_string(item).map_or(0, |text| estimate_tokens(&text) as u64)
                })
                .sum();
            if lead_tokens >= tokens || lead.len() >= items {
                (Vec::new(), shown.first_at(shown.start())?)
            } else {
                let page =
                    shown.page_after(shown.start(), tokens - lead_tokens, items - lead.len())?;
                (page.entries, page.next)
            }
        }
        Some(opened) => {
            let PagePosition::Entry { key, inside } = &opened.position else {
                return Err(Error::invalid_query(
                    "the cursor names a row position, not a session entry",
                ));
            };
            let place = locate(key)?.ok_or_else(|| {
                Error::stale_cursor(
                    "the message or event this page continues from is no longer in the reading",
                )
            })?;
            let after = (place.0, place.1 + 1);
            match opened.direction {
                // The rest of an entry too large for one page is a page of
                // its own.
                ReadDirection::After if inside.is_some() => {
                    let entry = shown
                        .first_at(place)?
                        .filter(|entry| entry.place() == place)
                        .ok_or_else(|| {
                            Error::stale_cursor("the message or event this page continues from is no longer shown by these options")
                        })?;
                    (vec![entry], shown.first_at(after)?)
                }
                ReadDirection::After => {
                    let page = shown.page_after(place, tokens, items)?;
                    (page.entries, page.next)
                }
                ReadDirection::Before => (
                    shown.page_before(place, tokens, items)?,
                    shown.first_at(place)?,
                ),
                ReadDirection::Past => {
                    let page = shown.page_after(after, tokens, items)?;
                    (page.entries, page.next)
                }
            }
        }
    };
    still_wanted()?;
    let start = match page_entries.first().or(next_entry.as_ref()) {
        Some(first) => shown.position(first.place())?,
        None => shown.len(),
    };
    let mut header = header(reader, &scope.session, target, options)?;
    if scope.logical {
        header.kind = logical_header(reader, &scope.session, &composed)?;
    }
    let mut items_out: Vec<ReadItem> = Vec::new();
    // Lead lines open the reading: on its first page, and on a backward
    // page that reaches the start; never again after a lead-only page.
    let continues_forward = opened.as_ref().is_some_and(|opened| {
        matches!(opened.direction, ReadDirection::After | ReadDirection::Past)
    });
    // A forward page at the start that left out the lead lines has them
    // on the page before it.
    let lead_before = start == 0 && continues_forward;
    if start == 0 && !continues_forward {
        items_out.extend(lead);
    }
    let mut laid = lay_out(
        reader,
        d,
        &composed,
        &runs,
        &page_entries,
        options,
        operation.ran_at,
    )?;
    still_wanted()?;
    decorate(reader, &composed, &shown, &page_entries, &mut laid, options)?;
    still_wanted()?;
    // Whole entries up to the transfer size; an entry too large to travel
    // alone is replaced by the ready call that reads its block in pages.
    let room = limits.bytes.get();
    let used = serde_json::to_vec(&header)
        .map(|bytes| bytes.len())
        .unwrap_or(0)
        + items_out
            .iter()
            .map(|item| serde_json::to_vec(item).map_or(0, |bytes| bytes.len()))
            .sum::<usize>();
    let sizes: Vec<usize> = laid
        .iter()
        .map(|items| {
            items
                .iter()
                .map(|item| serde_json::to_vec(item).map_or(0, |bytes| bytes.len()))
                .sum()
        })
        .collect();
    // Whole entries that fit the transfer, at least one. A backward page
    // keeps the entries next to its cursor and gives up its first ones.
    let backward = opened
        .as_ref()
        .is_some_and(|opened| opened.direction == ReadDirection::Before);
    let fit = |order: &mut dyn Iterator<Item = usize>| {
        let mut total = used;
        let mut count = 0;
        for index in order {
            if count > 0 && total + sizes[index] > room {
                break;
            }
            total += sizes[index];
            count += 1;
        }
        count
    };
    let count = if backward {
        fit(&mut (0..laid.len()).rev())
    } else {
        fit(&mut (0..laid.len()))
    };
    let keep = if backward {
        laid.len() - count..laid.len()
    } else {
        0..count
    };
    let start = start + keep.start as u64;
    // An entry continued inside, or too large for the page alone, is sent
    // as fragments of its items' encodings; the page ends where they stop.
    let inside = opened.as_ref().and_then(|opened| match &opened.position {
        PagePosition::Entry { inside, .. } => *inside,
        _ => None,
    });
    let mut unfinished = None;
    for (offset, entry_items) in laid.into_iter().enumerate() {
        if !keep.contains(&offset) {
            continue;
        }
        if inside.is_some() || (keep.len() == 1 && sizes[offset] > room) {
            let entry = &page_entries[offset];
            let kind = match entry.entry.key {
                EntryKey::Message { .. } => ReadRecordKind::Message,
                EntryKey::Event { .. } => ReadRecordKind::Event,
            };
            // `used` already holds the header and the lead lines, the only
            // items before an entry that is alone on its page.
            unfinished = fragments(
                entry_items,
                inside,
                target,
                kind,
                room.saturating_sub(used),
                tokens,
                &mut items_out,
            )?
            .map(|place| (entry.entry.key.clone(), place));
        } else {
            items_out.extend(entry_items);
        }
    }
    let kept = &page_entries[keep.clone()];
    // The entry after the page: the first one left out, or the one after
    // every entry read.
    let after_page = page_entries.get(keep.end).or(next_entry.as_ref());
    let call = |entry: &PageEntry, direction: ReadDirection| -> Result<ReadyCall, Error> {
        let next = PageCursor {
            target: target.clone(),
            options: digest.clone(),
            variants: request.options.variants.clone(),
            versions,
            position: PagePosition::Entry {
                key: entry.entry.key.clone(),
                inside: None,
            },
            direction,
        };
        super::next_call(d, request, next.seal(&key), super::deadline(operation))
    };
    let finished = unfinished.is_none();
    let next = match unfinished {
        Some((entry_key, place)) => {
            let cursor = PageCursor {
                target: target.clone(),
                options: digest.clone(),
                variants: request.options.variants.clone(),
                versions,
                position: PagePosition::Entry {
                    key: entry_key,
                    inside: Some(place),
                },
                direction: ReadDirection::After,
            };
            Some(super::next_call(
                d,
                request,
                cursor.seal(&key),
                super::deadline(operation),
            )?)
        }
        None => after_page
            .map(|entry| call(entry, ReadDirection::After))
            .transpose()?,
    };
    let first = kept.first().or(after_page);
    let footer = ReadFooter {
        estimator: Some(TOKEN_ESTIMATOR.into()),
        follow: match (finished && after_page.is_none(), kept.last()) {
            (true, Some(last)) => Some(call(last, ReadDirection::Past)?),
            _ => None,
        },
        next,
        previous: match first {
            Some(first) if start > 0 || lead_before => Some(call(first, ReadDirection::Before)?),
            _ => None,
        },
        shown: items_out.len() as u64,
        position: Some(start),
        count: ValueState::known(shown.len()),
        ..ReadFooter::default()
    };
    let positions = reader.change_position()?;
    let snapshot = ReadPageSnapshot {
        query: target.clone(),
        ran_at: operation.ran_at,
        read_at: positions,
        freshness: d.freshness(reader)?,
    };
    Ok(ReadPage {
        format,
        snapshot,
        records: vec![],
        document: ReadDocument {
            header,
            items: items_out,
            footer,
        },
    })
}

use super::cursor_failure;

pub(super) fn header(
    reader: &Reader<'_>,
    session: &SessionId,
    target: &str,
    options: &ReadOptions,
) -> Result<ReadHeader, Error> {
    let stored = reader
        .session(session)?
        .ok_or_else(|| Error::not_found(format!("no session {session}")))?;
    let title = reader
        .display_title(session)?
        .map(ValueState::known)
        .unwrap_or_default();
    Ok(ReadHeader {
        target: target.to_owned(),
        view: options.view,
        title,
        kind: HeaderKind::Session {
            session: Box::new(stored),
        },
        figures: Vec::new(),
        zone: None,
    })
}

/// The first page begins with what the reading leaves out: an inherited
/// prefix shown as one line, link problems, and for the context view what
/// was not recorded and what the newest boundary removed.
fn lead_items(
    composed: &Composed,
    runs: &Runs,
    options: &ReadOptions,
    target: &str,
    session: &polylogue_model::Session,
    inherited: u64,
    assertions: Vec<Annotation>,
) -> Vec<ReadItem> {
    // The read session's card opens the reading, with whether its lineage
    // composed without a problem.
    let mut out = vec![ReadItem {
        source: Source::record(format!("session:{}", session.id)),
        kind: ItemKind::Card {
            card: Box::new(Card::Session {
                assertions,
                session: Box::new(session.clone()),
                // Whether the parents composed, when the reading shows them,
                // as the stream's session record says.
                lineage_complete: if options.inherited {
                    ValueState::known(composed.issues.is_empty())
                } else {
                    ValueState::NotApplicable
                },
                open: None,
            }),
        },
    }];
    if let Some(position) = options.annotation_position() {
        out.push(ReadItem {
            source: Source::record(target),
            kind: ItemKind::Text {
                text: format!("Annotations as of {position}; expiry uses this read's clock."),
            },
        });
    }
    let source = Source::record(target);
    for issue in composed.issues.iter().filter(|_| options.inherited) {
        out.push(ReadItem {
            source: source.clone(),
            kind: ItemKind::LinkIssue {
                issue: issue.clone(),
            },
        });
    }
    if !options.inherited
        && options.view != View::Context
        && let Some(parent) = composed.parts.iter().rev().find(|part| part.inherited)
    {
        out.push(ReadItem {
            source: source.clone(),
            kind: ItemKind::Indicator {
                indicator: IndicatorKind::Continues,
                text: format!(
                    "continues {} ({inherited} earlier messages; --show inherited to show)",
                    parent.session
                ),
                expand: None,
            },
        });
    }
    if let Some((complete, reason)) = &runs.context {
        out.push(ReadItem {
            source: source.clone(),
            kind: ItemKind::Omission {
                count: ValueState::NotRecorded,
                unit: "system prompt, tool definitions".into(),
                why: "not recorded in context".into(),
                continue_call: None,
            },
        });
        for key in &runs.evicted {
            out.push(ReadItem {
                source: Source::record(key.reference()),
                kind: ItemKind::Omission {
                    count: ValueState::known(1),
                    unit: "message".into(),
                    why: if runs.left {
                        "left the context between the two places".into()
                    } else {
                        "removed from context".into()
                    },
                    continue_call: None,
                },
            });
        }
        if !complete {
            out.push(ReadItem {
                source,
                kind: ItemKind::Omission {
                    count: ValueState::Unknown {
                        reason: reason.clone(),
                    },
                    unit: "context".into(),
                    why: reason
                        .clone()
                        .unwrap_or_else(|| "context is incomplete".into()),
                    continue_call: None,
                },
            });
        }
    }
    out
}

/// The items of one entry as whole items while they fit `room` bytes, and
/// from the first that does not as fragments of its archive JSON encoding,
/// starting at `from` when the page continues inside the entry. Returns the
/// place the next page starts at when the entry is not finished.
/// Concatenating the fragments gives back each item's encoding.
fn fragments(
    items: Vec<ReadItem>,
    from: Option<ItemPlace>,
    target: &str,
    kind: ReadRecordKind,
    room: usize,
    tokens: u64,
    out: &mut Vec<ReadItem>,
) -> Result<Option<ItemPlace>, Error> {
    let first = from.map_or(0, |place| place.item as usize);
    if first >= items.len() && !items.is_empty() {
        return Err(Error::stale_cursor(
            "the part this page continues from is no longer in its message or event",
        ));
    }
    let mut used = 0;
    let mut placed = false;
    for (index, item) in items.into_iter().enumerate().skip(first) {
        let encoded = polylogue_render::json_bytes(&item).map_err(Error::internal)?;
        let text = String::from_utf8(encoded).map_err(Error::internal)?;
        let item_hash = polylogue_model::BlobHash::of(text.as_bytes());
        let place = |character: usize| ItemPlace {
            selection: SelectionLocator {
                row: 0,
                record: ReadRecordLocator { kind, ordinal: 0 },
            },
            item: index as u64,
            character: character as u64,
            item_hash,
        };
        let start = match from {
            Some(from) if index == first => {
                if from.item_hash != item_hash {
                    return Err(Error::stale_cursor(
                        "the part this page continues from changed since this page was issued",
                    ));
                }
                from.character as usize
            }
            _ => 0,
        };
        if start == 0 && used + text.len() <= room {
            used += text.len();
            placed = true;
            out.push(item);
            continue;
        }
        if placed {
            return Ok(Some(place(0)));
        }
        let boundaries: Vec<usize> = text
            .char_indices()
            .map(|(at, _)| at)
            .chain(std::iter::once(text.len()))
            .collect();
        let total = boundaries.len() - 1;
        if start >= total {
            return Err(Error::stale_cursor(
                "the part this page continues from is shorter than when this page was issued",
            ));
        }
        // A fragment is sourced at the reading, as the row pager's are;
        // its position names the item.
        let make = |until: usize| ReadItem {
            source: Source::record(target),
            kind: ItemKind::Fragment {
                text: text[boundaries[start]..boundaries[until]].into(),
                position: place(start),
                until: until as u64,
                total: total as u64,
            },
        };
        let fits = |until: usize| -> Result<bool, Error> {
            let candidate = polylogue_render::json_bytes(&make(until)).map_err(Error::internal)?;
            let payload = &text[boundaries[start]..boundaries[until]];
            Ok(used + candidate.len() <= room && estimate_tokens(payload) as u64 <= tokens)
        };
        // At least one character, so every page moves forward.
        let (mut low, mut high) = (start + 1, total);
        while low < high {
            let middle = low + (high - low).div_ceil(2);
            if fits(middle)? {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        let fragment = make(low);
        used += polylogue_render::json_bytes(&fragment)
            .map_err(Error::internal)?
            .len();
        placed = true;
        out.push(fragment);
        if low < total {
            return Ok(Some(place(low)));
        }
    }
    Ok(None)
}

/// The entry an item belongs to, from its source.
fn item_key(item: &ReadItem) -> Option<EntryKey> {
    match (&item.source.message, item.source.block) {
        (Some(message), block) => Some(EntryKey::Message {
            message: message.clone(),
            block,
        }),
        (None, _) => {
            let reference = &item.source.reference;
            if let Some(event) = reference.strip_prefix("event:") {
                Some(EntryKey::Event {
                    event: event.to_owned(),
                })
            } else {
                // A line beside a message (a collapsed harness message's
                // indicator) belongs to that message's header.
                reference
                    .strip_prefix("message:")
                    .and_then(|id| MessageId::parse(id).ok())
                    .map(|message| EntryKey::header(&message))
            }
        }
    }
}

/// Lay out the page's entries, grouped by entry in page order. Only the
/// messages and events of the page are read.
fn lay_out(
    reader: &Reader<'_>,
    d: &impl ReadServices,
    composed: &Composed,
    reading: &Runs,
    page: &[PageEntry],
    options: &ReadOptions,
    at: polylogue_model::Timestamp,
) -> Result<Vec<Vec<ReadItem>>, Error> {
    // A raw reading is the stored records, which `decorate` adds.
    if options.view == View::Raw {
        return Ok(vec![Vec::new(); page.len()]);
    }
    let mut slots: Vec<Vec<ReadItem>> = vec![Vec::new(); page.len()];
    let mut wanted: std::collections::HashMap<EntryKey, usize> = std::collections::HashMap::new();
    for (slot, entry) in page.iter().enumerate() {
        wanted.insert(entry.entry.key.clone(), slot);
    }
    // The page's entries grouped into runs of one part.
    let mut runs: Vec<(usize, Vec<&IndexEntry>)> = Vec::new();
    for entry in page {
        match runs.last_mut() {
            Some((part, entries)) if *part == entry.part => entries.push(&entry.entry),
            _ => runs.push((entry.part, vec![&entry.entry])),
        }
    }
    let selected = |source: &Source| -> Result<bool, Error> {
        let Some(message) = &source.message else {
            return Ok(false);
        };
        let key = EntryKey::Message {
            message: message.clone(),
            block: source.block,
        };
        if wanted.contains_key(&key) {
            return Ok(true);
        }
        for (part, start, end) in &reading.spans {
            if reader
                .item_place(composed.indexes[*part].pk, &key)?
                .is_some_and(|entry| (*start..*end).contains(&(entry as usize)))
            {
                return Ok(true);
            }
        }
        Ok(false)
    };
    let ready_attachment = |message: &MessageId, block, index, input: Option<&str>| {
        d.attachment_call(&crate::AttachmentPlace {
            message: message.clone(),
            block,
            attachment: index,
            input: input.map(str::to_owned),
        })
    };
    let mut footer = ReadFooter::default();
    // Each item goes to the entry its source names; items of entries not on
    // this page are left for the page that holds them.
    let mut place = |item: ReadItem, _: &mut Option<usize>| {
        if let Some(slot) = item_key(&item).and_then(|key| wanted.get(&key).copied()) {
            slots[slot].push(item);
        }
    };
    // A message split across pieces (a spawning call inside it) is laid
    // out once: its items go to every slot of the page that holds them.
    let mut laid_messages: HashSet<MessageId> = HashSet::new();
    for (run_part, run_entries) in runs {
        let part = &composed.parts[run_part];
        let inherited = part.inherited;
        let mut messages: Vec<MessageId> = Vec::new();
        for entry in run_entries {
            match &entry.key {
                EntryKey::Message { message, .. } => {
                    if messages.last() != Some(message) {
                        messages.push(message.clone());
                    }
                }
                EntryKey::Event { event } => {
                    let record = reader.event_record(event)?.ok_or_else(|| {
                        Error::stale_cursor(format!("event {event} is no longer in this session"))
                    })?;
                    emit(
                        reader,
                        Record::Event(record),
                        LayoutFacts::default(),
                        options,
                        at,
                        &selected,
                        &ready_attachment,
                        &mut footer,
                        &mut |item| {
                            let mut last = None;
                            place(item, &mut last);
                        },
                    )?;
                }
            }
        }
        messages.retain(|message| laid_messages.insert(message.clone()));
        if messages.is_empty() {
            continue;
        }
        let selection = reader.select_messages(&part.session, false)?;
        let start = selection.locate(Some(&messages[0]))?.ok_or_else(|| {
            Error::stale_cursor(format!(
                "message {} is no longer in this session",
                messages[0]
            ))
        })?;
        let wanted_messages: HashSet<&MessageId> = messages.iter().collect();
        let last_message = messages.last().expect("one message").clone();
        let mut result = Ok(());
        selection.visit_with_sources(
            start,
            |message,
             polylogue_store::read::MessageSource {
                 blocks: block_selection,
                 committed: commit,
                 ordinal: order,
             }| {
                let id = message.id.clone();
                if wanted_messages.contains(&id) {
                    let step = (|| {
                        let commit = commit.ok_or_else(|| {
                            Error::not_yet_available(format!(
                                "message:{id} has no stored commit position for its source"
                            ))
                        })?;
                        let record =
                            message_record(reader, message, block_selection, commit, order)?;
                        let facts = LayoutFacts {
                            inherited,
                            ..Default::default()
                        };
                        let mut last = None;
                        emit(
                            reader,
                            record,
                            facts,
                            options,
                            at,
                            &selected,
                            &ready_attachment,
                            &mut footer,
                            &mut |item| place(item, &mut last),
                        )
                    })();
                    if let Err(error) = step {
                        result = Err(error);
                        return ControlFlow::Break(());
                    }
                }
                if id == last_message {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
        )?;
        result?;
    }
    Ok(slots)
}

/// What a reading adds around laid-out entries: `continues` lines and
/// cards from composition, turn dividers, the context meter on model
/// headers, and for the raw view each record's stored bytes.
fn decorate(
    reader: &Reader<'_>,
    composed: &Composed,
    shown: &ShownEntries,
    page: &[PageEntry],
    laid: &mut [Vec<ReadItem>],
    options: &ReadOptions,
) -> Result<(), Error> {
    let human_prompt = |entry: &IndexEntry| {
        entry.kind == EntryKind::Header
            && entry.written_by == polylogue_model::WrittenBy::Human
            && entry.prompt_kind == Some(polylogue_model::PromptKind::Prose)
    };
    let mut placed = place_cards(composed, shown, page, options)?;
    for (offset, items) in laid.iter_mut().enumerate() {
        let PageEntry {
            part,
            seq: at,
            entry,
        } = &page[offset];
        let (part, at) = (*part, *at);
        let mut before: Vec<ReadItem> =
            composed.before.get(&entry.key).cloned().unwrap_or_default();
        if options.turns && human_prompt(entry) {
            let (number, calls) = turn(reader, composed, part, at)?;
            before.push(ReadItem {
                source: Source::record(entry.key.reference()),
                kind: ItemKind::Indicator {
                    indicator: IndicatorKind::Turn,
                    text: format!("turn {number} · {calls} calls"),
                    expand: None,
                },
            });
        }
        if options.context_meter && entry.written_by == polylogue_model::WrittenBy::Model {
            let meter = context_meter(reader, composed, part, at, entry)?;
            for item in items.iter_mut() {
                if let ItemKind::MessageHeader { header } = &mut item.kind {
                    header.context = Some(match meter {
                        ContextMeter::Reported(tokens) => ValueState::known(tokens),
                        ContextMeter::Estimated(tokens) => ValueState::Inferred { value: tokens },
                        ContextMeter::Unknown => ValueState::Unknown {
                            reason: Some("no usage recorded".into()),
                        },
                    });
                }
            }
        }
        if options.view == View::Raw {
            let record = match &entry.key {
                EntryKey::Message {
                    message,
                    block: None,
                } => Some((
                    polylogue_store::items::RecordTable::Message,
                    message.to_string(),
                )),
                EntryKey::Event { event } => {
                    Some((polylogue_store::items::RecordTable::Event, event.clone()))
                }
                _ => None,
            };
            if let Some((table, id)) = record {
                let stored = reader.stored_record(table, &id)?.ok_or_else(|| {
                    Error::not_yet_available(format!(
                        "{} has no stored record",
                        entry.key.reference()
                    ))
                })?;
                let media_type = if stored.bytes.first() == Some(&b'{') {
                    "application/json"
                } else {
                    "application/octet-stream"
                };
                items.push(ReadItem {
                    source: Source::record(entry.key.reference()),
                    kind: ItemKind::Raw {
                        record: Box::new(RawRecord {
                            reference: entry.key.reference(),
                            input: stored.input,
                            media_type: media_type.into(),
                            bytes: stored.bytes,
                        }),
                    },
                });
            }
        }
        let (ahead, after) = placed.remove(&offset).unwrap_or_default();
        let mut joined = before;
        joined.extend(ahead);
        joined.append(items);
        joined.extend(after);
        *items = joined;
    }
    Ok(())
}

/// The number of the turn a prompt opens, and the calls up to the next
/// prompt. Turns are counted over the composed reading within each part's
/// bounds; an inline spawned session numbers its own turns. The calls run
/// across the pieces of a part split at a spawning call.
fn turn(
    reader: &Reader<'_>,
    composed: &Composed,
    part: usize,
    at: usize,
) -> Result<(u64, u64), Error> {
    let bounds = &composed.parts[part];
    let same = composed.same_context(part);
    let mut earlier = 0;
    for (index, other) in composed.parts[..part].iter().enumerate() {
        if same(other) {
            earlier += composed.count(
                reader,
                index,
                (other.start, other.end),
                EntryClass::is_prompt,
            )?;
        }
    }
    let number =
        earlier + composed.count(reader, part, (bounds.start, at), EntryClass::is_prompt)? + 1;
    let mut ranges = vec![(part, at + 1, bounds.end)];
    for (offset, other) in composed.parts[part + 1..].iter().enumerate() {
        if other.session == bounds.session {
            ranges.push((part + 1 + offset, other.start, other.end));
        }
    }
    let mut calls = 0;
    for (index, start, end) in ranges {
        if start >= end {
            continue;
        }
        let next = reader.item_find(
            composed.indexes[index].pk,
            ItemLookup::Prompt,
            start as u64..end as u64,
            true,
        )?;
        let end = next.map_or(end, |next| next as usize);
        calls += composed.count(reader, index, (start, end), EntryClass::is_call)?;
        if next.is_some() {
            break;
        }
    }
    Ok((number, calls))
}

/// The context meter of a model message: the provider-reported input of its
/// response, or else the estimate of every entry in context at it (since
/// the newest compaction, less the results a later microcompaction
/// removed), from the stored running sums.
fn context_meter(
    reader: &Reader<'_>,
    composed: &Composed,
    part: usize,
    at: usize,
    entry: &IndexEntry,
) -> Result<ContextMeter, Error> {
    if let Some(tokens) = entry.context_tokens {
        return Ok(ContextMeter::Reported(tokens));
    }
    let parts = composed.context_parts(part);
    let mut ranges: Vec<(usize, usize, usize)> = parts
        .iter()
        .enumerate()
        .take(part + 1)
        .map(|(index, other)| {
            let end = if index == part {
                (at + 1).min(other.end)
            } else {
                other.end
            };
            (index, other.start, end)
        })
        .filter(|(_, start, end)| start < end)
        .collect();
    for which in (0..ranges.len()).rev() {
        let (index, start, end) = ranges[which];
        if let Some(boundary) = reader.item_find(
            composed.indexes[index].pk,
            ItemLookup::Compaction,
            start as u64..end as u64,
            false,
        )? {
            ranges.drain(..which);
            ranges[0].1 = boundary as usize + 1;
            break;
        }
    }
    let mut estimate: u64 = 0;
    for &(index, start, end) in &ranges {
        if start < end {
            estimate += composed
                .counts_at(reader, index, end)?
                .since(&composed.counts_at(reader, index, start)?)
                .tokens_full;
        }
    }
    // A microcompaction removes the results it names that came before it.
    let mut removed: HashSet<(usize, u64)> = HashSet::new();
    for &(index, start, end) in &ranges {
        if start >= end {
            continue;
        }
        let pk = composed.indexes[index].pk;
        for micro in
            reader.item_find_all(pk, ItemLookup::Microcompaction, start as u64..end as u64)?
        {
            let Some(cleared) = composed.entry(reader, index, micro as usize)? else {
                continue;
            };
            for call in &cleared.cleared {
                for &(other, other_start, other_end) in &ranges {
                    for stored in reader.item_calls(composed.indexes[other].pk, call)? {
                        let place = (other, stored.seq as usize);
                        if stored.entry.kind == EntryKind::Result
                            && (other_start..other_end).contains(&place.1)
                            && place < (index, micro as usize)
                            && removed.insert((other, stored.seq))
                        {
                            estimate = estimate.saturating_sub(stored.entry.tokens_full);
                        }
                    }
                }
            }
        }
    }
    Ok(if estimate == 0 {
        ContextMeter::Unknown
    } else {
        ContextMeter::Estimated(estimate)
    })
}

/// Per page slot, the cards placed before its entry and after it.
type PlacedCards = std::collections::HashMap<usize, (Vec<ReadItem>, Vec<ReadItem>)>;

/// Cards for each page slot, placed after the entry they follow when it is
/// shown, or before the next shown entry when the reading hides that entry
/// (a spawning call in the dialogue view), so a card is never lost. A card
/// belongs to this page when that entry comes after the shown entry before
/// the page and no later than the page's last entry.
fn place_cards(
    composed: &Composed,
    shown: &ShownEntries,
    page: &[PageEntry],
    options: &ReadOptions,
) -> Result<PlacedCards, Error> {
    let mut placed = PlacedCards::new();
    let (Some(first), Some(last)) = (page.first(), page.last()) else {
        return Ok(placed);
    };
    if composed.after.is_empty() {
        return Ok(placed);
    }
    let previous = shown.last_before(first.place())?.map(|entry| entry.place());
    for card_slot in &composed.after {
        let Some(place) = composed
            .parts
            .iter()
            .position(|bounds| {
                bounds.session == card_slot.session
                    && (bounds.start..bounds.end).contains(&card_slot.seq)
            })
            .map(|part| (part, card_slot.seq))
        else {
            continue;
        };
        if place > last.place() || previous.is_some_and(|previous| place <= previous) {
            continue;
        }
        let slot = page.partition_point(|entry| entry.place() < place);
        let cards = card_slot.cards.iter().filter(|item| {
            options.spawns != SpawnFold::Hidden
                || !matches!(&item.kind, ItemKind::Card { card } if matches!(**card, Card::Delegation { .. }))
        });
        let held = placed.entry(slot).or_default();
        if page[slot].place() == place {
            held.1.extend(cards.cloned());
        } else {
            held.0.extend(cards.cloned());
        }
    }
    Ok(placed)
}

#[allow(clippy::too_many_arguments)]
fn emit(
    reader: &Reader<'_>,
    record: Record,
    facts: LayoutFacts,
    options: &ReadOptions,
    at: polylogue_model::Timestamp,
    selected: &dyn Fn(&Source) -> Result<bool, Error>,
    ready_attachment: &super::attachment_facts::ReadyAttachment<'_>,
    footer: &mut ReadFooter,
    out: &mut dyn FnMut(ReadItem),
) -> Result<(), Error> {
    // A context reading's entries were chosen from the index as members, so
    // each one laid out is selected; what it leaves out is said once, in
    // the reading's first lines.
    let mut facts = facts;
    if options.assertions {
        layout_facts::annotate(
            reader,
            &record,
            &mut facts,
            at,
            options.annotation_position(),
        )?;
    }
    if options.view == View::Context {
        facts.context = Some(ContextState {
            selected: true,
            complete: true,
            reason: None,
            evicted: Vec::new(),
            not_recorded: Vec::new(),
        });
    }
    let Some(record) = output::project(
        SelectedRecord::Original(Box::new(record)),
        options,
        &facts,
        false,
        footer,
    )?
    else {
        return Ok(());
    };
    let layout = layout_facts::Layout {
        options,
        facts: &facts,
        selected,
        ready_attachment,
        backwards: false,
        page_start: None,
    };
    layout_facts::visit_parts(reader, &record, &layout, |item, _| {
        out(item);
        Ok(true)
    })?;
    Ok(())
}
