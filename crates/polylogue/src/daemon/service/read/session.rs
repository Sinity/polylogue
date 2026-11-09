//! Paged session reads over the session item index. A page is a run of
//! shown entries chosen by running token sums; only the bodies of that
//! page's messages and events are read. A continuation names its first
//! entry by key and binds the rewrite version of every composed session,
//! so appends never stale it and rewrites always do. Nothing is written.
use super::super::{Daemon, Failure};
use super::ReadRequest;
use super::records::{SelectedRecord, message_record};
use super::{evidence, output};
use crate::daemon::items::CachedIndex;
use polylogue_model::{MessageId, Reference, SessionId, bundle::Record, read::*};
use polylogue_store::Reader;
use std::collections::HashSet;
use std::ops::ControlFlow;
use std::sync::Arc;

mod entries;
use entries::ShownEntries;

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
pub(super) fn scope(reader: &Reader<'_>, target: &str) -> Result<Option<Scope>, Failure> {
    if let Some(root) = target.strip_prefix("root:") {
        let reference = Reference::parse(&format!("session:{root}"))
            .map_err(|e| Failure::new("bad_request", e))?;
        let Reference::Session(session) = reference else {
            return Err(Failure::new(
                "bad_request",
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
    let owner = |message: &MessageId| -> Result<SessionId, Failure> {
        reader
            .message_session(message)?
            .ok_or_else(|| Failure::new("not_found", format!("no message {message}")))
    };
    Ok(match reference {
        Reference::Session(session) => Some(Scope {
            session,
            range: None,
            logical: false,
        }),
        Reference::Message(message) => Some(Scope {
            session: owner(&message)?,
            range: Some((message.clone(), message)),
            logical: false,
        }),
        Reference::MessageRange { first, last } => {
            let session = owner(&first)?;
            if owner(&last)? != session {
                return Err(Failure::new(
                    "bad_request",
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

/// One composed session reading: its parts, each with the cached index of
/// its session, and the link problems found while composing.
struct Composed {
    parts: Vec<Part>,
    indexes: Vec<Arc<CachedIndex>>,
    issues: Vec<ParentIssue>,
    /// Lines shown before an entry (a `continues` line at a continuation).
    before: std::collections::HashMap<EntryKey, Vec<ReadItem>>,
    /// Cards shown after an entry (a fork at its branch point, a spawned
    /// session at its call).
    after: std::collections::HashMap<EntryKey, Vec<ReadItem>>,
}
/// A part that reads a spawned session inline.
fn spawned(part: &Part) -> bool {
    part.joined_by
        .as_deref()
        .is_some_and(|how| how.starts_with("subagent:"))
}

impl Composed {
    fn entries(&self, part: usize) -> &[IndexEntry] {
        &self.indexes[part].entries
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
    fn versions(&self, reader: &Reader<'_>) -> Result<Vec<SessionVersion>, Failure> {
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

/// The parent prefix of `session` through its branch point, as parts of
/// parent sessions, all marked inherited. The walk is iterative over
/// sessions already on the path; problems are reported, never resolved by
/// choosing one link.
fn prefix(
    d: &Daemon,
    reader: &Reader<'_>,
    session: &SessionId,
    composed: &mut Composed,
) -> Result<(), Failure> {
    let mut chain: Vec<(SessionId, Option<String>)> = Vec::new();
    let mut on_path: HashSet<SessionId> = HashSet::from([session.clone()]);
    let mut current = session.clone();
    loop {
        let links = reader.composing_links(&current)?;
        let forks: std::collections::BTreeSet<(String, String)> = links
            .iter()
            .filter(|link| link.how == "fork")
            .filter_map(|link| Some((link.parent.clone(), link.branch_point.clone()?)))
            .collect();
        if forks.len() > 1 {
            let mut recorded = forks.into_iter();
            let (first, first_anchor) = recorded.next().expect("two forks");
            let (second, second_anchor) = recorded.next().expect("two forks");
            if let (Ok(first), Ok(second)) = (SessionId::parse(&first), SessionId::parse(&second)) {
                composed.issues.push(ParentIssue::ConflictingParents {
                    child: current.clone(),
                    first,
                    first_anchor,
                    second,
                    second_anchor,
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
            let (first, first_anchor) = recorded.next().expect("two parents");
            let (second, second_anchor) = recorded.next().expect("two parents");
            if let (Ok(first), Ok(second)) = (SessionId::parse(first), SessionId::parse(second)) {
                composed.issues.push(ParentIssue::ConflictingParents {
                    child: current.clone(),
                    first,
                    first_anchor: first_anchor.to_owned(),
                    second,
                    second_anchor: second_anchor.to_owned(),
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
        if link.how == "fork" && link.branch_point.is_none() {
            composed.issues.push(ParentIssue::MissingAnchor {
                child: current.clone(),
                parent,
                anchor: None,
            });
            break;
        }
        chain.push((parent.clone(), link.branch_point.clone()));
        current = parent;
    }
    // Compose from the oldest ancestor down, cutting each at the branch
    // point its child continues from.
    let mut parts: Vec<(Part, Arc<CachedIndex>)> = Vec::new();
    for (at, (parent, branch_point)) in chain.iter().enumerate().rev() {
        let child = if at == 0 {
            session.clone()
        } else {
            chain[at - 1].0.clone()
        };
        let Some(index) = d.warm.items.get(reader, parent)? else {
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
                end: index.entries.len(),
                inherited: true,
                joined_by: None,
            },
            index,
        ));
        if let Some(anchor) = branch_point {
            let cut = MessageId::parse(anchor).ok().and_then(|anchor| {
                parts
                    .iter()
                    .enumerate()
                    .rev()
                    .find_map(|(which, (part, index))| {
                        prefix_through(&index.entries[part.start..part.end], &anchor)
                            .map(|end| (which, part.start + end))
                    })
            });
            match cut {
                Some((which, end)) => {
                    parts.truncate(which + 1);
                    parts[which].0.end = end;
                }
                None => {
                    composed.issues.push(ParentIssue::MissingAnchor {
                        child,
                        parent: parent.clone(),
                        anchor: Some(anchor.clone()),
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
    d: &Daemon,
    reader: &Reader<'_>,
    scope: &Scope,
    options: &ReadOptions,
    target: &str,
) -> Result<Composed, Failure> {
    let around = options.around.unwrap_or(0);
    let own = d
        .warm
        .items
        .get(reader, &scope.session)?
        .ok_or_else(|| Failure::new("not_found", format!("no session {}", scope.session)))?;
    let mut composed = Composed {
        parts: Vec::new(),
        indexes: Vec::new(),
        issues: Vec::new(),
        before: Default::default(),
        after: Default::default(),
    };
    // The prefix is composed always: hidden, it is one `continues` line.
    if scope.range.is_none() {
        prefix(d, reader, &scope.session, &mut composed)?;
    }
    let headers: Vec<usize> = if around > 0 && scope.range.is_some() {
        own.entries
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.kind == EntryKind::Header)
            .map(|(at, _)| at)
            .collect()
    } else {
        Vec::new()
    };
    let (start, end) = match &scope.range {
        None => (0, own.entries.len()),
        Some((first, last)) => {
            let start = own
                .places
                .get(&EntryKey::header(first))
                .copied()
                .ok_or_else(|| {
                    Failure::new(
                        "not_found",
                        format!("message {first} is not in the reading"),
                    )
                })?;
            let last_at = own
                .places
                .get(&EntryKey::header(last))
                .copied()
                .ok_or_else(|| {
                    Failure::new("not_found", format!("message {last} is not in the reading"))
                })?;
            if last_at < start {
                return Err(Failure::new(
                    "bad_request",
                    "a message range ends before it starts",
                ));
            }
            // `--around N`: N messages either side.
            let (start, last_at) = if around > 0 {
                let first = headers.partition_point(|at| *at < start);
                let last = headers.partition_point(|at| *at <= last_at);
                let widened = around as usize;
                (
                    headers[first.saturating_sub(widened)],
                    headers[(last + widened).min(headers.len()) - 1],
                )
            } else {
                (start, last_at)
            };
            let end = own.entries[last_at + 1..]
                .iter()
                .position(|entry| entry.kind == EntryKind::Header)
                .map_or(own.entries.len(), |offset| last_at + 1 + offset);
            (start, end)
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
    children(d, reader, scope, &mut composed, options.view, target)?;
    if let SpawnFold::Inline { depth } = options.spawns {
        inline_spawns(d, reader, &mut composed, depth, options.view, target)?;
    }
    Ok(composed)
}

/// Continuations and resumes follow their parent in a logical reading,
/// each after a `continues` line; forks and spawned sessions appear as cards
/// at their branch point or spawning call. The walk is iterative and visits
/// each session once.
fn children(
    d: &Daemon,
    reader: &Reader<'_>,
    scope: &Scope,
    composed: &mut Composed,
    view: View,
    target: &str,
) -> Result<(), Failure> {
    let mut visited: HashSet<SessionId> = HashSet::from([scope.session.clone()]);
    let mut at = composed.parts.len() - 1;
    while at < composed.parts.len() {
        let session = composed.parts[at].session.clone();
        let index = Arc::clone(&composed.indexes[at]);
        at += 1;
        for link in reader.child_links(&session)? {
            if link.kind == "reference" {
                continue;
            }
            let Ok(child) = SessionId::parse(&link.child) else {
                continue;
            };
            let joins = link.kind == "continuation" && link.how != "fork";
            if joins && scope.logical {
                if !visited.insert(child.clone()) {
                    composed.issues.push(ParentIssue::Cycle { session: child });
                    continue;
                }
                let Some(child_index) = d.warm.items.get(reader, &child)? else {
                    composed.issues.push(ParentIssue::MissingParent {
                        child,
                        parent: session.clone(),
                    });
                    continue;
                };
                if let Some(first) = child_index.entries.first() {
                    composed
                        .before
                        .entry(first.key.clone())
                        .or_default()
                        .push(ReadItem {
                            source: Source::record(format!("session:{child}")),
                            kind: ItemKind::Indicator {
                                indicator: IndicatorKind::Continues,
                                text: format!("continues {session} ({})", link.how),
                                expand: None,
                            },
                        });
                }
                composed.parts.push(Part {
                    session: child,
                    start: 0,
                    end: child_index.entries.len(),
                    inherited: false,
                    joined_by: Some(format!("{}:{}", link.kind, link.how)),
                });
                composed.indexes.push(child_index);
                continue;
            }
            if joins {
                continue;
            }
            // A card at the branch point or the spawning call.
            let anchor = match link.kind.as_str() {
                "subagent" => link.spawning_call.as_ref().and_then(|call| {
                    index
                        .entries
                        .iter()
                        .find(|entry| {
                            entry.kind == EntryKind::Call && entry.call_id.as_deref() == Some(call)
                        })
                        .map(|entry| entry.key.clone())
                }),
                _ => link
                    .branch_point
                    .as_deref()
                    .and_then(|point| MessageId::parse(point).ok())
                    .and_then(|point| {
                        let end = prefix_through(&index.entries, &point)?;
                        index
                            .entries
                            .get(end.checked_sub(1)?)
                            .map(|entry| entry.key.clone())
                    }),
            };
            let Some(anchor) = anchor else { continue };
            let card = card(d, reader, &link, &child, &index, view, target)?;
            composed.after.entry(anchor).or_default().push(card);
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
    d: &Daemon,
    reader: &Reader<'_>,
    composed: &mut Composed,
    depth: Option<u32>,
    view: View,
    target: &str,
) -> Result<(), Failure> {
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
            if link.kind != "subagent" {
                continue;
            }
            let Ok(child) = SessionId::parse(&link.child) else {
                continue;
            };
            let Some(entry) = link.spawning_call.as_deref().and_then(|call| {
                index.entries[part.start..part.end]
                    .iter()
                    .position(|entry| {
                        entry.kind == EntryKind::Call && entry.call_id.as_deref() == Some(call)
                    })
                    .map(|offset| part.start + offset)
            }) else {
                continue;
            };
            spawned.push((entry, child, link));
        }
        spawned.sort_by_key(|(entry, _, _)| *entry);
        // (part, its index, its spawn level, whether its spawns are placed)
        let mut pieces: Vec<(Part, Arc<CachedIndex>, u32, bool)> = Vec::new();
        let mut cut = part.start;
        for (entry, child, link) in spawned {
            // The first level's cards come from composition; deeper levels
            // place their own.
            if level > 0 {
                let card = card(d, reader, &link, &child, &index, view, target)?;
                composed
                    .after
                    .entry(index.entries[entry].key.clone())
                    .or_default()
                    .push(card);
            }
            if level >= deepest || !seen.insert(child.clone()) {
                continue;
            }
            let Some(child_index) = d.warm.items.get(reader, &child)? else {
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
            let end = child_index.entries.len();
            pieces.push((
                Part {
                    session: child,
                    start: 0,
                    end,
                    inherited: false,
                    joined_by: Some(format!("{}:{}", link.kind, link.how)),
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
fn logical_header(root: &SessionId, composed: &Composed) -> HeaderKind {
    let messages = |inherited: bool| -> u64 {
        composed
            .parts
            .iter()
            .enumerate()
            .filter(|(_, part)| part.inherited == inherited && !spawned(part))
            .map(|(at, part)| {
                composed.entries(at)[part.start..part.end]
                    .iter()
                    .filter(|entry| entry.kind == EntryKind::Header)
                    .count() as u64
            })
            .sum()
    };
    let mut physical: Vec<SessionId> = Vec::new();
    for part in &composed.parts {
        if !part.inherited && !spawned(part) && !physical.contains(&part.session) {
            physical.push(part.session.clone());
        }
    }
    HeaderKind::Logical {
        root: root.clone(),
        physical_sessions: physical,
        own_messages: ValueState::known(messages(false)),
        inherited_messages: ValueState::known(messages(true)),
    }
}

/// The card for a fork or a spawned session: a delegation card for a
/// spawn, a session card for a fork, each with the call that reads it in
/// the reading's view.
fn card(
    d: &Daemon,
    reader: &Reader<'_>,
    link: &polylogue_store::items::SessionLink,
    child: &SessionId,
    parent: &CachedIndex,
    view: View,
    target: &str,
) -> Result<ReadItem, Failure> {
    let reference = format!("session:{child}");
    let open = d
        .ready_call(&command_surface::RunnableRequest {
            operation: "read".into(),
            arguments: serde_json::json!({ "target": reference, "view": view }),
            request_id: format!(
                "read-child-{}",
                blake3::hash(format!("{target}\0{reference}").as_bytes()).to_hex()
            ),
            deadline: command_surface::Deadline::Default,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        })
        .map_err(Failure::from)?;
    let kind = if link.kind == "subagent" {
        let objective = link
            .spawning_call
            .as_deref()
            .and_then(|call| {
                parent.entries.iter().find(|entry| {
                    entry.kind == EntryKind::Call && entry.call_id.as_deref() == Some(call)
                })
            })
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
                session: Box::new(session),
                lineage_complete: ValueState::NotApplicable,
                open: Some(Box::new(open)),
            },
            None => Card::Fields {
                title: format!("{} {reference} (not in the archive)", link.how),
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
    evicted: Vec<Place>,
    context: Option<(bool, Option<String>)>,
    /// A context difference: `evicted` is what left between the two places
    /// and the spans are what entered.
    left: bool,
}

fn runs(composed: &Composed, options: &ReadOptions) -> Result<Runs, Failure> {
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
    let entries = |part: usize| composed.entries(part);
    let last = composed.parts.len() - 1;
    let latest = Place {
        part: last,
        entry: composed.parts[last].end.saturating_sub(1),
    };
    let place_of = |message: &MessageId| -> Result<Place, Failure> {
        find(&composed.parts, entries, &EntryKey::header(message)).ok_or_else(|| {
            Failure::new(
                "not_found",
                format!("message {message} is not in the reading"),
            )
        })
    };
    let at = match &options.at {
        Some(message) => place_of(message)?,
        None => latest,
    };
    let context = context_at(&composed.parts, entries, at);
    let Some(diff) = &options.diff else {
        return Ok(Runs {
            spans: spans(&context.members),
            evicted: context.evicted,
            context: Some((context.complete, context.reason)),
            left: false,
        });
    };
    // The other place: a message, or just before the Nth-last boundary.
    let other = if let Some(rest) = diff.strip_prefix("compaction") {
        let back: usize = match rest.strip_prefix(':') {
            Some(n) => n
                .parse()
                .map_err(|_| Failure::new("bad_request", "compaction:N takes a count"))?,
            None if rest.is_empty() => 1,
            None => {
                return Err(Failure::new(
                    "bad_request",
                    "diff takes a message or compaction[:N]",
                ));
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
            .ok_or_else(|| Failure::new("not_found", "the reading has no such compaction"))?;
        if boundary.entry == composed.parts[boundary.part].start {
            return Err(Failure::new(
                "not_found",
                "the compaction is the first entry",
            ));
        }
        Place {
            part: boundary.part,
            entry: boundary.entry - 1,
        }
    } else {
        place_of(&MessageId::parse(diff).map_err(|e| Failure::new("bad_request", e))?)?
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
        evicted: difference.left,
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

/// A run of shown entries from one session, with their keys.
struct PageEntries {
    part: usize,
    entries: Vec<usize>,
}

pub(super) fn page(
    reader: &Reader<'_>,
    d: &Daemon,
    request: &ReadRequest,
    operation: &Arc<crate::operation::Context>,
    scope: Scope,
) -> Result<ReadPage, Failure> {
    let ReadRequest {
        target,
        options,
        format,
        budget,
        cursor,
        ..
    } = request;
    let format =
        Format::parse(format).ok_or_else(|| Failure::new("bad_request", "unknown read format"))?;
    let mut limits = PageBudget::default();
    if let Some(tokens) = budget {
        limits.tokens = std::num::NonZeroU64::new(*tokens as u64)
            .ok_or_else(|| Failure::new("bad_request", "read budget must be positive"))?;
    }
    let digest = options_digest(&(options, format));
    let key = super::cursor_seal(reader)?;
    let opened = cursor
        .as_deref()
        .map(|text| {
            let opened = PageCursor::open(text, &key).map_err(cursor_failure)?;
            opened.check(target, &digest).map_err(cursor_failure)?;
            Ok::<_, Failure>(opened)
        })
        .transpose()?;
    if options.physical && !target.starts_with("root:") {
        return Err(Failure::new(
            "bad_request",
            "--physical applies to root:REF; a session reference is already one physical session",
        ));
    }
    let scope = Scope {
        logical: scope.logical && !options.physical,
        ..scope
    };
    let composed = compose(d, reader, &scope, options, target)?;
    let read_session = reader
        .session(&scope.session)?
        .ok_or_else(|| Failure::new("not_found", format!("no session {}", scope.session)))?;
    let versions = PageCursor::version(&composed.versions(reader)?);
    if let Some(opened) = &opened
        && opened.rewritten(&versions)
    {
        return Err(Failure::new(
            "stale_cursor",
            format!(
                "the composed reading changed since this cursor was issued; read {target} again"
            ),
        ));
    }
    let runs = runs(&composed, options)?;
    let shown = ShownEntries::new(d, &composed, &runs, FoldProfile::of(options));
    // The first shown entry at or after a key, and whether it is the key.
    let locate = |key: &EntryKey| -> Option<(usize, bool)> {
        let (part, entry) = composed
            .parts
            .iter()
            .zip(&composed.indexes)
            .enumerate()
            .find_map(|(at, (part, index))| {
                index
                    .places
                    .get(key)
                    .filter(|entry| (part.start..part.end).contains(*entry))
                    .map(|entry| (at, *entry))
            })?;
        let at = shown.partition_point(|place| place < (part, entry));
        Some((at, shown.get(at) == Some((part, entry))))
    };
    let tokens = limits.tokens.get();
    let items = limits.items.get();
    let (start, end) = match &opened {
        None if options.latest => (shown.page_before(shown.len(), tokens, items), shown.len()),
        // The first page's lead lines spend its budget first; when they
        // use all of it, the page is those lines alone.
        None => {
            let lead = lead_items(&composed, &runs, options, target, &read_session);
            let lead_tokens: u64 = lead
                .iter()
                .map(|item| {
                    serde_json::to_string(item).map_or(0, |text| estimate_tokens(&text) as u64)
                })
                .sum();
            if !lead.is_empty() && (lead_tokens >= tokens || lead.len() >= items) {
                (0, 0)
            } else {
                (
                    0,
                    shown.page_after(0, tokens - lead_tokens, items - lead.len()),
                )
            }
        }
        Some(opened) => {
            let PagePosition::Entry { key, inside } = &opened.position else {
                return Err(Failure::new(
                    "invalid_query",
                    "the cursor names a row position, not a session entry",
                ));
            };
            let (at, exact) = locate(key).ok_or_else(|| {
                Failure::new(
                    "stale_cursor",
                    "the cursor's entry is no longer in the reading",
                )
            })?;
            match opened.direction {
                // The rest of an entry too large for one page is a page of
                // its own.
                ReadDirection::After if inside.is_some() => {
                    if !exact {
                        return Err(Failure::new(
                            "stale_cursor",
                            "the cursor's entry is no longer shown",
                        ));
                    }
                    (at, at + 1)
                }
                ReadDirection::After => (at, shown.page_after(at, tokens, items)),
                ReadDirection::Before => (shown.page_before(at, tokens, items), at),
                ReadDirection::Past => {
                    let at = at + usize::from(exact);
                    (at, shown.page_after(at, tokens, items))
                }
            }
        }
    };
    let mut header = header(reader, &scope.session, target, options)?;
    if scope.logical {
        header.kind = logical_header(&scope.session, &composed);
    }
    let mut items_out: Vec<ReadItem> = Vec::new();
    // Lead lines open the reading: on its first page, and on a backward
    // page that reaches the start; never again after a lead-only page.
    let continues_forward = opened.as_ref().is_some_and(|opened| {
        matches!(opened.direction, ReadDirection::After | ReadDirection::Past)
    });
    // A forward page at the start that left out the lead lines has them
    // on the page before it.
    let lead_before = start == 0
        && continues_forward
        && !lead_items(&composed, &runs, options, target, &read_session).is_empty();
    if start == 0 && !continues_forward {
        items_out.extend(lead_items(&composed, &runs, options, target, &read_session));
    }
    let selected = shown.range(start..end);
    let mut laid = lay_out(reader, d, &composed, &runs, &selected, options)?;
    decorate(reader, &composed, &shown, start, &mut laid, options)?;
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
    let (start, end) = (start + keep.start, start + keep.end);
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
            let (part, entry) = shown
                .get(start - keep.start + offset)
                .expect("selected entry");
            let kind = match composed.entries(part)[entry].key {
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
            .map(|place| (part, entry, place));
        } else {
            items_out.extend(entry_items);
        }
    }
    let call = |position: usize, direction: ReadDirection| -> Result<ReadyCall, Failure> {
        let (part, entry) = shown.get(position).expect("cursor entry");
        let next = PageCursor {
            target: target.clone(),
            options: digest.clone(),
            versions,
            position: PagePosition::Entry {
                key: composed.entries(part)[entry].key.clone(),
                inside: None,
            },
            direction,
        };
        super::next_call(d, request, next.seal(&key), super::deadline(operation))
    };
    let next = match unfinished {
        Some((part, entry, place)) => {
            let cursor = PageCursor {
                target: target.clone(),
                options: digest.clone(),
                versions,
                position: PagePosition::Entry {
                    key: composed.entries(part)[entry].key.clone(),
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
        None => (end < shown.len())
            .then(|| call(end, ReadDirection::After))
            .transpose()?,
    };
    let footer = ReadFooter {
        estimator: Some(TOKEN_ESTIMATOR.into()),
        follow: (unfinished.is_none() && end == shown.len() && end > 0)
            .then(|| call(end - 1, ReadDirection::Past))
            .transpose()?,
        next,
        previous: ((start > 0 || lead_before) && start < shown.len())
            .then(|| call(start, ReadDirection::Before))
            .transpose()?,
        shown: items_out.len() as u64,
        count: ValueState::Unknown { reason: None },
        ..ReadFooter::default()
    };
    let positions = reader.change_position()?;
    let snapshot = ReadPageSnapshot {
        query: target.clone(),
        ran_at: operation.ran_at,
        caught_up: d.warm.caught_up_for_read(reader, &positions.to_string())?,
        read_at: positions,
        currency: d.currency(reader)?,
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

fn header(
    reader: &Reader<'_>,
    session: &SessionId,
    target: &str,
    options: &ReadOptions,
) -> Result<ReadHeader, Failure> {
    let stored = reader
        .session(session)?
        .ok_or_else(|| Failure::new("not_found", format!("no session {session}")))?;
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
) -> Vec<ReadItem> {
    // The read session's card opens the reading, with whether its lineage
    // composed without a problem.
    let mut out = vec![ReadItem {
        source: Source::record(format!("session:{}", session.id)),
        kind: ItemKind::Card {
            card: Box::new(Card::Session {
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
    let source = Source::record(target);
    for issue in composed.issues.iter().filter(|_| options.inherited) {
        out.push(ReadItem {
            source: source.clone(),
            kind: ItemKind::LinkIssue {
                issue: issue.clone(),
            },
        });
    }
    if !options.inherited && options.view != View::Context {
        let inherited: u64 = composed
            .parts
            .iter()
            .enumerate()
            .filter(|(_, part)| part.inherited)
            .map(|(at, part)| {
                composed.entries(at)[part.start..part.end]
                    .iter()
                    .filter(|entry| entry.kind == EntryKind::Header)
                    .count() as u64
            })
            .sum();
        if let Some(parent) = composed.parts.iter().rev().find(|part| part.inherited) {
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
        for place in &runs.evicted {
            out.push(ReadItem {
                source: Source::record(composed.entries(place.part)[place.entry].key.reference()),
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
) -> Result<Option<ItemPlace>, Failure> {
    let first = from.map_or(0, |place| place.item as usize);
    if first >= items.len() && !items.is_empty() {
        return Err(Failure::new(
            "stale_cursor",
            "the cursor's item is no longer in its entry",
        ));
    }
    let mut used = 0;
    let mut placed = false;
    for (index, item) in items.into_iter().enumerate().skip(first) {
        let encoded =
            polylogue_render::json_bytes(&item).map_err(|e| Failure::new("internal", e))?;
        let text = String::from_utf8(encoded).map_err(|e| Failure::new("internal", e))?;
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
                    return Err(Failure::new(
                        "stale_cursor",
                        "the cursor's item changed since the cursor was issued",
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
            return Err(Failure::new("stale_cursor", "the cursor is past its item"));
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
        let fits = |until: usize| -> Result<bool, Failure> {
            let candidate = polylogue_render::json_bytes(&make(until))
                .map_err(|e| Failure::new("internal", e))?;
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
            .map_err(|e| Failure::new("internal", e))?
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
    d: &Daemon,
    composed: &Composed,
    reading: &Runs,
    page: &[(usize, usize)],
    options: &ReadOptions,
) -> Result<Vec<Vec<ReadItem>>, Failure> {
    // A raw reading is the stored records, which `decorate` adds.
    if options.view == View::Raw {
        return Ok(vec![Vec::new(); page.len()]);
    }
    let mut slots: Vec<Vec<ReadItem>> = vec![Vec::new(); page.len()];
    let mut wanted: std::collections::HashMap<EntryKey, usize> = std::collections::HashMap::new();
    for (slot, (part, entry)) in page.iter().enumerate() {
        wanted.insert(composed.entries(*part)[*entry].key.clone(), slot);
    }
    // The page's entries grouped into runs of one session.
    let mut runs: Vec<PageEntries> = Vec::new();
    for (part, entry) in page {
        match runs.last_mut() {
            Some(run) if run.part == *part => run.entries.push(*entry),
            _ => runs.push(PageEntries {
                part: *part,
                entries: vec![*entry],
            }),
        }
    }
    let selected = |source: &Source| -> Result<bool, Failure> {
        let Some(message) = &source.message else {
            return Ok(false);
        };
        let key = EntryKey::Message {
            message: message.clone(),
            block: source.block,
        };
        Ok(reading.spans.iter().any(|(part, start, end)| {
            composed.indexes[*part]
                .places
                .get(&key)
                .is_some_and(|entry| *entry >= *start && *entry < *end)
        }))
    };
    let ready_attachment = |message: &MessageId, block, index, input: Option<&str>| {
        let arguments = serde_json::to_value(crate::protocol::attachment::Arguments {
            message: message.to_string(),
            block,
            attachment: index,
            input: input.map(str::to_owned),
            length: crate::daemon::attachment::PAGE_BYTES as u64,
        })
        .map_err(|e| Failure::new("internal", e))?;
        let identity = blake3::hash(arguments.to_string().as_bytes());
        d.ready_call(&command_surface::RunnableRequest {
            operation: "attachment".into(),
            arguments,
            request_id: format!("read-attachment-{}", identity.to_hex()),
            deadline: command_surface::Deadline::Default,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        })
        .map_err(Failure::from)
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
    for run in runs {
        let part = &composed.parts[run.part];
        let entries = composed.entries(run.part);
        let inherited = part.inherited;
        let mut messages: Vec<MessageId> = Vec::new();
        for at in &run.entries {
            let entry = &entries[*at];
            match &entry.key {
                EntryKey::Message { message, .. } => {
                    if messages.last() != Some(message) {
                        messages.push(message.clone());
                    }
                }
                EntryKey::Event { event } => {
                    let record = reader.event_record(event)?.ok_or_else(|| {
                        Failure::new("stale_cursor", format!("event {event} is gone"))
                    })?;
                    emit(
                        reader,
                        Record::Event(record),
                        ProjectionEvidence::default(),
                        options,
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
            Failure::new("stale_cursor", format!("message {} is gone", messages[0]))
        })?;
        let wanted_messages: HashSet<&MessageId> = messages.iter().collect();
        let last_message = messages.last().expect("one message").clone();
        let mut result = Ok(());
        selection.visit_from_with_record_evidence(
            start,
            |message, projection, commit, order| {
                let id = message.id.clone();
                if wanted_messages.contains(&id) {
                    let step = (|| {
                        let commit = commit.ok_or_else(|| {
                            Failure::new(
                                "not_yet_available",
                                format!("message:{id} has no retained source admission evidence"),
                            )
                        })?;
                        let record = message_record(reader, message, projection, commit, order)?;
                        let evidence = ProjectionEvidence {
                            inherited,
                            ..Default::default()
                        };
                        let mut last = None;
                        emit(
                            reader,
                            record,
                            evidence,
                            options,
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
    start: usize,
    laid: &mut [Vec<ReadItem>],
    options: &ReadOptions,
) -> Result<(), Failure> {
    let human_prompt = |entry: &IndexEntry| {
        entry.kind == EntryKind::Header
            && entry.written_by == polylogue_model::WrittenBy::Human
            && entry.prompt_kind == Some(polylogue_model::PromptKind::Prose)
    };
    let mut placed = place_cards(composed, shown, start, laid.len(), options);
    for (offset, items) in laid.iter_mut().enumerate() {
        let (part, at) = shown.get(start + offset).expect("decorated entry");
        let entries = composed.entries(part);
        let entry = &entries[at];
        let mut before: Vec<ReadItem> =
            composed.before.get(&entry.key).cloned().unwrap_or_default();
        if options.turns && human_prompt(entry) {
            // The turn's number and its calls, up to the next prompt.
            // Counted over the composed reading, within each part's bounds.
            // An inline spawned session numbers its own turns.
            let bounds = &composed.parts[part];
            let same = composed.same_context(part);
            let earlier: usize = composed.parts[..part]
                .iter()
                .enumerate()
                .filter(|(_, other)| same(other))
                .map(|(index, other)| {
                    composed.entries(index)[other.start..other.end]
                        .iter()
                        .filter(|entry| human_prompt(entry))
                        .count()
                })
                .sum();
            let number = earlier
                + entries[bounds.start..at]
                    .iter()
                    .filter(|entry| human_prompt(entry))
                    .count()
                + 1;
            // Up to the next prompt, across the pieces of a part split at a
            // spawning call.
            let calls = entries[at + 1..bounds.end]
                .iter()
                .chain(
                    composed.parts[part + 1..]
                        .iter()
                        .enumerate()
                        .filter(|(_, other)| other.session == bounds.session)
                        .flat_map(|(offset, other)| {
                            composed.entries(part + 1 + offset)[other.start..other.end].iter()
                        }),
                )
                .take_while(|entry| !human_prompt(entry))
                .filter(|entry| entry.kind == EntryKind::Call)
                .count();
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
            let meter = context_meter(
                &composed.context_parts(part),
                |part| composed.entries(part),
                Place { part, entry: at },
            );
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
                    Failure::new(
                        "not_yet_available",
                        format!("{} has no stored record", entry.key.reference()),
                    )
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

/// Cards for each page slot, placed after their anchor entry when it is
/// shown, or before the next shown entry when the reading hides the anchor
/// (a spawning call in the dialogue view), so a card is never lost.
fn place_cards(
    composed: &Composed,
    shown: &ShownEntries,
    start: usize,
    slots: usize,
    options: &ReadOptions,
) -> std::collections::HashMap<usize, (Vec<ReadItem>, Vec<ReadItem>)> {
    let mut placed: std::collections::HashMap<usize, (Vec<ReadItem>, Vec<ReadItem>)> =
        std::collections::HashMap::new();
    for (key, cards) in &composed.after {
        let Some(place) = composed
            .parts
            .iter()
            .enumerate()
            .find_map(|(part, bounds)| {
                composed.indexes[part]
                    .places
                    .get(key)
                    .filter(|entry| (bounds.start..bounds.end).contains(*entry))
                    .map(|entry| (part, *entry))
            })
        else {
            continue;
        };
        let at = shown.partition_point(|shown| shown < place);
        if at < start || at >= start + slots {
            continue;
        }
        let cards = cards.iter().filter(|item| {
            options.spawns != SpawnFold::Hidden
                || !matches!(&item.kind, ItemKind::Card { card } if matches!(**card, Card::Delegation { .. }))
        });
        let slot = placed.entry(at - start).or_default();
        if shown.get(at) == Some(place) {
            slot.1.extend(cards.cloned());
        } else {
            slot.0.extend(cards.cloned());
        }
    }
    placed
}

#[allow(clippy::too_many_arguments)]
fn emit(
    reader: &Reader<'_>,
    record: Record,
    evidence: ProjectionEvidence,
    options: &ReadOptions,
    selected: &dyn Fn(&Source) -> Result<bool, Failure>,
    ready_attachment: &super::attachment_evidence::ReadyAttachment<'_>,
    footer: &mut ReadFooter,
    out: &mut dyn FnMut(ReadItem),
) -> Result<(), Failure> {
    // A context reading's entries were chosen from the index as members, so
    // each one laid out is selected; what it leaves out is said once, in
    // the reading's first lines.
    let mut evidence = evidence;
    if options.view == View::Context {
        evidence.context = Some(ContextEvidence {
            selected: true,
            complete: true,
            reason: None,
            evicted: Vec::new(),
            not_recorded: Vec::new(),
        });
    }
    let Some(record) = output::project(
        SelectedRecord::Canonical(Box::new(record)),
        options,
        &evidence,
        false,
        footer,
    )?
    else {
        return Ok(());
    };
    let layout = evidence::Layout {
        options,
        evidence: &evidence,
        selected,
        ready_attachment,
        backwards: false,
        anchor: None,
    };
    evidence::visit_parts(reader, &record, &layout, |item, _| {
        out(item);
        Ok(true)
    })?;
    Ok(())
}

/// One node of a links tree, in tree order.
struct Node {
    session: String,
    parents: Vec<String>,
    /// The tree link that reached this node: kind and how.
    joined: Option<(String, String)>,
    /// Links to a node already shown: drawn once, never expanded again.
    cross: Vec<(String, String)>,
    /// `reference` links from this node, which never join the tree.
    references: Vec<(String, String)>,
}

/// Every session linked to the seed, as a tree from the roots of its
/// logical session: each node once in depth-first order, a node reached
/// again as a cross-link, `reference` links as cross-links that are not part
/// of the tree. Only link rows are read. The walk is iterative.
fn tree(reader: &Reader<'_>, seed: &SessionId) -> Result<Vec<Node>, Failure> {
    let mut family: Vec<String> = vec![seed.to_string()];
    family.extend(reader.lineage(seed, polylogue_store::items::Lineage::Ancestors)?);
    let mut roots = Vec::new();
    for member in &family {
        let Ok(id) = SessionId::parse(member) else {
            roots.push(member.clone());
            continue;
        };
        if reader
            .parent_links(&id)?
            .iter()
            .all(|link| link.kind == "reference")
        {
            roots.push(member.clone());
        }
    }
    if roots.is_empty() {
        // Every member has a parent: the family is a cycle; start at the seed.
        roots.push(seed.to_string());
    }
    let mut shown: HashSet<String> = HashSet::new();
    let mut nodes = Vec::new();
    for root in roots {
        type ParentEdge = (String, String, String);
        let mut stack: Vec<(String, Option<ParentEdge>)> = vec![(root, None)];
        while let Some((session, reached)) = stack.pop() {
            let parent = reached.as_ref().map(|(parent, _, _)| parent.clone());
            if !shown.insert(session.clone()) {
                if let (Some(parent), Some(node)) = (
                    parent,
                    nodes
                        .iter_mut()
                        .find(|node: &&mut Node| node.session == session),
                ) {
                    node.cross.push((parent, session.clone()));
                }
                continue;
            }
            let mut node = Node {
                session: session.clone(),
                parents: parent.into_iter().collect(),
                joined: reached.map(|(_, kind, how)| (kind, how)),
                cross: Vec::new(),
                references: Vec::new(),
            };
            let children = match SessionId::parse(&session) {
                Ok(id) => reader.child_links(&id)?,
                Err(_) => Vec::new(),
            };
            let mut next = Vec::new();
            for link in children {
                if link.kind == "reference" {
                    node.references.push((link.child, link.how));
                } else {
                    next.push((link.child, Some((session.clone(), link.kind, link.how))));
                }
            }
            // Depth first, children in link order.
            stack.extend(next.into_iter().rev());
            nodes.push(node);
        }
    }
    Ok(nodes)
}

pub(super) fn links_page(
    reader: &Reader<'_>,
    d: &Daemon,
    request: &ReadRequest,
    operation: &Arc<crate::operation::Context>,
    scope: Scope,
) -> Result<ReadPage, Failure> {
    let ReadRequest {
        target,
        options,
        format,
        budget,
        cursor,
        ..
    } = request;
    let format =
        Format::parse(format).ok_or_else(|| Failure::new("bad_request", "unknown read format"))?;
    let mut limits = PageBudget::default();
    if let Some(tokens) = budget {
        limits.tokens = std::num::NonZeroU64::new(*tokens as u64)
            .ok_or_else(|| Failure::new("bad_request", "read budget must be positive"))?;
    }
    let digest = options_digest(&(options, format));
    let key = super::cursor_seal(reader)?;
    let nodes = tree(reader, &scope.session)?;
    let start = match cursor.as_deref() {
        None => 0,
        Some(text) => {
            let opened = PageCursor::open(text, &key).map_err(cursor_failure)?;
            opened.check(target, &digest).map_err(cursor_failure)?;
            let PagePosition::Node { session } = &opened.position else {
                return Err(Failure::new(
                    "invalid_query",
                    "the cursor names no links node",
                ));
            };
            nodes
                .iter()
                .position(|node| node.session == session.to_string())
                .ok_or_else(|| {
                    Failure::new("stale_cursor", "the cursor's node is no longer linked")
                })?
        }
    };
    let header = header(reader, &scope.session, target, options)?;
    let mut items = Vec::new();
    let mut used = serde_json::to_vec(&header).map_or(0, |bytes| bytes.len());
    let mut end = start;
    for node in &nodes[start..] {
        let mut node_items = Vec::new();
        let reference = format!("session:{}", node.session);
        let id = SessionId::parse(&node.session).ok();
        let stored = match &id {
            Some(id) => reader.session(id)?.is_some(),
            None => false,
        };
        let title = match &id {
            Some(id) if stored => reader
                .display_title(id)?
                .map(ValueState::known)
                .unwrap_or_default(),
            _ => ValueState::Unavailable {
                reason: "not in the archive".into(),
            },
        };
        let own = match &id {
            Some(id) if stored => reader
                .own_message_count(id)?
                .map_or(ValueState::Unknown { reason: None }, ValueState::known),
            _ => ValueState::Unavailable {
                reason: "not in the archive".into(),
            },
        };
        node_items.push(ReadItem {
            source: Source::record(&reference),
            kind: ItemKind::Card {
                card: Box::new(Card::LinkNode {
                    reference: reference.clone(),
                    title,
                    parents: node
                        .parents
                        .iter()
                        .map(|parent| format!("session:{parent}"))
                        .collect(),
                    own_messages: own,
                    inherited_messages: ValueState::Unknown { reason: None },
                    own_cost: ValueState::Unknown { reason: None },
                    tree_cost: ValueState::Unknown { reason: None },
                    seed: node.session == scope.session.to_string(),
                    open: None,
                }),
            },
        });
        // The tree link that reached the node, then links to nodes already
        // shown and `reference` links, as cross-links.
        let joined = node
            .parents
            .first()
            .zip(node.joined.as_ref())
            .map(|(parent, (kind, how))| {
                (
                    parent.clone(),
                    node.session.clone(),
                    kind.as_str(),
                    Some(how.clone()),
                    false,
                )
            });
        for (from, to, relation, how, cross) in joined
            .into_iter()
            .chain(
                node.cross
                    .iter()
                    .map(|(from, to)| (from.clone(), to.clone(), "tree", None, true)),
            )
            .chain(node.references.iter().map(|(to, how)| {
                (
                    node.session.clone(),
                    to.clone(),
                    "reference",
                    Some(how.clone()),
                    true,
                )
            }))
        {
            node_items.push(ReadItem {
                source: Source::record(&reference),
                kind: ItemKind::Card {
                    card: Box::new(Card::LinkEdge {
                        from: format!("session:{from}"),
                        to: format!("session:{to}"),
                        relation: relation.into(),
                        how,
                        cross_link: cross,
                    }),
                },
            });
        }
        let size: usize = node_items
            .iter()
            .map(|item| serde_json::to_vec(item).map_or(0, |bytes| bytes.len()))
            .sum();
        if end > start && (used + size > limits.bytes.get() || end - start >= limits.items.get()) {
            break;
        }
        used += size;
        items.extend(node_items);
        end += 1;
    }
    let call = |at: usize, direction: ReadDirection| -> Result<ReadyCall, Failure> {
        let session = SessionId::parse(&nodes[at].session)
            .map_err(|e| Failure::new("not_yet_available", e))?;
        let next = PageCursor {
            target: target.clone(),
            options: digest.clone(),
            versions: PageCursor::version(&[]),
            position: PagePosition::Node { session },
            direction,
        };
        super::next_call(d, request, next.seal(&key), super::deadline(operation))
    };
    let previous_start = start.saturating_sub(limits.items.get());
    let footer = ReadFooter {
        estimator: Some(TOKEN_ESTIMATOR.into()),
        next: (end < nodes.len())
            .then(|| call(end, ReadDirection::After))
            .transpose()?,
        previous: (start > 0)
            .then(|| call(previous_start, ReadDirection::After))
            .transpose()?,
        shown: items.len() as u64,
        count: ValueState::known(nodes.len() as u64),
        ..ReadFooter::default()
    };
    let positions = reader.change_position()?;
    let snapshot = ReadPageSnapshot {
        query: target.clone(),
        ran_at: operation.ran_at,
        caught_up: d.warm.caught_up_for_read(reader, &positions.to_string())?,
        read_at: positions,
        currency: d.currency(reader)?,
    };
    Ok(ReadPage {
        format,
        snapshot,
        records: vec![],
        document: ReadDocument {
            header,
            items,
            footer,
        },
    })
}
