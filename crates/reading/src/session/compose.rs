//! Composing a session reading: the parts of physical sessions it reads (an
//! inherited prefix, the session itself, joined continuations and inline
//! spawned sessions), the cards placed after entries, and the runs of
//! entries the view shows.
use super::{Scope, still_wanted};
use crate::Error;
use polylogue_model::Reason;
use polylogue_model::{MessageId, SessionId, read::*};
use polylogue_store::Reader;
use polylogue_store::items::ItemVersions;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// What a reading needs of one composed session before reading its
/// entries: where its index is, its counters, and its counts.
pub(super) struct SessionIndex {
    pub pk: i64,
    pub versions: ItemVersions,
    pub entries: u64,
    pub counts: ItemCounts,
}

/// A session's index, or `None` when the session is not stored.
pub(super) fn session_index(
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

/// A card shown after one entry: a fork at its branch point, a spawned
/// session at its call. Composition finds where every card goes; the card
/// itself is built for the page that shows it, since a session can have
/// thousands of children.
pub(super) struct CardAfterEntry {
    pub session: SessionId,
    pub seq: usize,
    pub link: polylogue_store::lineage::SessionLink,
    pub child: SessionId,
}

/// One composed session reading: its parts, each with the index of its
/// session, and the link problems found while composing.
pub(super) struct Composed {
    pub parts: Vec<Part>,
    pub indexes: Vec<Arc<SessionIndex>>,
    pub issues: Vec<ParentIssue>,
    /// A copied prefix whose source session is not stored.
    pub unstored_copy: bool,
    /// Lines shown before an entry (a `continues` line at a continuation).
    pub before: HashMap<EntryKey, Vec<ReadItem>>,
    pub after: Vec<CardAfterEntry>,
}
/// A part that reads a spawned session inline.
pub(super) fn spawned(part: &Part) -> bool {
    part.joined_by.as_deref() == Some("spawn")
}

impl Composed {
    /// The counts of a part's session before `seq`.
    pub(super) fn counts_at(
        &self,
        reader: &Reader<'_>,
        part: usize,
        seq: usize,
    ) -> Result<ItemCounts, Error> {
        super::entries::counts_before(reader, &self.indexes[part], seq)
    }
    /// The entries of a part's session in `start..end` whose class `which`
    /// accepts.
    pub(super) fn count(
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
    pub(super) fn entry(
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
    pub(super) fn same_context(&self, part: usize) -> impl Fn(&Part) -> bool + '_ {
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
    pub(super) fn context_parts(&self, part: usize) -> Vec<Part> {
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
    pub(super) fn versions(&self, reader: &Reader<'_>) -> Result<Vec<SessionVersion>, Error> {
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

/// Where a session's index holds the spawning call of each spawn link that
/// names one, by call ID. A session can have thousands of spawn links, so
/// they are looked up together.
fn spawning_calls(
    reader: &Reader<'_>,
    index: &SessionIndex,
    links: &[polylogue_store::lineage::SessionLink],
) -> Result<HashMap<String, u64>, Error> {
    let calls: Vec<&str> = links
        .iter()
        .filter(|link| link.kind == "spawn")
        .filter_map(|link| link.spawning_call.as_deref())
        .collect();
    Ok(reader.item_call_places(index.pk, &calls)?)
}

/// The parent prefix of `session` through its branch point, as parts of
/// parent sessions, all marked inherited. The walk is iterative over
/// sessions already on the path; problems are reported, never resolved by
/// choosing one link.
pub(super) fn prefix(
    reader: &Reader<'_>,
    session: &SessionId,
    composed: &mut Composed,
) -> Result<(), Error> {
    let mut chain: Vec<(SessionId, Option<String>, bool)> = Vec::new();
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
        chain.push((
            parent.clone(),
            link.branch_point.clone(),
            link.copies_prefix,
        ));
        current = parent;
    }
    // Compose from the oldest ancestor down, cutting each at the branch
    // point its child continues from.
    let mut parts: Vec<(Part, Arc<SessionIndex>)> = Vec::new();
    for (at, (parent, branch_point, copies_prefix)) in chain.iter().enumerate().rev() {
        let child = if at == 0 {
            session.clone()
        } else {
            chain[at - 1].0.clone()
        };
        let Some(index) = session_index(reader, parent)? else {
            // A copied prefix whose source is not stored is not a broken
            // link: the child's messages are its own.
            if *copies_prefix {
                composed.unstored_copy = true;
            } else {
                composed.issues.push(ParentIssue::MissingParent {
                    child,
                    parent: parent.clone(),
                });
            }
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

/// Composes the reading. Reading a due session's lines waits for its
/// summary and starts a newer read ([`Reader::restarts`]); then every
/// session is composed again, so that all its parts come from one read.
pub(super) fn compose(
    reader: &Reader<'_>,
    scope: &Scope,
    options: &ReadOptions,
) -> Result<Composed, Error> {
    loop {
        let restarts = reader.restarts();
        let composed = compose_once(reader, scope, options)?;
        if reader.restarts() == restarts {
            return Ok(composed);
        }
    }
}

fn compose_once(
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
        unstored_copy: false,
        before: Default::default(),
        after: Vec::new(),
    };
    // The prefix is composed always: hidden, it is one `continues` line.
    let context = options.view == View::Context && scope.has_context_anchor() && around == 0;
    if scope.range.is_none() || context {
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
                return Err(Reason::RangeEndsBeforeStart {}.into());
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
                if context { 0 } else { start },
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
    children(reader, scope, &mut composed)?;
    if let SpawnDisplay::Inline { depth } = options.spawns() {
        inline_spawns(reader, &mut composed, depth)?;
    }
    Ok(composed)
}

/// Continuations and resumes follow their parent in a logical reading,
/// each after a `continues` line; forks and spawned sessions appear as cards
/// at their branch point or spawning call. The walk is iterative and visits
/// each session once.
fn children(reader: &Reader<'_>, scope: &Scope, composed: &mut Composed) -> Result<(), Error> {
    let mut visited: HashSet<SessionId> = HashSet::from([scope.session.clone()]);
    let mut at = composed.parts.len() - 1;
    while at < composed.parts.len() {
        let session = composed.parts[at].session.clone();
        let index = Arc::clone(&composed.indexes[at]);
        at += 1;
        let links = reader.child_links(&session)?;
        let calls = spawning_calls(reader, &index, &links)?;
        for link in links {
            still_wanted()?;
            if link.is_reference() {
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
            let slot = match link.kind.as_str() {
                "spawn" => link
                    .spawning_call
                    .as_deref()
                    .and_then(|call| calls.get(call))
                    .map(|seq| *seq as usize),
                _ => {
                    let point = link
                        .branch_point
                        .as_deref()
                        .and_then(|point| MessageId::parse(point).ok());
                    let end = match point {
                        Some(point) => reader.item_message_span(index.pk, &point)?,
                        None => None,
                    };
                    end.and_then(|(_, end)| end.checked_sub(1))
                        .map(|seq| seq as usize)
                }
            };
            let Some(slot) = slot else { continue };
            composed.after.push(CardAfterEntry {
                session: session.clone(),
                seq: slot,
                link,
                child,
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
    reader: &Reader<'_>,
    composed: &mut Composed,
    depth: Option<u32>,
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
        let links = reader.child_links(&part.session)?;
        let calls = spawning_calls(reader, &index, &links)?;
        for link in links {
            still_wanted()?;
            if link.kind != "spawn" {
                continue;
            }
            let Ok(child) = SessionId::parse(&link.child) else {
                continue;
            };
            let Some(entry) = link
                .spawning_call
                .as_deref()
                .and_then(|call| calls.get(call))
                .map(|seq| *seq as usize)
                .filter(|seq| (part.start..part.end).contains(seq))
            else {
                continue;
            };
            spawned.push((entry, child, link));
        }
        spawned.sort_by_key(|(entry, _, _)| *entry);
        // (part, its index, its spawn level, whether its spawns are placed)
        let mut pieces: Vec<(Part, Arc<SessionIndex>, u32, bool)> = Vec::new();
        let mut cut = part.start;
        for (entry, child, link) in spawned {
            still_wanted()?;
            // The first level's cards come from composition; deeper levels
            // place their own.
            if level > 0 {
                composed.after.push(CardAfterEntry {
                    session: part.session.clone(),
                    seq: entry,
                    link: link.clone(),
                    child: child.clone(),
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
pub(super) fn logical_header(
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

/// The runs of entries a reading shows, as segments over the parts.
pub(super) struct Runs {
    /// (part, first entry, end entry) for each segment.
    pub spans: Vec<(usize, usize, usize)>,
    pub evicted: Vec<EntryKey>,
    pub context: Option<(bool, Option<String>)>,
    /// A context difference: `evicted` is what left between the two places
    /// and the spans are what entered.
    pub left: bool,
}

pub(super) fn runs(
    reader: &Reader<'_>,
    composed: &Composed,
    options: &ReadOptions,
    show_inherited: bool,
) -> Result<Runs, Error> {
    if options.view != View::Context || composed.parts.iter().all(|part| part.start == part.end) {
        return Ok(Runs {
            spans: composed
                .parts
                .iter()
                .enumerate()
                .filter(|(_, part)| show_inherited || !part.inherited)
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
    // The composition ends at the selected message, or the session's latest.
    let at = Place {
        part: last,
        entry: composed.parts[last].end.saturating_sub(1),
    };
    let place_of = |message: &MessageId| -> Result<Place, Error> {
        find(&composed.parts, entries, &EntryKey::header(message))
            .ok_or_else(|| Error::not_found(format!("message {message} is not in the reading")))
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
                .map_err(|_| Reason::CompactionCount { text: n.to_owned() })?,
            None if rest.is_empty() => 1,
            None => {
                return Err(Reason::DiffTarget {}.into());
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
        place_of(
            &MessageId::parse(diff).map_err(|error| Reason::BadReference {
                text: diff.to_owned(),
                problem: error.to_string(),
            })?,
        )?
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
