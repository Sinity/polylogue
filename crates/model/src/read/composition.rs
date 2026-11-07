//! A composed reading is a list of parts, each a run of one physical
//! session's index entries; nothing is copied. The context at a message is a
//! function over the same parts: the newest compaction boundary before it is
//! chosen first and then checked, and an incomplete newest boundary is
//! reported, never replaced by an older one. Tool results a later
//! microcompaction names are then removed.
use super::{Boundary, EntryKey, EntryKind, IndexEntry};
use crate::{MessageId, SessionId};

/// A run of one session's entries, `start..end`, in a composed reading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Part {
    pub session: SessionId,
    pub start: usize,
    pub end: usize,
    /// Messages the read session did not write: a parent's prefix.
    pub inherited: bool,
    /// For a logical reading, the link that brought this session in.
    pub joined_by: Option<String>,
}

/// The end (exclusive) of a parent's prefix through its branch point: after
/// the branch point message's last entry, before the next message.
pub fn prefix_through(entries: &[IndexEntry], branch_point: &MessageId) -> Option<usize> {
    let at = entries.iter().position(|entry| {
        entry.kind == EntryKind::Header && entry.key.message() == Some(branch_point)
    })?;
    Some(
        entries[at + 1..]
            .iter()
            .position(|entry| entry.kind == EntryKind::Header)
            .map_or(entries.len(), |offset| at + 1 + offset),
    )
}

/// A place in a composed reading: the part and the entry index in that
/// part's session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Place {
    pub part: usize,
    pub entry: usize,
}

/// What the model had been given at a place.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ContextReading {
    /// Entries in context, in the order the model received them.
    pub members: Vec<Place>,
    /// Messages the applied boundary removed, by their header entries.
    pub evicted: Vec<Place>,
    /// The boundary applied, when there is one.
    pub boundary: Option<Place>,
    pub complete: bool,
    pub reason: Option<String>,
}

/// The context at `at` (inclusive). `entries(part)` gives a part's
/// session entries.
pub fn context_at<'a>(
    parts: &[Part],
    entries: impl Fn(usize) -> &'a [IndexEntry],
    at: Place,
) -> ContextReading {
    let mut places = Vec::new();
    for (index, part) in parts.iter().enumerate().take(at.part + 1) {
        let end = if index == at.part {
            (at.entry + 1).min(part.end)
        } else {
            part.end
        };
        places.extend((part.start..end).map(|entry| Place { part: index, entry }));
    }
    let boundary = places
        .iter()
        .rposition(|place| entries(place.part)[place.entry].boundary == Some(Boundary::Compaction));
    let Some(boundary_at) = boundary else {
        return ContextReading {
            members: without_microcompacted(places, &entries),
            evicted: Vec::new(),
            boundary: None,
            complete: true,
            reason: None,
        };
    };
    let boundary_place = places[boundary_at];
    let evicted = places[..boundary_at]
        .iter()
        .copied()
        .filter(|place| entries(place.part)[place.entry].kind == EntryKind::Header)
        .collect();
    let after = &places[boundary_at + 1..];
    let summary = after.iter().find(|place| {
        entries(place.part)[place.entry].boundary == Some(Boundary::CompactionSummary)
    });
    let complete = summary.is_some();
    ContextReading {
        members: without_microcompacted(after.to_vec(), &entries),
        evicted,
        boundary: Some(boundary_place),
        complete,
        reason: (!complete).then(|| "the newest compaction's summary is not recorded".into()),
    }
}

/// Members less the tool results that a later microcompaction among them
/// removed. A microcompaction clears only results that came before it.
fn without_microcompacted<'a>(
    members: Vec<Place>,
    entries: &impl Fn(usize) -> &'a [IndexEntry],
) -> Vec<Place> {
    let mut cleared: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut keep = vec![true; members.len()];
    for (at, place) in members.iter().enumerate().rev() {
        let entry = &entries(place.part)[place.entry];
        if entry.boundary == Some(Boundary::Microcompaction) {
            cleared.extend(entry.cleared.iter().map(String::as_str));
        } else if entry.kind == EntryKind::Result
            && entry
                .call_id
                .as_deref()
                .is_some_and(|call| cleared.contains(call))
        {
            keep[at] = false;
        }
    }
    members
        .into_iter()
        .zip(keep)
        .filter_map(|(place, keep)| keep.then_some(place))
        .collect()
}

/// The two halves of a context difference between two places.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ContextDifference {
    /// In context at the earlier place and not at the later one.
    pub left: Vec<Place>,
    /// In context at the later place and not at the earlier one.
    pub entered: Vec<Place>,
}

pub fn context_difference(earlier: &ContextReading, later: &ContextReading) -> ContextDifference {
    let later_set: std::collections::HashSet<_> = later.members.iter().collect();
    let earlier_set: std::collections::HashSet<_> = earlier.members.iter().collect();
    ContextDifference {
        left: earlier
            .members
            .iter()
            .filter(|place| !later_set.contains(place))
            .copied()
            .collect(),
        entered: later
            .members
            .iter()
            .filter(|place| !earlier_set.contains(place))
            .copied()
            .collect(),
    }
}

/// The context meter of a model message: the provider-reported input of the
/// response, or else the running estimate since the last boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextMeter {
    Reported(u64),
    Estimated(u64),
    Unknown,
}

pub fn context_meter<'a>(
    parts: &[Part],
    entries: impl Fn(usize) -> &'a [IndexEntry] + Copy,
    at: Place,
) -> ContextMeter {
    let entry = &entries(at.part)[at.entry];
    if let Some(tokens) = entry.context_tokens {
        return ContextMeter::Reported(tokens);
    }
    let context = context_at(parts, entries, at);
    let estimate: u64 = context
        .members
        .iter()
        .map(|place| entries(place.part)[place.entry].tokens_full)
        .sum();
    if estimate == 0 {
        ContextMeter::Unknown
    } else {
        ContextMeter::Estimated(estimate)
    }
}

/// Where a key sits in a composed reading.
pub fn find<'a>(
    parts: &[Part],
    entries: impl Fn(usize) -> &'a [IndexEntry],
    key: &EntryKey,
) -> Option<Place> {
    parts.iter().enumerate().find_map(|(index, part)| {
        entries(index)[part.start..part.end]
            .iter()
            .position(|entry| &entry.key == key)
            .map(|offset| Place {
                part: index,
                entry: part.start + offset,
            })
    })
}

#[cfg(test)]
mod tests;
