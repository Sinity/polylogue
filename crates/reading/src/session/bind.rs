//! Binding a composed reading to its accepted start: the seen marks of
//! each physical owner, a historical context's message, the cut at the
//! accepted mark, and the marks a displayed page leaves.
use super::compose::{Composed, spawned};
use super::entries::PageEntry;
use crate::{Error, ReadRequest};
use polylogue_model::Reason;
use polylogue_model::{Reference, SessionId, read::*};
use polylogue_store::Reader;
use std::collections::HashSet;

/// A historical context has two coordinates: its message and its annotation
/// cutoff. Keep the first in the existing replay selector, so appending a later
/// message cannot move the context while a client follows its pages.
pub(super) fn bind_context(
    reader: &Reader<'_>,
    request: &ReadRequest,
    composed: &mut Composed,
) -> Result<Option<ReadRequest>, Error> {
    let anchor = match Reference::parse(&request.target).ok() {
        Some(Reference::Message(message)) => Some(message),
        _ => None,
    };
    for part in (0..composed.parts.len()).rev() {
        let range = &composed.parts[part];
        let index = &composed.indexes[part];
        let header = if let Some(anchor) = &anchor {
            let Some(seq) = reader.item_place(index.pk, &EntryKey::header(anchor))? else {
                continue;
            };
            if !(range.start as u64..range.end as u64).contains(&seq) {
                continue;
            }
            reader
                .item_window(index.pk, seq..seq + 1, true, 1, EntryClass::is_header)?
                .pop()
        } else {
            reader
                .item_window(
                    index.pk,
                    range.start as u64..range.end as u64,
                    false,
                    1,
                    EntryClass::is_header,
                )?
                .pop()
        };
        let Some(header) = header else {
            continue;
        };
        let Some(message) = header.entry.key.message().cloned() else {
            return Err(Error::internal(
                "a context header has no message coordinate",
            ));
        };
        let end = reader.item_unit_end(index.pk, header.seq)? as usize;
        composed.parts.truncate(part + 1);
        composed.indexes.truncate(part + 1);
        composed.parts[part].end = end;
        let mut bound = request.clone();
        let reference = Reference::Message(message);
        bound.target = reference.to_string();
        bound.references = Some(vec![reference]);
        return Ok(Some(bound));
    }
    Ok(None)
}

pub(super) fn session_read(request: &ReadRequest) -> Option<&SessionRead> {
    match &request.start {
        crate::ReadStart::Bound(bound) => Some(bound),
        crate::ReadStart::Beginning | crate::ReadStart::Seen => None,
    }
}

/// Snapshot each physical owner's mark once. Root and continuation content
/// remains independently unread even when a later part has already been read.
pub(super) fn bind_seen(
    reader: &Reader<'_>,
    mut request: ReadRequest,
    operation: &crate::operation::Context,
    composed: &Composed,
) -> Result<ReadRequest, Error> {
    if request.cursor.is_none()
        && let crate::ReadStart::Bound(bound) = &mut request.start
        && bound.logical
    {
        let mut seen = HashSet::new();
        let owners: Vec<_> = composed
            .parts
            .iter()
            .filter(|part| !spawned(part))
            .map(|part| {
                if part.inherited {
                    &bound.session
                } else {
                    &part.session
                }
            })
            .filter(|session| seen.insert((*session).clone()))
            .cloned()
            .collect();
        let targets: Vec<_> = owners.iter().cloned().map(Reference::Session).collect();
        let mut marks = std::collections::HashMap::new();
        if bound.from_seen {
            for mark in reader.seen_marks(&operation.authorship.author.to_string(), &targets)? {
                if marks.insert(mark.target.clone(), mark.item).is_some() {
                    return Err(Error::conflict(format!(
                        "multiple current seen marks for {}",
                        mark.target
                    )));
                }
            }
        }
        bound.parts = owners
            .into_iter()
            .map(|session| {
                let after = marks
                    .remove(&Reference::Session(session.clone()).to_string())
                    .flatten();
                SessionSeen { session, after }
            })
            .collect();
    }
    Ok(request)
}

/// Cut the existing composition at the accepted message/block, preserving
/// its source order and membership; pagination then uses its normal index.
pub(super) fn trim_start(
    reader: &Reader<'_>,
    request: &ReadRequest,
    composed: &mut Composed,
) -> Result<(), Error> {
    let Some(bound) = session_read(request) else {
        return Ok(());
    };
    if bound.logical {
        for owner in &bound.parts {
            trim_owner(
                reader,
                composed,
                &bound.session,
                &owner.session,
                owner.after.as_deref(),
                true,
            )?;
        }
        return Ok(());
    }
    trim_owner(
        reader,
        composed,
        &bound.session,
        &bound.session,
        bound.after.as_deref(),
        false,
    )
}

fn trim_owner(
    reader: &Reader<'_>,
    composed: &mut Composed,
    root: &SessionId,
    owner: &SessionId,
    after: Option<&str>,
    logical: bool,
) -> Result<(), Error> {
    let Some(after) = after else {
        return Ok(());
    };
    // Inline children travel inside their surrounding owner's transcript.
    // They never supply its mark, but a later parent mark skips them too.
    let mut current = root;
    let owned: Vec<_> = composed
        .parts
        .iter()
        .map(|part| {
            if !spawned(part) {
                current = if part.inherited { root } else { &part.session };
            }
            !logical || current == owner
        })
        .collect();
    let (key, whole_message) =
        match Reference::parse(after).map_err(|error| Reason::BadReference {
            text: after.to_owned(),
            problem: error.to_string(),
        })? {
            Reference::Message(message) => (EntryKey::header(&message), true),
            Reference::Block { message, n } => (
                EntryKey::Message {
                    message,
                    block: Some(n),
                },
                false,
            ),
            _ => return Err(Error::conflict("the seen mark is not a message or block")),
        };
    // Inline children can split one physical owner into several parts.
    // Find the last such segment beginning no later than the accepted item.
    for at in (0..composed.parts.len()).rev() {
        if !owned[at] || logical && spawned(&composed.parts[at]) {
            continue;
        }
        let index = &composed.indexes[at];
        if let Some(seq) = reader.item_place(index.pk, &key)? {
            let part = &composed.parts[at];
            if (seq as usize) < part.start || ((seq as usize) >= part.end && &part.session != owner)
            {
                continue;
            }
            let end = if whole_message {
                reader.item_unit_end(index.pk, seq)?
            } else {
                seq + 1
            } as usize;
            for (at, before) in composed.parts[..at].iter_mut().enumerate() {
                if owned[at] {
                    before.end = before.start;
                }
            }
            let part = &mut composed.parts[at];
            part.start = part.start.max(end).min(part.end);
            return Ok(());
        }
    }
    // A continuation's physical mark can still name its copied prefix;
    // logical composition shows only that continuation's own entries.
    if logical
        && owner != root
        && let Some(message) = key.message()
        && reader
            .select_messages(owner, true)?
            .message_key(message)?
            .is_some()
    {
        return Ok(());
    }
    Err(Error::conflict(
        "the seen mark is no longer in this physical session",
    ))
}

/// The seen marks a displayed page leaves: for each physical owner, the
/// last complete entry of `kept`, never the entry the page stops inside.
pub(super) fn seen_marks(
    reader: &Reader<'_>,
    composed: &Composed,
    request: &ReadRequest,
    kept: &[PageEntry],
    unfinished: Option<&EntryKey>,
    positions: &polylogue_model::ChangePosition,
) -> Result<Vec<polylogue_model::user::SeenMark>, Error> {
    let options = &request.options;
    let mut marks = Vec::new();
    let mut marked = HashSet::new();
    if let Some(bound) = session_read(request) {
        for entry in kept.iter().rev() {
            let part = &composed.parts[entry.part];
            // Inline children have their own identities. A physical mark
            // belongs to this session and its recorded inherited prefix.
            let owner = if bound.logical {
                if spawned(part) {
                    continue;
                }
                if part.inherited {
                    &bound.session
                } else {
                    &part.session
                }
            } else {
                if part.session != bound.session && !part.inherited {
                    continue;
                }
                &bound.session
            };
            if marked.contains(owner) {
                continue;
            }
            if unfinished.is_some_and(|key| *key == entry.entry.key) {
                continue;
            }
            let EntryKey::Message { message, block } = &entry.entry.key else {
                continue;
            };
            if options.view == View::Raw {
                // The original message is displayed at its header. Its index
                // block slots account for size but display nothing themselves.
                if block.is_some() {
                    continue;
                }
            } else if block.is_none() {
                // Context can stop at the anchor's header, before its body.
                // A complete header must not consume those unread blocks.
                let index = &composed.indexes[entry.part];
                let next = reader.item_window(
                    index.pk,
                    entry.seq as u64 + 1..index.entries,
                    true,
                    1,
                    |_| true,
                )?;
                if next.first().is_some_and(|next| {
                    next.entry.key.message() == Some(message) && next.entry.key != entry.entry.key
                }) {
                    continue;
                }
            }
            marks.push(polylogue_model::user::SeenMark {
                target: Reference::Session(owner.clone()).to_string(),
                item: Some(entry.entry.key.reference()),
                at: positions.clone(),
            });
            marked.insert(owner.clone());
        }
    }
    Ok(marks)
}
