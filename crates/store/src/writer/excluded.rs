//! Keeping excised content out: a batch that names an excised session or
//! message is written without it, so excised content never comes back by
//! any route.

use super::*;

/// Include every identity the filter can remove; this is the only walk that
/// decides whether a group needs filtering.
pub(super) fn matching(
    group: &[Box<Batch>],
    cache: &crate::excise::Refused,
) -> Result<crate::excise::RefusedIds> {
    use std::collections::BTreeSet;
    let mut sessions = BTreeSet::new();
    let mut messages = BTreeSet::new();
    let mut documents = BTreeSet::new();
    for b in group {
        for p in &b.parse {
            let o = &p.output;
            documents.extend(o.documents.iter().map(|row| row.document.id.to_string()));
            sessions.extend(o.sessions.iter().map(|s| s.id.to_string()));
            for m in &o.messages {
                sessions.insert(m.message.session.to_string());
                messages.insert(m.message.id.to_string());
            }
            sessions.extend(
                o.events
                    .iter()
                    .filter_map(|e| e.session.as_ref())
                    .map(ToString::to_string),
            );
            for link in &o.links {
                sessions.insert(link.parent.to_string());
                sessions.insert(link.child.to_string());
            }
            sessions.extend(o.background.iter().map(|b| b.session.to_string()));
            if let Some(snapshot) = &p.snapshot {
                sessions.extend(snapshot.covered.iter().map(ToString::to_string));
            }
        }
    }
    let mut selected = cache.matching(
        &sessions.into_iter().collect::<Vec<_>>(),
        &messages.into_iter().collect::<Vec<_>>(),
    )?;
    selected.documents = cache.matching_documents(&documents.into_iter().collect::<Vec<_>>())?;
    Ok(selected)
}

pub(super) fn refused_message(m: &Message, r: &crate::excise::RefusedIds) -> bool {
    r.messages.contains(&m.id.to_string()) || r.sessions.contains(&m.session.to_string())
}

/// A copy of a batch without the rows of excised sessions and messages:
/// they are never written again, from any input (architecture 5.6). The
/// bytes and parse position stay, so the input is not parsed again for
/// them.
pub(super) fn without_refused(b: &Batch, r: &crate::excise::RefusedIds) -> Batch {
    let mut out = b.clone();
    for p in &mut out.parse {
        let keep_msgs: Vec<bool> = p
            .output
            .messages
            .iter()
            .map(|m| !refused_message(&m.message, r))
            .collect();
        let keep_documents: Vec<bool> = p
            .output
            .documents
            .iter()
            .map(|row| !r.documents.contains(&row.document.id.to_string()))
            .collect();
        let mut documents = keep_documents.iter();
        p.output
            .documents
            .retain(|_| *documents.next().expect("one decision per Document"));
        p.prepared.retain_documents(&keep_documents);
        let keep_events: Vec<bool> = p
            .output
            .events
            .iter()
            .map(|e| {
                e.session
                    .as_ref()
                    .is_none_or(|id| !r.sessions.contains(&id.to_string()))
            })
            .collect();
        let mut i = 0;
        p.output.messages.retain(|_| {
            i += 1;
            keep_msgs[i - 1]
        });
        p.prepared.retain_messages(&keep_msgs);
        let mut i = 0;
        p.output.events.retain(|_| {
            i += 1;
            keep_events[i - 1]
        });
        p.prepared.retain_events(&keep_events);
        p.output
            .sessions
            .retain(|s| !r.sessions.contains(&s.id.to_string()));
        p.output.links.retain(|l| {
            !r.sessions.contains(&l.parent.to_string())
                && !r.sessions.contains(&l.child.to_string())
        });
        p.output
            .background
            .retain(|u| !r.sessions.contains(&u.session.to_string()));
        if let Some(snapshot) = &mut p.snapshot {
            snapshot
                .covered
                .retain(|session| !r.sessions.contains(&session.to_string()));
        }
    }
    out
}
