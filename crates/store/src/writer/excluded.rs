//! Keeping excised content out: a batch that names an excised session or
//! message is written without it, so excised content never comes back by
//! any route.

use super::*;

/// Whether a batch's rows name an excised session or message.
pub(super) fn touches(b: &Batch, r: &crate::excise::Refused) -> bool {
    b.parse.iter().any(|p| {
        let o = &p.output;
        o.sessions
            .iter()
            .any(|s| r.sessions.contains(&s.id.to_string()))
            || o.messages.iter().any(|m| refused_message(&m.message, r))
            || o.events
                .iter()
                .any(|e| r.sessions.contains(&e.session.to_string()))
            || o.links.iter().any(|l| {
                r.sessions.contains(&l.parent.to_string())
                    || r.sessions.contains(&l.child.to_string())
            })
    })
}

pub(super) fn refused_message(m: &Message, r: &crate::excise::Refused) -> bool {
    r.messages.contains(&m.id.to_string()) || r.sessions.contains(&m.session.to_string())
}

/// A copy of a batch without the rows of excised sessions and messages:
/// they are never written again, from any input (architecture 5.6). The
/// bytes and parse position stay, so the input is not parsed again for
/// them.
pub(super) fn without_refused(b: &Batch, r: &crate::excise::Refused) -> Batch {
    let mut out = b.clone();
    for p in &mut out.parse {
        let keep_msgs: Vec<bool> = p
            .output
            .messages
            .iter()
            .map(|m| !refused_message(&m.message, r))
            .collect();
        let keep_events: Vec<bool> = p
            .output
            .events
            .iter()
            .map(|e| !r.sessions.contains(&e.session.to_string()))
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
