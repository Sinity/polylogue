//! The shown value of an item seen through several routes: a pure function of
//! each input's latest reading and the route precedence (docs/sessions.md, one
//! conversation through several routes). The writer's membership tables
//! (`resolve/membership.rs`) gather the readings and apply the result.
use polylogue_model::{Completion, FieldGroup, Message, Route, Timestamp};
use std::cmp::Reverse;

/// One input's latest reading of a message.
pub(super) struct MessageReading<'a> {
    /// The input's ID: between readings with the same content, the main
    /// route is the one whose input ID sorts first, which changes nothing
    /// shown but where the message's reading is stored.
    pub input: &'a str,
    pub route: Route,
    /// The reading's session is a main session, not a sidechain's copy of it.
    pub main: bool,
    /// This source names the work session rather than carrying unknown compaction history.
    pub explicit_owner: bool,
    pub completion: Completion,
    pub hash: &'a [u8],
    pub message: &'a Message,
}

/// The shown message composed from several readings.
#[derive(Debug, PartialEq)]
pub(super) struct Composition {
    /// The reading whose content is shown: the main route.
    pub main: usize,
    /// Each other field group whose value comes from another reading.
    pub groups: Vec<(FieldGroup, usize)>,
    pub message: Message,
}

/// Content comes from an explicit work-session owner when one is recorded.
/// Each field group otherwise prefers a reading that has a value for it, then
/// a main session's reading, then the group's route precedence, then
/// finished before unknown before in-progress, and last by content hash, so
/// the result never depends on arrival order. `None` without readings.
pub(super) fn message(readings: &[MessageReading<'_>]) -> Option<Composition> {
    let best = |group: FieldGroup| {
        (0..readings.len()).min_by_key(|&n| {
            let r = &readings[n];
            (
                group == FieldGroup::Content && !r.explicit_owner,
                !group.present(r.message),
                Reverse(r.main),
                group.rank(r.route),
                r.completion,
                r.hash,
                r.input,
            )
        })
    };
    let main = best(FieldGroup::Content)?;
    let mut message = readings[main].message.clone();
    let mut groups = Vec::new();
    for &group in FieldGroup::ALL {
        if group == FieldGroup::Content {
            continue;
        }
        let winner = best(group).expect("readings are not empty");
        if winner != main && group.present(readings[winner].message) {
            group.take(&mut message, readings[winner].message);
            groups.push((group, winner));
        }
    }
    Some(Composition {
        main,
        groups,
        message,
    })
}

/// One input's latest reading of a document.
pub(super) struct DocumentReading<'a> {
    pub input: &'a str,
    pub time: Option<Timestamp>,
    pub route: Route,
    pub hash: &'a [u8],
}

/// The shown document: the newest source time, then route precedence, then
/// the content hash; between equal versions, the first input ID.
pub(super) fn document(readings: &[DocumentReading<'_>]) -> Option<usize> {
    (0..readings.len()).min_by_key(|&n| {
        let r = &readings[n];
        (
            Reverse(r.time),
            FieldGroup::Content.rank(r.route),
            Reverse(r.hash),
            r.input,
        )
    })
}

/// The shown reading of an event or link value: route precedence, then the
/// value's hash; between equal values, the first input ID.
pub(super) fn value(readings: &[(Route, &[u8], &str)]) -> Option<usize> {
    (0..readings.len()).min_by_key(|&n| {
        let (route, hash, input) = readings[n];
        (FieldGroup::Content.rank(route), hash, input)
    })
}
