use super::*;
use crate::{
    Block, Message, PromptKind, Role, Usage,
    read::{MessagePlace, event_entry, message_entries},
};

fn message(session: &str, n: u32, role: Role) -> Message {
    Message {
        id: MessageId::parse(&format!("test:{session}:m{n}")).unwrap(),
        session: SessionId::parse(&format!("test:{session}")).unwrap(),
        parent: None,
        role,
        at: None,
        at_quality: crate::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        model: None,
        prompt_kind: (role == Role::User).then_some(PromptKind::Prose),
        blocks: vec![Block::Text {
            text: format!("message {n} of {session}"),
        }],
        usage: Usage::default(),
        stop_reason: None,
        reported_cost_usd: None,
        duration_ms: None,
        production: Default::default(),
        attrs: Default::default(),
    }
}

fn entries(session: &str, messages: u32) -> Vec<IndexEntry> {
    (0..messages)
        .flat_map(|n| {
            let role = if n.is_multiple_of(2) {
                Role::User
            } else {
                Role::Assistant
            };
            message_entries(&message(session, n, role), &[0], MessagePlace::default())
        })
        .collect()
}

fn whole(session: &str, entries: &[IndexEntry]) -> Part {
    Part {
        session: SessionId::parse(&format!("test:{session}")).unwrap(),
        start: 0,
        end: entries.len(),
        inherited: false,
        joined_by: None,
    }
}

#[test]
fn a_prefix_ends_after_the_branch_point_message() {
    let parent = entries("parent", 5);
    let branch = MessageId::parse("test:parent:m2").unwrap();
    assert_eq!(prefix_through(&parent, &branch), Some(6));
    let missing = MessageId::parse("test:parent:m9").unwrap();
    assert_eq!(prefix_through(&parent, &missing), None);
}

/// Six messages, a compaction after the third, then its summary event and
/// three more messages.
fn compacted() -> Vec<IndexEntry> {
    let mut out = entries("one", 3);
    out.push(event_entry("boundary", "compaction", None, 10, Vec::new()));
    out.push(event_entry(
        "summary",
        "compaction_summary",
        None,
        10,
        Vec::new(),
    ));
    out.extend((3..6).flat_map(|n| {
        message_entries(
            &message("one", n, Role::User),
            &[0],
            MessagePlace::default(),
        )
    }));
    out
}

#[test]
fn the_newest_boundary_replaces_everything_before_it() {
    let entries = compacted();
    let parts = [whole("one", &entries)];
    let last = Place {
        part: 0,
        entry: entries.len() - 1,
    };
    let context = context_at(&parts, |_| &entries, last);
    assert!(context.complete);
    assert_eq!(context.boundary, Some(Place { part: 0, entry: 6 }));
    assert_eq!(context.evicted.len(), 3);
    assert_eq!(context.members.first(), Some(&Place { part: 0, entry: 7 }));
    assert_eq!(context.members.len(), entries.len() - 7);
    // Before the boundary everything so far is in context.
    let early = context_at(&parts, |_| &entries, Place { part: 0, entry: 3 });
    assert_eq!(early.members.len(), 4);
    assert!(early.evicted.is_empty());
}

#[test]
fn an_incomplete_newest_boundary_never_falls_back_to_an_older_summary() {
    let mut entries = compacted();
    entries.push(event_entry("second", "compaction", None, 10, Vec::new()));
    entries.extend(message_entries(
        &message("one", 9, Role::User),
        &[0],
        MessagePlace::default(),
    ));
    let parts = [whole("one", &entries)];
    let context = context_at(
        &parts,
        |_| &entries,
        Place {
            part: 0,
            entry: entries.len() - 1,
        },
    );
    assert!(!context.complete);
    assert!(context.reason.is_some());
    assert_eq!(
        context.boundary,
        Some(Place {
            part: 0,
            entry: entries.len() - 3
        })
    );
}

#[test]
fn context_at_a_minus_left_plus_entered_is_context_at_b() {
    let entries = compacted();
    let parts = [whole("one", &entries)];
    for a in 0..entries.len() {
        for b in a..entries.len() {
            let earlier = context_at(&parts, |_| &entries, Place { part: 0, entry: a });
            let later = context_at(&parts, |_| &entries, Place { part: 0, entry: b });
            let difference = context_difference(&earlier, &later);
            let mut rebuilt: Vec<Place> = earlier
                .members
                .iter()
                .filter(|place| !difference.left.contains(place))
                .copied()
                .collect();
            rebuilt.extend(difference.entered.iter().copied());
            rebuilt.sort();
            let mut expected = later.members.clone();
            expected.sort();
            assert_eq!(rebuilt, expected, "{a}..{b}");
        }
    }
}

#[test]
fn the_meter_is_reported_then_estimated_never_zero_for_unknown() {
    let mut reported = message("one", 1, Role::Assistant);
    reported.usage.input = Some(1234);
    let mut entries = message_entries(&reported, &[0], MessagePlace::default());
    entries.extend(message_entries(
        &message("one", 2, Role::Assistant),
        &[0],
        MessagePlace::default(),
    ));
    let parts = [whole("one", &entries)];
    assert_eq!(
        context_meter(&parts, |_| &entries, Place { part: 0, entry: 0 }),
        ContextMeter::Reported(1234)
    );
    assert!(matches!(
        context_meter(&parts, |_| &entries, Place { part: 0, entry: 2 }),
        ContextMeter::Estimated(tokens) if tokens > 0
    ));
    let empty = [event_entry("e", "compaction", None, 0, Vec::new())];
    let parts = [whole("one", &empty)];
    assert_eq!(
        context_meter(&parts, |_| &empty, Place { part: 0, entry: 0 }),
        ContextMeter::Unknown
    );
}

#[test]
fn a_key_is_found_in_its_part() {
    let parent = entries("parent", 4);
    let child = entries("child", 2);
    let parts = [
        Part {
            inherited: true,
            end: 4,
            ..whole("parent", &parent)
        },
        whole("child", &child),
    ];
    let lookup = |part: usize| if part == 0 { &parent[..] } else { &child[..] };
    let key = EntryKey::header(&MessageId::parse("test:child:m1").unwrap());
    assert_eq!(
        find(&parts, lookup, &key),
        Some(Place { part: 1, entry: 2 })
    );
    let outside = EntryKey::header(&MessageId::parse("test:parent:m3").unwrap());
    assert_eq!(find(&parts, lookup, &outside), None);
}

/// A result entry for `call`, made from a message's text entry.
fn result(n: u32, call: &str) -> Vec<IndexEntry> {
    let mut out = message_entries(
        &message("micro", n, Role::User),
        &[0],
        MessagePlace::default(),
    );
    out[1].kind = EntryKind::Result;
    out[1].call_id = Some(call.into());
    out
}

#[test]
fn a_microcompaction_removes_the_results_it_names_before_it() {
    let data = serde_json::json!({
        "microcompactMetadata": {"compactedToolIds": ["first"], "clearedAttachmentUUIDs": []}
    });
    assert_eq!(crate::read::microcompacted_calls(&data), ["first"]);
    let mut entries = result(0, "first");
    entries.extend(result(1, "second"));
    entries.push(event_entry(
        "micro",
        "microcompaction",
        None,
        10,
        crate::read::microcompacted_calls(&data),
    ));
    // A later result with a cleared ID is not removed by an earlier event.
    entries.extend(result(2, "first"));
    let parts = [whole("micro", &entries)];
    let at = |entry| Place { part: 0, entry };
    let context = context_at(&parts, |_| &entries, at(entries.len() - 1));
    assert!(context.complete);
    assert!(context.boundary.is_none());
    assert!(!context.members.contains(&at(1)), "the named result left");
    assert!(context.members.contains(&at(3)), "an unnamed result stays");
    assert!(context.members.contains(&at(0)), "its message header stays");
    assert!(context.members.contains(&at(6)), "a later result stays");
    assert_eq!(context.members.len(), entries.len() - 1);
    // Before the event, the result is still in context.
    let before = context_at(&parts, |_| &entries, at(3));
    assert!(before.members.contains(&at(1)));
}
