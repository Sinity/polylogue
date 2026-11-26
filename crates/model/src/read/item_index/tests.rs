use super::*;
use crate::{MessageId, Role, SessionId, Usage};

fn message(n: u32, role: Role, prompt_kind: Option<PromptKind>, blocks: Vec<Block>) -> Message {
    Message {
        id: MessageId::parse(&format!("test:one:m{n}")).unwrap(),
        session: SessionId::parse("test:one").unwrap(),
        parent: None,
        role,
        at: None,
        at_quality: crate::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        model: None,
        prompt_kind,
        blocks,
        usage: Usage::default(),
        stop_reason: None,
        reported_cost_usd: None,
        duration_ms: None,
        production: Default::default(),
        attrs: Default::default(),
    }
}

fn text(body: &str) -> Block {
    Block::Text { text: body.into() }
}

fn call(id: &str, command: &str) -> Block {
    Block::ToolCall {
        written_text: None,
        call_id: Some(id.into()),
        tool: Some("Bash".into()),
        action: ToolAction::Shell,
        input: serde_json::json!({ "command": command }),
        command: Some(command.into()),
        path: None,
        run_by_script: false,
    }
}

fn result(id: &str, lines: usize) -> Block {
    Block::ToolResult {
        reported_duration_ms: None,
        call_id: Some(id.into()),
        outcome: crate::ToolOutcome::Ok,
        output: Some(
            (0..lines)
                .map(|line| format!("line {line}"))
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        data: None,
        attachments: vec![],
    }
}

fn entries_of(messages: &[Message]) -> Vec<IndexEntry> {
    messages
        .iter()
        .flat_map(|message| {
            let ordinals: Vec<u64> = (0..message.blocks.len() as u64).collect();
            message_entries(message, &ordinals, MessagePlace::default())
        })
        .collect()
}

#[test]
fn a_message_gives_a_header_then_one_entry_per_block() {
    let message = message(
        1,
        Role::Assistant,
        None,
        vec![text("hello"), call("c1", "ls"), result("c1", 3)],
    );
    let entries = message_entries(&message, &[0, 2, 5], MessagePlace::default());
    let kinds: Vec<_> = entries.iter().map(|entry| entry.kind).collect();
    assert_eq!(
        kinds,
        [
            EntryKind::Header,
            EntryKind::Text,
            EntryKind::Call,
            EntryKind::Result
        ]
    );
    assert_eq!(
        entries[2].key,
        EntryKey::Message {
            message: message.id.clone(),
            block: Some(2)
        }
    );
    assert_eq!(entries[2].target.as_deref(), Some("ls"));
    assert_eq!(entries[3].output_lines, 3);
}

#[test]
fn the_header_carries_reported_input_and_nothing_when_unreported() {
    let mut reported = message(1, Role::Assistant, None, vec![text("a")]);
    reported.usage = Usage {
        input: Some(10),
        cached_input: Some(1000),
        cache_write: Some(5),
        ..Usage::default()
    };
    let entries = message_entries(&reported, &[0], MessagePlace::default());
    assert_eq!(entries[0].context_tokens, Some(1015));
    let silent = message(2, Role::Assistant, None, vec![text("a")]);
    let entries = message_entries(&silent, &[0], MessagePlace::default());
    assert_eq!(entries[0].context_tokens, None);
}

#[test]
fn dialogue_shows_only_people_and_model_text() {
    let messages = vec![
        message(0, Role::User, Some(PromptKind::Prose), vec![text("ask")]),
        message(
            1,
            Role::Assistant,
            None,
            vec![call("c1", "ls"), result("c1", 2)],
        ),
        message(2, Role::User, Some(PromptKind::Harness), vec![text("hook")]),
        message(3, Role::Assistant, None, vec![text("done")]),
    ];
    let entries = entries_of(&messages);
    let options = ReadOptions {
        view: View::Dialogue,
        ..ReadOptions::default()
    };
    let sums = visible_entries(
        &[Segment {
            entries: &entries,
            inherited: false,
        }],
        &ShowProfile::of(&options),
    );
    let keys: Vec<String> = sums
        .shown
        .iter()
        .map(|shown| entries[shown.entry].key.reference())
        .collect();
    assert_eq!(
        keys,
        [
            "message:test:one:m0",
            "block:test:one:m0#0",
            "message:test:one:m3",
            "block:test:one:m3#0"
        ]
    );
}

#[test]
fn branches_and_harness_follow_the_options() {
    let mut off = message(1, Role::Assistant, None, vec![text("other version")]);
    off.id = MessageId::parse("test:one:off").unwrap();
    let entries = message_entries(
        &off,
        &[0],
        MessagePlace {
            active_path: Some(false),
            branch_point: false,
        },
    );
    let segment = [Segment {
        entries: &entries,
        inherited: false,
    }];
    let hidden = visible_entries(&segment, &ShowProfile::of(&ReadOptions::default()));
    assert!(hidden.is_empty());
    let shown = visible_entries(
        &segment,
        &ShowProfile::of(&ReadOptions {
            show: vec!["branches".into()],
            ..ReadOptions::default()
        }),
    );
    assert_eq!(shown.len(), 2);
}

/// Every kind of entry a reading can show or hide: prompts, model replies
/// with thinking, calls that change files and calls that do not, results,
/// attachments, harness messages, messages off the active path, a message
/// with no blocks, and events of every boundary.
fn varied_entries() -> Vec<IndexEntry> {
    let attachment = Block::Image(crate::Attachment {
        name: Some("a.png".into()),
        ..Default::default()
    });
    let lookup = Block::ToolCall {
        written_text: None,
        call_id: Some("c2".into()),
        tool: Some("Read".into()),
        action: ToolAction::Other,
        input: serde_json::json!({ "path": "/home/op/a" }),
        command: None,
        path: Some("/home/op/a".into()),
        run_by_script: false,
    };
    let thinking = Block::Thinking {
        text: Some("synthetic thought".into()),
        signature: None,
        redacted: false,
    };
    let encrypted = Block::Thinking {
        text: None,
        signature: Some("sig".into()),
        redacted: false,
    };
    let messages = [
        (
            message(0, Role::User, Some(PromptKind::Prose), vec![text("prompt")]),
            None,
        ),
        (
            message(
                1,
                Role::Assistant,
                None,
                vec![thinking, text("reply"), call("c1", "ls")],
            ),
            None,
        ),
        (message(2, Role::User, None, vec![result("c1", 30)]), None),
        (
            message(3, Role::Assistant, None, vec![lookup, encrypted]),
            None,
        ),
        (
            message(4, Role::User, None, vec![result("c2", 2), attachment]),
            None,
        ),
        (
            message(
                5,
                Role::User,
                Some(PromptKind::Harness),
                vec![text("injected")],
            ),
            None,
        ),
        (
            message(6, Role::Assistant, None, vec![text("abandoned")]),
            Some(false),
        ),
        (message(7, Role::Assistant, None, vec![]), None),
        (
            message(
                8,
                Role::Assistant,
                None,
                vec![Block::Other {
                    type_name: "custom".into(),
                    data: serde_json::json!({ "k": 1 }),
                }],
            ),
            None,
        ),
    ];
    let mut out = Vec::new();
    for (index, (message, active_path)) in messages.iter().enumerate() {
        let ordinals: Vec<u64> = (0..message.blocks.len() as u64).collect();
        out.extend(message_entries(
            message,
            &ordinals,
            MessagePlace {
                active_path: *active_path,
                branch_point: false,
            },
        ));
        let event = match index {
            2 => Some("compaction"),
            4 => Some("compaction_summary"),
            5 => Some("microcompaction"),
            6 => Some("hook"),
            _ => None,
        };
        if let Some(kind) = event {
            out.push(event_entry(&format!("e{index}"), kind, None, 40, vec![]));
        }
    }
    out
}

/// The word that asks for `level` of `kind`: a hidden kind is a hide word,
/// a full one its full level (tool output at its default 100 lines), none
/// no word.
fn asked(options: &mut ReadOptions, kind: &str, level: Option<Expansion>) {
    match level {
        None => {}
        Some(Expansion::None) => options.hide.push(kind.into()),
        Some(Expansion::Collapsed) => options.show.push(format!("{kind}:line")),
        Some(Expansion::Full) if kind == "tools" => options.show.push("tools:100".into()),
        Some(Expansion::Full) => options.show.push(format!("{kind}:full")),
    }
}

/// Every combination of the options that decide what a reading shows.
fn every_profile() -> Vec<ShowProfile> {
    let levels = [
        None,
        Some(Expansion::None),
        Some(Expansion::Collapsed),
        Some(Expansion::Full),
    ];
    let mut out = Vec::new();
    for view in View::ALL {
        for tools in levels {
            for thinking in levels {
                for harness in levels {
                    for branches in [false, true] {
                        let mut options = ReadOptions {
                            view,
                            ..ReadOptions::default()
                        };
                        for (kind, level) in [
                            ("tools", tools),
                            ("thinking", thinking),
                            ("harness", harness),
                        ] {
                            asked(&mut options, kind, level);
                        }
                        if branches {
                            options.show.push("branches".into());
                        }
                        out.push(ShowProfile::of(&options));
                    }
                }
            }
        }
    }
    out
}

#[test]
fn entry_classes_show_exactly_what_visible_entries_shows() {
    let entries = varied_entries();
    let classes = entry_classes(&entries);
    assert_eq!(classes.len(), entries.len());
    let segment = [Segment {
        entries: &entries,
        inherited: false,
    }];
    let mut shown_somewhere = vec![false; entries.len()];
    for profile in every_profile() {
        let expected: Vec<usize> = visible_entries(&segment, &profile)
            .shown
            .iter()
            .map(|shown| shown.entry)
            .collect();
        let by_class: Vec<usize> = (0..entries.len())
            .filter(|at| profile.shows(classes[*at]))
            .collect();
        assert_eq!(by_class, expected, "{profile:?}");
        for at in expected {
            shown_somewhere[at] = true;
        }
        let mut counts = ItemCounts::default();
        for (entry, class) in entries.iter().zip(&classes) {
            counts.add(entry, *class);
        }
        assert_eq!(counts.shown(&profile), by_class.len() as u64);
    }
    // Every entry is shown by some reading (the raw view shows them all).
    assert!(shown_somewhere.iter().all(|shown| *shown));
    for class in &classes {
        assert_eq!(EntryClass::from_bits(class.bits()), Some(*class));
    }
}

#[test]
fn stored_class_bits_with_the_reserved_top_bit_are_rejected() {
    assert_eq!(
        EntryClass::from_bits(11),
        Some(EntryClass::new(
            ClassKind::Event,
            ClassAuthor::Other,
            false,
            0
        ))
    );
    for bits in [0x8000, 0x800B] {
        assert_eq!(EntryClass::from_bits(bits), None, "{bits:#x}");
    }
}

fn entry_of_one_block(author: Role, prompt_kind: Option<PromptKind>, block: Block) -> IndexEntry {
    let message = message(1, author, prompt_kind, vec![block]);
    message_entries(&message, &[0], MessagePlace::default()).remove(1)
}

#[test]
fn a_hidden_harness_shows_none_of_its_messages_and_a_collapsed_one_a_line() {
    // A long text, so the collapsed line is cut at its fixed 24 tokens.
    let long = vec!["word"; 400].join(" ");
    let messages = vec![
        message(0, Role::User, Some(PromptKind::Prose), vec![text("ask")]),
        message(1, Role::User, Some(PromptKind::Harness), vec![text(&long)]),
    ];
    let entries = entries_of(&messages);
    let shown = |harness| {
        let mut options = ReadOptions::default();
        asked(&mut options, "harness", Some(harness));
        visible_entries(
            &[Segment {
                entries: &entries,
                inherited: false,
            }],
            &ShowProfile::of(&options),
        )
    };
    let none = shown(Expansion::None);
    assert_eq!(
        none.shown.iter().map(|s| s.entry).collect::<Vec<_>>(),
        [0, 1]
    );
    let collapsed = shown(Expansion::Collapsed);
    assert_eq!(
        collapsed.shown.iter().map(|s| s.entry).collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
    assert_eq!(collapsed.shown[3].cost, 24);
    let full = shown(Expansion::Full);
    assert_eq!(full.shown[3].cost, entries[3].tokens_full);
    assert!(full.shown[3].cost > 24);
}

#[test]
fn a_shown_tool_result_costs_its_head_and_tail_lines() {
    // 40 lines of 7 characters: 319 bytes, so 80 tokens (a quarter of the
    // bytes outweighs 1.3 tokens a word).
    let output = vec!["xxxxxxx"; 40].join("\n");
    let block = Block::ToolResult {
        reported_duration_ms: None,
        call_id: Some("c1".into()),
        outcome: crate::ToolOutcome::Ok,
        output: Some(output),
        data: None,
        attachments: vec![],
    };
    let entry = entry_of_one_block(Role::Assistant, None, block);
    assert_eq!((entry.tokens_full, entry.output_lines), (80, 40));
    let cost = |tools, output_lines| {
        let word = match (tools, output_lines) {
            (Expansion::Full, 0) => "tools:full".to_owned(),
            (Expansion::Full, lines) => format!("tools:{lines}"),
            (Expansion::Collapsed, _) => "tools:line".to_owned(),
            (Expansion::None, _) => String::new(),
        };
        ShowProfile::of(&if word.is_empty() {
            ReadOptions {
                hide: vec!["tools".into()],
                ..ReadOptions::default()
            }
        } else {
            ReadOptions {
                show: vec![word],
                ..ReadOptions::default()
            }
        })
        .cost(&entry)
    };
    // The first and last 5 lines are 10 of 40: a quarter of 80 tokens, and
    // the line that says the middle was cut.
    assert_eq!(cost(Expansion::Full, 5), Some(24));
    // The first and last 20 lines are all of it, and 0 keeps all of it.
    assert_eq!(cost(Expansion::Full, 20), Some(80));
    assert_eq!(cost(Expansion::Full, 0), Some(80));
    assert_eq!(cost(Expansion::Collapsed, 5), Some(0));
    assert_eq!(cost(Expansion::None, 5), None);
}
