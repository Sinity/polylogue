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

fn session(messages: usize) -> Vec<Message> {
    (0..messages)
        .map(|n| {
            let n = n as u32;
            if n.is_multiple_of(2) {
                message(
                    n,
                    Role::User,
                    Some(PromptKind::Prose),
                    vec![text(&"word ".repeat(1 + (n as usize * 7) % 40))],
                )
            } else {
                message(
                    n,
                    Role::Assistant,
                    None,
                    vec![
                        Block::Thinking {
                            text: Some("considering".into()),
                            signature: None,
                            redacted: false,
                        },
                        text(&"answer ".repeat(1 + (n as usize * 13) % 90)),
                        call(&format!("c{n}"), "cargo test"),
                        result(&format!("c{n}"), (n as usize * 31) % 400),
                    ],
                )
            }
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
        &FoldProfile::of(&options),
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
    let hidden = visible_entries(&segment, &FoldProfile::of(&ReadOptions::default()));
    assert!(hidden.is_empty());
    let shown = visible_entries(
        &segment,
        &FoldProfile::of(&ReadOptions {
            branches: true,
            ..ReadOptions::default()
        }),
    );
    assert_eq!(shown.len(), 2);
}

#[test]
fn page_boundaries_from_running_sums_equal_greedy_layout() {
    for size in [0, 1, 2, 7, 40, 301] {
        let entries = entries_of(&session(size));
        let segment = [Segment {
            entries: &entries,
            inherited: false,
        }];
        for options in [
            ReadOptions::default(),
            ReadOptions {
                tools: Some(Expansion::Full),
                thinking: Some(Expansion::Full),
                ..ReadOptions::default()
            },
            ReadOptions {
                view: View::Dialogue,
                ..ReadOptions::default()
            },
            ReadOptions {
                tools: Some(Expansion::Full),
                output_lines: 0,
                ..ReadOptions::default()
            },
        ] {
            let profile = FoldProfile::of(&options);
            let sums = visible_entries(&segment, &profile);
            for (tokens, items) in [(50, 50), (500, 50), (4000, 50), (4000, 3), (1, 1)] {
                // Greedy layout: add whole entries in order while they fit.
                let costs: Vec<u64> = sums.shown.iter().map(|shown| shown.cost).collect();
                let mut start = 0;
                let mut index_start = 0;
                while start < costs.len() {
                    let mut used = 0u64;
                    let mut end = start;
                    while end < costs.len()
                        && end - start < items
                        && (end == start || used + costs[end] <= tokens)
                    {
                        used += costs[end];
                        end += 1;
                    }
                    let from_index = sums.page_after(index_start, tokens, items);
                    assert_eq!(
                        from_index, end,
                        "size {size} budget {tokens}/{items} at {start}"
                    );
                    start = end;
                    index_start = from_index;
                }
                assert_eq!(index_start, costs.len());
            }
        }
    }
}

#[test]
fn backward_pages_tile_the_reading_with_no_gap_or_overlap() {
    let entries = entries_of(&session(120));
    let segment = [Segment {
        entries: &entries,
        inherited: false,
    }];
    let sums = visible_entries(&segment, &FoldProfile::of(&ReadOptions::default()));
    for (tokens, items) in [(50, 50), (500, 7), (4000, 50)] {
        let mut end = sums.len();
        let mut seen = 0;
        while end > 0 {
            let start = sums.page_before(end, tokens, items);
            assert!(start < end, "a backward page always makes progress");
            assert!(end - start <= items);
            seen += end - start;
            end = start;
        }
        assert_eq!(seen, sums.len());
    }
}

#[test]
fn an_oversized_entry_still_makes_progress() {
    let messages = vec![message(
        0,
        Role::User,
        Some(PromptKind::Prose),
        vec![text(&"long ".repeat(5000))],
    )];
    let entries = entries_of(&messages);
    let sums = visible_entries(
        &[Segment {
            entries: &entries,
            inherited: false,
        }],
        &FoldProfile::of(&ReadOptions::default()),
    );
    assert_eq!(sums.page_after(0, 10, 50), 1);
    assert_eq!(sums.page_after(1, 10, 50), 2);
}
