use super::*;
use crate::parse::run;
use serde_json::json;

fn session() -> SessionId {
    session_id(super::super::ORIGIN, "example").unwrap()
}
fn record(index: u64, source: &str, kind: &str, extra: Json) -> Json {
    let mut value = json!({"step_index":index,"source":source,"type":kind,"status":"DONE","created_at":"2026-03-05T04:21:34Z"});
    value
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    value
}
fn bytes(rows: &[Json]) -> Vec<u8> {
    rows.iter()
        .flat_map(|row| {
            let mut line = serde_json::to_vec(row).unwrap();
            line.push(b'\n');
            line
        })
        .collect()
}

#[test]
fn antigravity_transcript_pairs_steps_with_calls_and_preserves_running_updates() {
    let rows = [
        record(
            0,
            "USER_EXPLICIT",
            "USER_INPUT",
            json!({"content":"Run the check"}),
        ),
        record(
            1,
            "MODEL",
            "PLANNER_RESPONSE",
            json!({"content":"Checking","thinking":"Inspect first","tool_calls":[{"name":"run_command","args":{"CommandLine":"printf example","Cwd":"/work"}},{"name":"view_file","args":{"AbsolutePath":"/work/example.rs"}}]}),
        ),
        record(
            2,
            "MODEL",
            "RUN_COMMAND",
            json!({"status":"RUNNING","content":"still running"}),
        ),
        record(2, "MODEL", "RUN_COMMAND", json!({"content":"finished"})),
        record(
            3,
            "MODEL",
            "VIEW_FILE",
            json!({"error":"missing file","error_code":2,"content":"not found"}),
        ),
        record(4, "SYSTEM", "CHECKPOINT", json!({"content":"checkpoint"})),
    ];
    let input = bytes(&rows);
    let step = run(
        &Transcript { session: session() },
        &mut State::default(),
        &input,
        0,
    )
    .unwrap();
    assert_eq!(step.output.sessions.len(), 1);
    assert_eq!(step.output.messages.len(), 5);
    let messages = &step.output.messages;
    assert_eq!(messages[0].message.prompt_kind, Some(PromptKind::Prose));
    assert_eq!(messages[1].message.role, Role::Assistant);
    assert!(
        matches!(&messages[1].message.blocks[0],Block::Thinking {text:Some(text),..} if text=="Inspect first")
    );
    assert!(
        matches!(&messages[1].message.blocks[2],Block::ToolCall {call_id:Some(call),action:ToolAction::Shell,command:Some(command),..} if call=="step-1-0" && command=="printf example")
    );
    assert!(
        matches!(&messages[1].message.blocks[3],Block::ToolCall {action:ToolAction::FileRead,path:Some(path),..} if path=="/work/example.rs")
    );
    assert!(
        matches!(&messages[2].message.blocks[0],Block::ToolResult {call_id:Some(call),outcome:ToolOutcome::Unknown {reason:UnknownReason::RunningInBackground},..} if call=="step-1-0")
    );
    assert!(
        matches!(&messages[3].message.blocks[0],Block::ToolResult {call_id:Some(call),outcome:ToolOutcome::Ok,..} if call=="step-1-0")
    );
    assert!(
        matches!(&messages[4].message.blocks[0],Block::ToolResult {call_id:Some(call),outcome:ToolOutcome::Error {..},..} if call=="step-1-1")
    );
    assert_eq!(messages[2].message.id, messages[3].message.id);
    assert_eq!(messages[2].message.parent, messages[3].message.parent);
    assert_ne!(
        messages[3].message.parent.as_ref(),
        Some(&messages[3].message.id)
    );
    assert!(
        messages[2..]
            .iter()
            .all(|row| row.message.prompt_kind == Some(PromptKind::Harness))
    );
    let event = &step.output.events[0];
    assert_eq!(event.kind, "antigravity_checkpoint");
    assert_eq!(event.data, rows[5]);
    let mut offset = 0;
    for (row, original) in messages.iter().zip(&rows) {
        let len = serde_json::to_vec(original).unwrap().len() as u64;
        assert_eq!((row.offset, row.end), (offset, offset + len));
        offset += len + 1;
    }
    assert_eq!(event.range, Some((offset, input.len() as u64 - 1)));
}

#[test]
fn transcript_append_state_survives_serialization_and_orphans_remain_unpaired() {
    let first = bytes(&[record(
        0,
        "MODEL",
        "PLANNER_RESPONSE",
        json!({"tool_calls":[{"name":"grep_search","args":{"Query":"example","SearchPath":"/work"}}]}),
    )]);
    let second = bytes(&[
        record(1, "MODEL", "GREP_SEARCH", json!({"content":"match"})),
        record(2, "MODEL", "GENERIC", json!({"content":"orphan"})),
    ]);
    let parser = Transcript { session: session() };
    let mut state = State::default();
    run(&parser, &mut state, &first, 0).unwrap();
    let mut state: State = serde_json::from_value(serde_json::to_value(state).unwrap()).unwrap();
    let tail = run(&parser, &mut state, &second, first.len() as u64).unwrap();
    assert!(
        matches!(&tail.output.messages[0].message.blocks[0],Block::ToolResult {call_id:Some(call),..} if call=="step-0-0")
    );
    assert!(matches!(
        &tail.output.messages[1].message.blocks[0],
        Block::ToolResult { call_id: None, .. }
    ));
    assert_eq!(tail.output.messages[0].offset, first.len() as u64);
    let all = run(&parser, &mut State::default(), &[first, second].concat(), 0).unwrap();
    assert_eq!(tail.output.messages, all.output.messages[1..]);
}

#[test]
fn transcript_identity_is_scoped_and_unknown_shapes_are_accounted() {
    let input = bytes(&[record(
        0,
        "USER_EXPLICIT",
        "USER_INPUT",
        json!({"content":"same prompt"}),
    )]);
    let one = run(
        &Transcript { session: session() },
        &mut State::default(),
        &input,
        0,
    )
    .unwrap();
    let two = run(
        &Transcript {
            session: session_id(super::super::ORIGIN, "another").unwrap(),
        },
        &mut State::default(),
        &input,
        0,
    )
    .unwrap();
    assert_ne!(
        one.output.messages[0].message.id,
        two.output.messages[0].message.id
    );
    let input = bytes(&[
        record(1, "NEW_SOURCE", "NEW_TYPE", json!({"content":"unknown"})),
        json!({"type":"USER_INPUT","source":"USER_EXPLICIT"}),
    ]);
    let step = run(
        &Transcript { session: session() },
        &mut State::default(),
        &input,
        0,
    )
    .unwrap();
    assert_eq!(step.counts.records, 2);
    assert_eq!(step.counts.skipped.values().sum::<u64>(), 1);
    assert_eq!(step.counts.unreadable.values().sum::<u64>(), 1);
    // NEW_SOURCE and NEW_TYPE are the only two values outside the known kinds.
    assert_eq!(
        step.counts
            .wire_shapes
            .values()
            .map(|shape| shape.unknown_kind_count)
            .sum::<u64>(),
        2
    );
    assert!(step.output.messages.is_empty());
}

#[test]
fn transcript_discovery_requires_the_documented_layout() {
    assert!(recognize(Path::new(
        "/home/example/.gemini/antigravity-cli/brain/native/.system_generated/logs/transcript_full.jsonl"
    )));
    assert!(!recognize(Path::new("/work/transcript_full.jsonl")));
    assert!(!recognize(Path::new(
        "brain/native/.system_generated/logs/other.jsonl"
    )));
}

#[test]
fn listing_a_directory_is_a_search_and_viewing_a_file_is_a_read() {
    assert_eq!(action("list_dir"), ToolAction::Search);
    assert_eq!(action("list_directory"), ToolAction::Search);
    assert_eq!(action("view_file"), ToolAction::FileRead);
}

#[test]
fn file_tools_are_writes_or_edits_by_what_they_do() {
    assert_eq!(action("write_to_file"), ToolAction::FileWrite);
    assert_eq!(action("replace_file_content"), ToolAction::FileEdit);
    assert_eq!(action("multi_replace_file_content"), ToolAction::FileEdit);
}
