use super::*;
use crate::encoding::{
    block_text, event_fields, message_content, read_json, read_message_content, restore_text,
};
use proptest::prelude::*;
use serde_json::json;

/// A message as the index reads it back: its content decoded, then each
/// block's text put back from its text row.
fn stored_message(message: &Message) -> Message {
    let texts: Vec<Option<String>> = message
        .blocks
        .iter()
        .map(|block| block_text(block).map(str::to_owned))
        .collect();
    let content = message_content(&mut message.clone(), &[]).unwrap();
    let (mut stored, _) = read_message_content(&content, true).unwrap();
    for (block, text) in stored.blocks.iter_mut().zip(texts) {
        if let Some(text) = text {
            restore_text(block, text);
        }
    }
    stored
}

/// An event as the index reads it back: its fields, then its data.
fn stored_event(event: &Event) -> Event {
    let mut stored: Event = read_json(&event_fields(&mut event.clone()).unwrap()).unwrap();
    stored.data = read_json(&serde_json::to_vec(&event.data).unwrap()).unwrap();
    stored
}

fn message(blocks: serde_json::Value) -> Message {
    serde_json::from_value(json!({
        "id": "claude-code:m", "session": "claude-code:s", "role": "user",
        "blocks": blocks, "usage": {}
    }))
    .unwrap()
}

fn line() -> impl Strategy<Value = String> {
    prop_oneof![
        "[a-z ]{0,30}",
        "[0-9a-f]{7,12}".prop_map(|hash| format!("[master {hash}] subject")),
        (1u32..9999).prop_map(|n| format!("https://github.com/o/r/pull/{n}/files.")),
        Just("<fork-boilerplate> go on".to_owned()),
        Just(String::new()),
    ]
}

fn text() -> impl Strategy<Value = String> {
    prop::collection::vec(line(), 0..4).prop_map(|lines| lines.join("\n"))
}

fn block() -> impl Strategy<Value = serde_json::Value> {
    prop_oneof![
        text().prop_map(|text| json!({"type": "text", "text": text})),
        (text(), any::<bool>()).prop_map(|(output, error)| json!({
            "type": "tool_result", "call_id": "call:1", "output": output,
            "outcome": if error { json!({"kind": "error"}) } else { json!({"kind": "ok"}) }
        })),
        text().prop_map(|command| json!({
            "type": "tool_call", "call_id": "call:1", "tool": "shell",
            "input": {"command": command}, "command": command
        })),
    ]
}

fn event() -> impl Strategy<Value = Event> {
    let kind = prop_oneof![
        Just("hook_pre_tool_use"),
        Just("process_started"),
        Just("compaction"),
        Just("microcompaction"),
        Just("permission_mode"),
    ];
    let data = prop_oneof![
        Just(json!({"tokens_saved": 3, "removed_calls": ["a", "b"]})),
        Just(
            json!({"summary": null, "pre_tokens": 10, "post_tokens": null, "trigger": "auto",
                    "kept_range": null, "kept_after": null})
        ),
        Just(json!({"cwd": "/w", "nested": [1, null, {"x": "λ"}]})),
        Just(json!(null)),
    ];
    (kind, data, any::<bool>()).prop_map(|(kind, data, timed)| {
        serde_json::from_value(json!({
            "session": "claude-code:s", "kind": kind,
            "at": if timed { json!("2026-01-02T03:04:05Z") } else { json!(null) },
            "data": data,
        }))
        .unwrap()
    })
}

proptest! {
    #[test]
    fn a_message_read_back_from_the_index_has_the_head_the_parser_gave(
        blocks in prop::collection::vec(block(), 0..6)
    ) {
        let parsed = message(json!(blocks));
        prop_assert_eq!(message_head(&stored_message(&parsed)), message_head(&parsed));
    }

    #[test]
    fn an_event_read_back_from_the_index_has_the_head_the_parser_gave(event in event()) {
        prop_assert_eq!(event_head(&stored_event(&event)), event_head(&event));
    }
}

#[test]
fn a_message_head_holds_the_prompt_title_and_each_result_s_artifacts() {
    let head = message_head(&message(json!([
        {"type": "tool_call", "call_id": "c", "tool": "shell", "input": {}},
        {"type": "text", "text": "\n  Fix the parser  \nmore"},
        {"type": "tool_result", "call_id": "c", "outcome": {"kind": "ok"},
         "output": "[main 1A2b3c4d] done\nhttps://github.com/o/r/pull/12/files."},
        {"type": "tool_result", "call_id": "d", "outcome": {"kind": "ok"}, "output": "nothing"},
    ])));
    assert_eq!(head.title.as_deref(), Some("Fix the parser"));
    assert_eq!(
        head.results,
        vec![ResultEvidence {
            block: 2,
            call_id: Some("c".into()),
            commits: vec!["1a2b3c4d".into()],
            pulls: vec![("o/r".into(), 12, "https://github.com/o/r/pull/12".into())],
        }]
    );
    let forked = message_head(&message(
        json!([{"type": "text", "text": "<fork-boilerplate>x"}]),
    ));
    assert_eq!(forked.title, None);
    let long = message_head(&message(json!([{"type": "text", "text": "é".repeat(81)}])));
    assert_eq!(long.title, Some(format!("{}…", "é".repeat(80))));
}

#[test]
fn an_event_head_holds_live_data_with_a_time_and_compaction_boundaries() {
    let event = |kind: &str, at: Option<&str>, data| -> Event {
        serde_json::from_value(json!({"kind": kind, "at": at, "data": data})).unwrap()
    };
    let at = Some("2026-01-02T03:04:05Z");
    assert_eq!(
        event_head(&event("hook_stop", at, json!({"a": 1}))).live,
        Some(json!({"a": 1}))
    );
    assert_eq!(
        event_head(&event("hook_stop", None, json!({"a": 1}))).live,
        None
    );
    assert_eq!(
        event_head(&event("permission_mode", at, json!({}))).live,
        None
    );
    assert_eq!(
        event_head(&event(
            "microcompaction",
            at,
            json!({"removed_calls": ["x"]})
        ))
        .removed_calls,
        vec!["x".to_owned()]
    );
    let compaction = event_head(&event("compaction", at, json!({"pre_tokens": 5})));
    assert_eq!(compaction.compaction.and_then(|b| b.pre_tokens), Some(5));
    assert_eq!(
        event_head(&event("compaction", at, json!({"bad": 1}))).compaction,
        None
    );
}
