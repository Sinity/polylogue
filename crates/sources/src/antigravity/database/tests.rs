use super::*;
use serde_json::json;

fn header() -> Json {
    json!({"schema":{"trajectory_meta":"CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT)",
        "steps":"CREATE TABLE steps (idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT)"}})
}
fn row(table: &str, key: i64, r: Json) -> Json {
    json!({"t":table,"k":[key],"r":r})
}
fn records() -> Vec<Json> {
    vec![
        header(),
        row(
            "trajectory_meta",
            1,
            json!({"trajectory_id":"trajectory-1","cascade_id":"cascade-1"}),
        ),
        row(
            "conversation_summaries",
            1,
            json!({"cascade_id":"cascade-1","title":"Native title"}),
        ),
        row(
            "parent_references",
            1,
            json!({"cascade_id":"cascade-1","parent_id":"parent-1"}),
        ),
        row(
            "steps",
            1,
            json!({"idx":0,"step_type":"message","step_format":"v1","step_payload":"{\"role\":\"user\",\"text\":\"hello\"}"}),
        ),
        row(
            "steps",
            2,
            json!({"idx":1,"step_type":"terminal_command","step_format":"v1","step_payload":{"tool_name":"shell","command":"printf hi"}}),
        ),
        row(
            "steps",
            3,
            json!({"idx":2,"step_type":"tool_result","step_format":"v1","step_payload":{"tool_name":"shell","output":"hi","status":"success"}}),
        ),
        row(
            "steps",
            4,
            json!({"idx":3,"step_type":"file_edit","step_format":"v1","step_payload":{"path":"README.md","old_string":"old","new_string":"new"}}),
        ),
        row(
            "steps",
            5,
            json!({"idx":4,"step_type":"future_step","step_format":"future","step_payload":{"opaque":true}}),
        ),
    ]
}
fn parse(records: Vec<Json>) -> Result<crate::Step, String> {
    let input = records.iter().map(|r| format!("{r}\n")).collect::<String>();
    crate::export::fixture::parse(
        &mut open(Path::new("conversation.db.jsonl")).unwrap(),
        input.as_bytes(),
    )
}
#[test]
fn trajectories_keep_identity_order_tools_summary_and_unknown_steps() {
    let mut input = records();
    input[4..].reverse();
    let parsed = parse(input).unwrap();
    assert_eq!(parsed.output.messages.len(), 4);
    let messages: Vec<_> = parsed.output.messages.iter().map(|r| &r.message).collect();
    assert_eq!(messages[0].session.to_string(), "antigravity:trajectory-1");
    assert_eq!(messages[0].text(), "hello");
    assert!(messages.iter().all(|m| m.at.is_none()));
    let Block::ToolCall { call_id, .. } = &messages[1].blocks[0] else {
        panic!("tool call")
    };
    assert!(
        matches!(&messages[2].blocks[0],Block::ToolResult{outcome:ToolOutcome::Ok,call_id:result,..} if result==call_id)
    );
    assert!(
        matches!(&messages[3].blocks[0],Block::ToolCall{action:ToolAction::FileEdit,input,..} if input["old_string"]=="old")
    );
    assert!(
        parsed
            .output
            .sessions
            .iter()
            .any(|s| s.title.as_ref().is_some_and(|t| t.text == "Native title"))
    );
    assert_eq!(
        parsed.output.links[0].parent.to_string(),
        "antigravity:parent-1"
    );
    assert!(
        parsed
            .output
            .events
            .iter()
            .any(|e| e.kind == "antigravity_unsupported_step")
    );
    assert_eq!(parsed.counts.records, 9);
}
#[test]
fn unknown_formats_and_malformed_payloads_do_not_become_prose() {
    for value in [json!("not-json"), json!({"role":"user","text":"hello"})] {
        let mut input = records();
        input[4]["r"]["step_payload"] = value;
        input[4]["r"]["step_format"] = json!("future");
        let parsed = parse(input).unwrap();
        assert_eq!(parsed.output.messages.len(), 3);
        assert!(
            parsed
                .output
                .messages
                .iter()
                .all(|r| r.message.text() != "hello")
        );
    }
}
#[test]
fn multiple_trajectories_never_receive_unkeyed_steps() {
    let mut input = records();
    input.push(row(
        "trajectory_meta",
        2,
        json!({"trajectory_id":"trajectory-2","cascade_id":"cascade-2"}),
    ));
    let parsed = parse(input).unwrap();
    assert!(parsed.output.messages.is_empty());
    assert!(
        parsed
            .output
            .events
            .iter()
            .any(|e| e.kind == "antigravity_unattributed_step")
    );
}
#[test]
fn row_revisions_keep_latest_content_and_empty_schema_is_attributable() {
    let mut input = records();
    let mut changed = input[4].clone();
    changed["r"]["step_payload"] = json!({"role":"user","text":"changed"});
    input.push(changed);
    let parsed = parse(input).unwrap();
    assert_eq!(parsed.output.messages[0].message.text(), "changed");
    assert_eq!(parsed.counts.records, 10);
    let empty = parse(vec![header()]).unwrap();
    assert_eq!(empty.output.sessions.len(), 1);
    assert!(parse(vec![json!({"schema":{}})]).is_err());
}

#[test]
fn ambiguous_adjacent_calls_do_not_invent_a_result_pair() {
    let mut input = records();
    input[7]["r"]["idx"] = json!(4);
    let mut another = input[5].clone();
    another["k"] = json!([10]);
    another["r"]["idx"] = json!(2);
    input[6]["r"]["idx"] = json!(3);
    input.push(another);
    let parsed = parse(input).unwrap();
    assert!(parsed.output.messages.iter().any(|r| matches!(
        &r.message.blocks[0],
        Block::ToolResult { call_id: None, .. }
    )));
}

#[test]
fn nested_steps_keep_outer_role_and_source_millisecond_clock() {
    let input = vec![
        header(),
        row(
            "trajectory_meta",
            1,
            json!({"trajectory_id":"one","cascade_id":"alias"}),
        ),
        row(
            "steps",
            1,
            json!({"idx":0,"step_type":"message","step_format":"v1","step_payload":{"role":"user","occurred_at_ms":1700000000123_i64,"payload":{"text":"nested text"}}}),
        ),
    ];
    let parsed = parse(input).unwrap();
    let message = &parsed.output.messages[0].message;
    assert_eq!(message.role, Role::User);
    assert_eq!(message.text(), "nested text");
    assert_eq!(message.at.unwrap().unix_ms(), 1700000000123);
}

#[test]
fn structural_exit_status_and_nonempty_errors_are_not_unknown() {
    for (extra, expected) in [
        (
            json!({"exit_code":7}),
            ToolOutcome::Error {
                reason: Some(polylogue_model::ErrorReason::ExitCode(7)),
            },
        ),
        (
            json!({"error":"failed"}),
            ToolOutcome::Error { reason: None },
        ),
        (
            json!({"error":{"reason":"failed"}}),
            ToolOutcome::Error { reason: None },
        ),
        (
            json!({"error_details":["failed"]}),
            ToolOutcome::Error { reason: None },
        ),
        (
            json!({}),
            ToolOutcome::Unknown {
                reason: UnknownReason::StatusFieldUnreadable,
            },
        ),
    ] {
        let mut result = json!({"idx":0,"step_type":"tool_result","step_format":"v1","step_payload":{"output":"result"}});
        result
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let parsed = parse(vec![
            header(),
            row(
                "trajectory_meta",
                1,
                json!({"trajectory_id":"one","cascade_id":"alias"}),
            ),
            row("steps", 1, result),
        ])
        .unwrap();
        assert!(
            matches!(&parsed.output.messages[0].message.blocks[0],Block::ToolResult{outcome,..} if outcome==&expected)
        );
    }
}

#[test]
fn parent_aliases_collapse_and_conflicting_parents_stay_unresolved() {
    for ambiguous in [false, true] {
        let mut input = vec![
            header(),
            row(
                "trajectory_meta",
                1,
                json!({"trajectory_id":"parent","cascade_id":"parent-alias"}),
            ),
            row(
                "trajectory_meta",
                2,
                json!({"trajectory_id":"child","cascade_id":"child-alias"}),
            ),
            row(
                "parent_references",
                1,
                json!({"cascade_id":"child","parent_id":"parent"}),
            ),
            row(
                "parent_references",
                2,
                json!({"cascade_id":"child-alias","parent_id":"parent-alias"}),
            ),
        ];
        if ambiguous {
            input.push(row(
                "parent_references",
                3,
                json!({"cascade_id":"child","parent_id":"other"}),
            ));
        }
        let parsed = parse(input).unwrap();
        if ambiguous {
            assert!(parsed.output.links.is_empty());
        } else {
            assert!(
                parsed
                    .output
                    .links
                    .iter()
                    .all(|l| l.parent.to_string() == "antigravity:parent"
                        && l.child.to_string() == "antigravity:child")
            );
        }
    }
}

#[test]
fn unnamed_trajectories_do_not_take_native_or_summary_ids() {
    for table in ["trajectory_meta", "conversation_summaries"] {
        let other = if table == "trajectory_meta" {
            json!({"trajectory_id":"conversation","cascade_id":"native"})
        } else {
            json!({"cascade_id":"conversation","title":"Unrelated summary"})
        };
        let parsed = parse(vec![
            header(),
            row(
                "trajectory_meta",
                1,
                json!({"trajectory_id":null,"cascade_id":null}),
            ),
            row(table, 2, other),
        ])
        .unwrap();
        let ids: BTreeSet<_> = parsed
            .output
            .sessions
            .iter()
            .map(|s| s.id.to_string())
            .collect();
        assert_eq!(ids.len(), 2, "{ids:?}");
        assert!(ids.contains("antigravity:conversation"));
        assert!(ids.contains("antigravity:conversation:trajectory"));
        if table == "conversation_summaries" {
            assert!(
                parsed
                    .output
                    .events
                    .iter()
                    .any(|e| e.kind == "antigravity_unmatched_summary"
                        && e.data["title"] == "Unrelated summary")
            );
        }
    }
}

#[test]
fn several_unnamed_trajectories_are_refused_instead_of_merged() {
    let unnamed = json!({"trajectory_id":null,"cascade_id":null});
    assert!(
        parse(vec![
            header(),
            row("trajectory_meta", 1, unnamed.clone()),
            row("trajectory_meta", 2, unnamed)
        ])
        .is_err()
    );
}

#[test]
fn utf8_json_blobs_use_the_current_sqlite_row_encoding() {
    use base64::Engine as _;
    for payload in [
        base64::engine::general_purpose::STANDARD
            .encode(br#"{"role":"user","text":"from a blob"}"#),
        "not base64!".into(),
    ] {
        let valid = payload != "not base64!";
        let parsed = parse(vec![header(), row("trajectory_meta",1,json!({"trajectory_id":"one","cascade_id":"alias"})),
            row("steps",1,json!({"idx":0,"step_type":"message","step_format":"v1","step_payload":{"base64":payload}}))]).unwrap();
        if valid {
            assert_eq!(parsed.output.messages[0].message.text(), "from a blob");
        } else {
            assert!(parsed.output.messages.is_empty());
            assert!(
                parsed
                    .output
                    .events
                    .iter()
                    .any(|e| e.kind == "antigravity_unsupported_step")
            );
        }
    }
}

#[test]
fn numeric_tool_ids_pair_across_an_intervening_message() {
    for id in [0, 7] {
        let parsed = parse(vec![header(),row("trajectory_meta",1,json!({"trajectory_id":"one","cascade_id":"alias"})),
            row("steps",1,json!({"idx":0,"step_type":"tool_call","step_format":"v1","step_payload":{"tool_name":"shell","tool_id":id,"command":"printf hi"}})),
            row("steps",2,json!({"idx":1,"step_type":"assistant","step_format":"v1","step_payload":{"text":"waiting"}})),
            row("steps",3,json!({"idx":2,"step_type":"tool_result","step_format":"v1","step_payload":{"tool_name":"shell","tool_id":id,"output":"hi","status":"success"}}))]).unwrap();
        let expected = id.to_string();
        assert!(
            matches!(&parsed.output.messages[0].message.blocks[0],Block::ToolCall{call_id:Some(found),..} if found==&expected)
        );
        assert!(
            matches!(&parsed.output.messages[2].message.blocks[0],Block::ToolResult{call_id:Some(found),..} if found==&expected)
        );
    }
}

#[test]
fn an_explicit_tool_message_is_not_assistant_prose() {
    for role in ["tool", "TOOL"] {
        let parsed=parse(vec![header(),row("trajectory_meta",1,json!({"trajectory_id":"one","cascade_id":"alias"})),
            row("steps",1,json!({"idx":0,"step_type":"message","step_format":"v1","step_payload":{"role":role,"text":"tool output"}}))]).unwrap();
        let message = &parsed.output.messages[0].message;
        assert_eq!(message.role, Role::User);
        assert_eq!(message.prompt_kind, Some(PromptKind::Harness));
        assert!(
            matches!(&message.blocks[0],Block::ToolResult{output:Some(text),..} if text=="tool output")
        );
    }
}

#[test]
fn native_summary_update_time_does_not_become_a_message_clock() {
    let mut input = records();
    input[2]["r"]["last_modified_time"] = json!("2026-03-05T04:21:34Z");
    let parsed = parse(input).unwrap();
    let event = parsed
        .output
        .events
        .iter()
        .find(|e| e.kind == "antigravity_conversation_summary")
        .unwrap();
    assert_eq!(
        event.at,
        crate::time::rfc3339(&json!("2026-03-05T04:21:34Z"))
    );
    assert!(
        parsed
            .output
            .messages
            .iter()
            .all(|row| row.message.at.is_none())
    );
}

#[test]
fn malformed_row_dump_frames_are_counted_as_unreadable() {
    for bad in [
        json!({"future":true}),
        json!({"t":"steps","k":[1],"r":9}),
        json!({"t":"steps","r":{}}),
    ] {
        let parsed = parse(vec![
            header(),
            row(
                "trajectory_meta",
                1,
                json!({"trajectory_id":"one","cascade_id":"alias"}),
            ),
            bad,
        ])
        .unwrap();
        assert_eq!(parsed.counts.unreadable.values().sum::<u64>(), 1);
    }
}

#[test]
fn a_parent_row_without_a_child_does_not_assert_an_edge() {
    let parsed = parse(vec![
        header(),
        row(
            "trajectory_meta",
            1,
            json!({"trajectory_id":"one","cascade_id":"alias"}),
        ),
        row("parent_references", 1, json!({"parent_id":"parent"})),
    ])
    .unwrap();
    assert!(parsed.output.links.is_empty());
    assert!(
        parsed
            .output
            .events
            .iter()
            .any(|e| e.kind == "antigravity_parent_reference" && e.session.is_none())
    );
}

#[test]
fn a_native_table_named_schema_is_not_a_schema_header() {
    let parsed = parse(vec![
        header(),
        row(
            "trajectory_meta",
            1,
            json!({"trajectory_id":"one","cascade_id":"alias"}),
        ),
        row("schema", 1, json!({"value":7})),
    ])
    .unwrap();
    assert!(
        parsed
            .output
            .events
            .iter()
            .any(|e| e.kind == "antigravity_database_row"
                && e.session.is_none()
                && e.data["table"] == "schema"
                && e.data["row"]["value"] == 7)
    );
}

#[test]
fn steps_without_metadata_are_not_reported_as_an_empty_trajectory() {
    let parsed = parse(vec![
        header(),
        row(
            "steps",
            1,
            json!({"idx":0,"step_type":"user","step_format":"v1","step_payload":{"text":"kept"}}),
        ),
    ])
    .unwrap();
    assert_eq!(parsed.output.messages[0].message.text(), "kept");
    assert!(
        !parsed
            .output
            .events
            .iter()
            .any(|e| e.kind == "antigravity_trajectory_empty")
    );
}

#[test]
fn null_step_ids_use_the_distinct_row_indices() {
    let parsed = parse(vec![
        header(),
        row("trajectory_meta", 1, json!({"trajectory_id":"one","cascade_id":"alias"})),
        row("steps", 1, json!({"step_id":null,"id":null,"idx":0,"step_type":"user","step_format":"v1","step_payload":{"text":"first"}})),
        row("steps", 2, json!({"step_id":null,"id":null,"idx":1,"step_type":"user","step_format":"v1","step_payload":{"text":"second"}})),
    ]).unwrap();
    assert_eq!(parsed.output.messages.len(), 2);
    assert_ne!(
        parsed.output.messages[0].message.id,
        parsed.output.messages[1].message.id
    );
}

#[test]
fn colon_containing_session_and_step_ids_remain_distinct() {
    let parsed = parse(vec![
        header(),
        row("trajectory_meta", 1, json!({"trajectory_id":"a","cascade_id":"first"})),
        row("trajectory_meta", 2, json!({"trajectory_id":"a:step:b","cascade_id":"second"})),
        row("steps", 1, json!({"trajectory_id":"a","step_id":"b:step:c","idx":0,"step_type":"user","step_format":"v1","step_payload":{"text":"first"}})),
        row("steps", 2, json!({"trajectory_id":"a:step:b","step_id":"c","idx":0,"step_type":"user","step_format":"v1","step_payload":{"text":"second"}})),
    ]).unwrap();
    assert_eq!(parsed.output.messages.len(), 2);
    assert_ne!(
        parsed.output.messages[0].message.id,
        parsed.output.messages[1].message.id
    );
}

#[test]
fn current_store_headers_are_recognized_before_the_schema_is_visible() {
    let mut value = header();
    value["db"] = json!("trajectory.db");
    value["at_ms"] = json!(1700000000000_i64);
    value["source"] = json!({"machine":"synthetic","path":"/example/trajectory.db"});
    let bytes = serde_json::to_vec(&value).unwrap();
    let head = &bytes[..64];
    assert!(!String::from_utf8_lossy(head).contains("trajectory_meta"));
    let path = Path::new("exports/antigravity-database/source/trajectory.db.jsonl");
    assert!(recognize(path, head));
    assert!(recognize(Path::new("trajectory.db.jsonl"), &bytes));
    assert!(!recognize(path, b"not JSON"));
    assert!(!recognize(
        Path::new("exports/hermes/source/state.db.jsonl"),
        head
    ));
    assert!(!recognize(
        Path::new("exports/antigravity-database/source/notes.jsonl"),
        head
    ));
    assert!(parse(vec![value]).is_ok());
    assert!(parse(vec![json!({"schema":{}})]).is_err());
}
