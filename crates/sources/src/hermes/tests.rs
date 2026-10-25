use super::*;
use polylogue_model::{Block, ErrorReason, Role, ToolOutcome, UnknownReason};
use serde_json::json;

fn header(profile: &str) -> Json {
    json!({"db":"state.db","at_ms":9,"schema":{"sessions":"CREATE TABLE sessions(id TEXT PRIMARY KEY)","messages":"CREATE TABLE messages(id INTEGER PRIMARY KEY)"},"source":{"machine":"synthetic-machine","path":format!("/synthetic/{profile}/state.db")}})
}
fn session(id: &str, parent: Option<&str>, end: Option<&str>) -> Json {
    json!({"t":"sessions","k":[id],"r":{"id":id,"source":"cli","model":"supplied-model","model_config":"{}","parent_session_id":parent,"end_reason":end,"started_at":0,"title":"Native title"}})
}
fn message(id: i64, session: &str, role: &str, content: &str) -> Json {
    json!({"t":"messages","k":[id],"r":{"id":id,"session_id":session,"role":role,"content":content,"timestamp":1,"active":1,"compacted":0}})
}
fn parse(records: Vec<Json>) -> Step {
    let input = records
        .into_iter()
        .map(|record| format!("{record}\n"))
        .collect::<String>();
    crate::export::fixture::parse(
        &mut open(Path::new("state.db.jsonl")).unwrap(),
        input.as_bytes(),
    )
    .unwrap()
}

#[test]
fn native_order_is_insertion_id_not_table_export_or_timestamp_order() {
    let mut second = message(2, "s", "assistant", "second");
    second["r"]["timestamp"] = json!(0);
    let parsed = parse(vec![
        header("a"),
        second,
        message(1, "s", "user", "first"),
        session("s", None, None),
    ]);
    assert_eq!(
        parsed
            .output
            .messages
            .iter()
            .map(|row| row.message.text())
            .collect::<Vec<_>>(),
        ["first", "second"]
    );
    assert_eq!(parsed.output.messages[0].message.role, Role::User);
    assert_eq!(
        parsed.output.messages[1].message.model.as_deref(),
        Some("supplied-model")
    );
    assert_eq!(
        parsed.output.messages[1].message.at.unwrap().to_string(),
        "1970-01-01T00:00:00Z"
    );
    assert_eq!(
        parsed.output.messages[1].message.parent.as_ref(),
        Some(&parsed.output.messages[0].message.id)
    );
}

#[test]
fn rowid_reinsertion_keeps_content_identity_and_equal_turns_keep_occurrences() {
    let first = parse(vec![
        header("a"),
        session("s", None, None),
        message(1, "s", "user", "same"),
    ]);
    let replaced = parse(vec![
        header("a"),
        session("s", None, None),
        message(99, "s", "user", "same"),
    ]);
    assert_eq!(
        first.output.messages[0].message.id,
        replaced.output.messages[0].message.id
    );
    let repeated = parse(vec![
        header("a"),
        session("s", None, None),
        message(1, "s", "user", "same"),
        message(2, "s", "user", "same"),
    ]);
    assert_eq!(repeated.output.messages.len(), 2);
    assert_ne!(
        repeated.output.messages[0].message.id,
        repeated.output.messages[1].message.id
    );
    assert!(
        repeated.output.messages[0]
            .message
            .id
            .native()
            .as_str()
            .contains(":~")
    );
}

#[test]
fn profiles_are_retained_bindings_and_unbound_or_conflicting_exports_do_not_guess() {
    let records = vec![session("s", None, None), message(1, "s", "user", "body")];
    let a = parse([vec![header("a")], records.clone()].concat());
    let b = parse([vec![header("b")], records.clone()].concat());
    assert_ne!(a.output.sessions[0].id, b.output.sessions[0].id);
    assert_ne!(
        a.output.messages[0].message.id,
        b.output.messages[0].message.id
    );
    let mut unbound = header("a");
    unbound.as_object_mut().unwrap().remove("source");
    let without_source = parse([vec![unbound], records.clone()].concat());
    assert!(without_source.output.sessions.is_empty());
    assert!(!without_source.accounting.unknown.is_empty());
    let conflicting = parse([vec![header("a"), header("b")], records].concat());
    assert!(conflicting.output.messages.is_empty());
    assert!(!conflicting.accounting.unknown.is_empty());
}

#[test]
fn latest_positive_rows_and_gone_keep_the_last_known_message_without_changing_its_raw_record() {
    let parsed = parse(vec![
        header("a"),
        session("s", None, None),
        message(1, "s", "user", "old"),
        message(1, "s", "user", "new"),
        json!({"t":"messages","k":[1],"gone":true}),
    ]);
    assert_eq!(parsed.output.messages.len(), 1);
    assert_eq!(parsed.output.messages[0].message.text(), "new");
    assert_eq!(parsed.accounting.framed, 5);
    assert_eq!(parsed.accounting.unknown.values().sum::<u64>(), 2);
}

#[test]
fn compression_ancestry_uses_nonempty_ancestor_and_its_actual_leaf() {
    let parsed = parse(vec![
        header("a"),
        session("child", Some("empty"), None),
        session("empty", Some("parent"), Some("compression")),
        session("parent", None, Some("compression")),
        message(1, "parent", "user", "retained ancestor"),
        message(2, "child", "assistant", "own tail"),
    ]);
    assert_eq!(parsed.output.sessions.len(), 2);
    let link = parsed
        .output
        .links
        .iter()
        .find(|link| link.child.native().as_str().starts_with("child@"))
        .unwrap();
    assert_eq!(link.kind, LinkKind::Continuation);
    assert!(link.parent.native().as_str().starts_with("parent@"));
    assert_eq!(
        link.at.as_ref(),
        Some(&parsed.output.messages[0].message.id)
    );
}

#[test]
fn source_prose_and_codex_reasoning_reach_blocks_without_duplicating_known_projections() {
    let mut row = message(1, "s", "assistant", "");
    row["r"]["reasoning_content"] = json!("primary reasoning");
    row["r"]["codex_message_items"]=json!(json!([{"type":"message","content":[{"type":"output_text","text":"only native item prose"}]}]).to_string());
    row["r"]["codex_reasoning_items"]=json!(json!([{"type":"reasoning","summary":[{"type":"summary_text","text":"native summary"}],"content":[{"type":"reasoning_text","text":"native detail"}],"encrypted_content":"opaque encrypted evidence"}]).to_string());
    let parsed = parse(vec![header("a"), session("s", None, None), row.clone()]);
    let blocks = &parsed.output.messages[0].message.blocks;
    assert!(
        blocks
            .iter()
            .any(|block| matches!(block,Block::Text{text} if text=="only native item prose"))
    );
    assert!(blocks.iter().any(|block|matches!(block,Block::Thinking{text:Some(text),signature:Some(signature),redacted:true} if text.contains("native summary")&&text.contains("native detail")&&signature=="opaque encrypted evidence")));
    row["r"]["content"] = json!("only native item prose");
    let parsed = parse(vec![header("a"), session("s", None, None), row]);
    assert_eq!(
        parsed.output.messages[0]
            .message
            .blocks
            .iter()
            .filter(|block| matches!(block, Block::Text { .. }))
            .count(),
        1
    );
    assert!(
        parsed
            .output
            .events
            .iter()
            .all(|event| !event.data.to_string().contains("only native item prose")),
        "ordered text projections must not be duplicated into independent event masking"
    );
}

#[test]
fn tool_envelopes_preserve_status_types_zero_exit_notes_and_unknown_limits() {
    for (raw, expected) in [
        (r#"{"output":"ok","exit_code":0}"#, ToolOutcome::Ok),
        (
            r#"{"output":"bad","exit_code":7}"#,
            ToolOutcome::Error {
                reason: Some(ErrorReason::ExitCode(7)),
            },
        ),
        (
            r#"{"output":"unknown","error":false,"exit_code":0}"#,
            ToolOutcome::Unknown {
                reason: UnknownReason::StatusFieldUnreadable,
            },
        ),
        (
            r#"{"output":"waiting","status":"running"}"#,
            ToolOutcome::Unknown {
                reason: UnknownReason::RunningInBackground,
            },
        ),
        (
            r#"{"output":"denied","status":"blocked"}"#,
            ToolOutcome::Error {
                reason: Some(ErrorReason::Denied("blocked".into())),
            },
        ),
        (
            r#"{"output":"none"}"#,
            ToolOutcome::Unknown {
                reason: UnknownReason::StatusFieldUnreadable,
            },
        ),
    ] {
        let (blocks, _) = outcome::result(Some(&json!(raw)), Some("call".into()));
        let Block::ToolResult { outcome, data, .. } = &blocks[0] else {
            panic!("not a result")
        };
        assert_eq!(outcome, &expected);
        assert_eq!(
            data.as_ref(),
            Some(&serde_json::from_str::<Json>(raw).unwrap())
        );
    }
    let (blocks, _) = outcome::result(
        Some(&json!(
            "\0json:{\"output\":\"kept\",\"exit_code\":0}\n[Hint: synthetic note]"
        )),
        None,
    );
    assert!(matches!(&blocks[1],Block::Text{text} if text=="\n[Hint: synthetic note]"));
    let (_, reason) = outcome::result(Some(&json!("\0json:{broken")), None);
    assert_eq!(reason, Some("json_marker_not_decoded"));
}

#[test]
fn unknown_authorship_and_unreadable_times_remain_explicit_without_guesses() {
    let mut facts = session("s", None, None);
    facts["r"]["model"] = json!(7);
    facts["r"]["started_at"] = json!("not a time");
    let mut invalid_time = message(2, "s", "assistant", "kept");
    invalid_time["r"]["timestamp"] = json!(true);
    let parsed = parse(vec![
        header("a"),
        facts,
        message(1, "s", "alien", "raw unknown authorship"),
        invalid_time,
    ]);
    assert_eq!(parsed.output.messages.len(), 1);
    assert_eq!(parsed.output.messages[0].message.role, Role::Assistant);
    assert!(parsed.output.messages[0].message.model.is_none());
    assert!(parsed.output.messages[0].message.at.is_none());
    assert!(parsed.output.sessions[0].started.is_none());
    assert!(!parsed.accounting.unknown.is_empty());
    assert!(
        parsed
            .output
            .events
            .iter()
            .any(|event| event.kind == "hermes_session_diagnostic")
    );
    assert!(
        parsed
            .output
            .events
            .iter()
            .any(|event| event.kind == "hermes_message_diagnostic")
    );
}

#[test]
fn inactive_positive_rows_are_not_empty_compression_intermediaries() {
    let mut row = message(1, "s", "user", "retained inactive source");
    row["r"]["active"] = json!(0);
    let parsed = parse(vec![
        header("a"),
        session("s", Some("parent"), None),
        session("parent", None, Some("compression")),
        row,
    ]);
    assert_eq!(parsed.output.sessions.len(), 2);
    assert_eq!(parsed.output.messages.len(), 1);
    assert!(
        parsed
            .output
            .events
            .iter()
            .any(|event| event.kind == "active_path_empty"
                && event.after.is_none()
                && event.data.is_null())
    );
}

#[test]
fn native_function_calls_reuse_known_projection_ids_and_keep_independent_calls() {
    let mut row = message(1, "s", "assistant", "");
    row["r"]["tool_calls"] =
        json!(json!([{"id":"same","function":{"name":"terminal","arguments":"{}"}}]).to_string());
    row["r"]["codex_message_items"]=json!(json!([{"type":"function_call","call_id":"same","name":"terminal","arguments":"{}"},{"type":"function_call","call_id":"other","name":"read_file","arguments":"{\"path\":\"synthetic.txt\"}"},{"type":"function_call","call_id":"same","name":"terminal","arguments":"{\"command\":\"different supplied input\"}"}]).to_string());
    let parsed = parse(vec![header("a"), session("s", None, None), row]);
    let calls = parsed.output.messages[0]
        .message
        .blocks
        .iter()
        .filter_map(|block| {
            if let Block::ToolCall { call_id, .. } = block {
                call_id.as_deref()
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(calls, ["same", "other", "same"]);
}

#[test]
fn fork_marker_does_not_invent_an_anchor_from_a_parent_that_can_continue() {
    let mut child = session("child", Some("parent"), None);
    child["r"]["model_config"] = json!("{\"_branched_from\":\"parent\"}");
    let parsed = parse(vec![
        header("a"),
        session("parent", None, None),
        child,
        message(1, "parent", "user", "earlier parent"),
        message(2, "child", "assistant", "own fork tail"),
        message(3, "parent", "assistant", "parent continued later"),
    ]);
    assert_eq!(parsed.output.links.len(), 1);
    assert_eq!(parsed.output.links[0].kind, LinkKind::Fork);
    assert!(parsed.output.links[0].at.is_none());
    assert_eq!(parsed.output.messages.len(), 3);
}

#[test]
fn missing_or_malformed_activity_does_not_assert_an_empty_path() {
    for flags in [
        json!({}),
        json!({"active":null,"compacted":null}),
        json!({"active":false,"compacted":false}),
        json!({"active":"0","compacted":0}),
    ] {
        let mut row = message(1, "s", "user", "retained unknown source");
        row["r"].as_object_mut().unwrap().remove("active");
        row["r"].as_object_mut().unwrap().remove("compacted");
        row["r"]
            .as_object_mut()
            .unwrap()
            .extend(flags.as_object().unwrap().clone());
        let parsed = parse(vec![header("a"), session("s", None, None), row]);
        assert_eq!(parsed.output.messages.len(), 1);
        assert!(
            parsed
                .output
                .events
                .iter()
                .any(|e| e.kind == "active_path_unknown" && e.after.is_none())
        );
        assert!(
            !parsed
                .output
                .events
                .iter()
                .any(|e| matches!(e.kind.as_str(), "active_leaf" | "active_path_empty"))
        );
    }
}

#[test]
fn folded_rows_replace_earlier_rows_and_keep_original_message_ranges() {
    let records = vec![
        header("a"),
        session("s", None, None),
        message(1, "s", "user", "old row"),
        json!({"t":"auxiliary","k":[1],"r":{"note":"later observation"}}),
    ];
    let raw = records.iter().map(|r| format!("{r}\n")).collect::<String>();
    let last_start = raw[..raw.len() - 1].rfind('\n').unwrap() + 1;
    let mut parser = Database::default();
    let mut count = 0;
    parser
        .scan(
            "retained-input",
            &mut std::io::Cursor::new(raw.as_bytes()),
            &mut |_: Range<u64>, _: Vec<u8>, _: Option<&str>, _: &[u8]| {
                count += 1;
                Ok(())
            },
        )
        .unwrap();
    assert!(
        parser
            .begin(raw.len() as u64, None)
            .unwrap()
            .unwrap()
            .restates
    );
    assert_eq!(count, records.len());
    let parsed = parse(records);
    let row = &parsed.output.messages[0];
    assert!(row.end <= last_start as u64);
    let source: Json = serde_json::from_str(&raw[row.offset as usize..row.end as usize]).unwrap();
    assert_eq!(source["r"]["content"], "old row");
}
