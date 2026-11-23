use super::*;

fn summary(extra: Json) -> Json {
    let mut value = json!({
        "type":"system", "subtype":"stop_hook_summary", "uuid":"hook1",
        "sessionId":"s1", "timestamp":"2026-01-01T00:00:00Z"
    });
    value
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    value
}

#[test]
fn hook_fields_cite_context_blocks_and_keep_message() {
    let value = summary(json!({
        "hookErrors":["diagnostic"], "preventedContinuation":true,
        "stopReason":"reason", "hookAdditionalContext":["context", "", "context"]
    }));
    let bytes = jsonl(&[value]);
    let step = run(&main_file("s1"), &mut ParserState::default(), &bytes, 41).unwrap();
    let [row] = &step.output.messages[..] else {
        panic!("one summary message")
    };
    let stated = &step.output.hook_fields[&0];
    assert_eq!(
        stated
            .iter()
            .map(|c| (c.block, c.pointer.as_str()))
            .collect::<Vec<_>>(),
        [
            (3, "/hookAdditionalContext/0"),
            (4, "/hookAdditionalContext/1"),
            (5, "/hookAdditionalContext/2")
        ]
    );
    assert!(
        stated
            .iter()
            .all(|c| c.start == 41 && c.end == 41 + bytes.len() as u64 - 1)
    );
    // The complete original message is unchanged. Attribution lives
    // in Output only and has no effect on this source-version identity.
    let expected: Message = serde_json::from_value(json!({
        "id":"claude-code:hook1", "session":"claude-code:s1", "role":"system",
        "at":"2026-01-01T00:00:00Z", "at_quality":"intrinsic", "at_precision":"second",
        "prompt_kind":"injected_context", "blocks":[
            {"type":"text","text":"Stop hook error:\ndiagnostic"},
            {"type":"text","text":"Stop hook prevented continuation."},
            {"type":"text","text":"Stop hook reason:\nreason"},
            {"type":"text","text":"context"}, {"type":"text","text":""},
            {"type":"text","text":"context"}
        ]
    }))
    .unwrap();
    assert_eq!(row.message, expected);
    assert_eq!(
        row.message.content_hash().unwrap(),
        expected.content_hash().unwrap()
    );
    let encoded = serde_json::to_value(&step.output).unwrap();
    assert!(
        encoded["messages"][0]["message"]
            .get("hook_fields")
            .is_none()
    );
    let restored: Output = serde_json::from_value(encoded).unwrap();
    assert_eq!(restored.hook_fields, step.output.hook_fields);
}

#[test]
fn hook_fields_keep_source_and_output_indices() {
    let values = [
        assistant("a1", "msg_1", json!({"type":"text","text":"earlier"})),
        summary(json!({"hookAdditionalContext":[false,"context",null,"later"]})),
    ];
    let step = run(
        &main_file("s1"),
        &mut ParserState::default(),
        &jsonl(&values),
        0,
    )
    .unwrap();
    assert_eq!(step.output.messages.len(), 2);
    assert!(!step.output.hook_fields.contains_key(&0));
    let stated = &step.output.hook_fields[&1];
    assert_eq!(
        stated
            .iter()
            .map(|c| (c.block, c.pointer.as_str()))
            .collect::<Vec<_>>(),
        [
            (0, "/hookAdditionalContext/1"),
            (1, "/hookAdditionalContext/3")
        ]
    );
}

#[test]
fn hook_fields_ignore_tool_results_and_forged_values() {
    let mut values = conversation();
    values.extend([
        json!({"type":"user","uuid":"forged","sessionId":"s1",
            "message":{"role":"user","content":"context"},
            "injected_context":true,"hookAdditionalContext":["context"],
            "hook_fields":{"0":[{"block":0,"pointer":"/hookAdditionalContext/0"}]}}),
        summary(
            json!({"hookErrors":["context"],"preventedContinuation":true,"stopReason":"context"}),
        ),
    ]);
    let step = run(
        &main_file("s1"),
        &mut ParserState::default(),
        &jsonl(&values),
        0,
    )
    .unwrap();
    assert!(step.output.hook_fields.is_empty());
    assert!(
        step.output
            .messages
            .iter()
            .any(|r| r.message.prompt_kind == Some(PromptKind::Harness))
    );
}

#[test]
fn hook_fields_follow_their_row_when_a_response_is_joined() {
    let values = [
        assistant("a1", "msg_1", json!({"type":"text","text":"earlier"})),
        summary(json!({"hookAdditionalContext":["context"]})),
        assistant("a2", "msg_1", json!({"type":"text","text":"later"})),
    ];
    let step = run(
        &main_file("s1"),
        &mut ParserState::default(),
        &jsonl(&values),
        0,
    )
    .unwrap();
    let stated = step.output.hook_fields[&1].clone();
    let step = crate::chain_names::tests::settled(&[step]).remove(0);
    assert_eq!(step.output.messages.len(), 2);
    assert_eq!(step.output.messages[0].message.text(), "earlier\n\nlater");
    assert_eq!(step.output.hook_fields.len(), 1);
    assert_eq!(step.output.hook_fields[&1], stated);
    assert_eq!(
        step.output.messages[1].message.id,
        message_id("hook1").unwrap()
    );
}
