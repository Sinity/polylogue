use super::*;

#[test]
fn requests_round_trip_with_an_op_tag() {
    let r = Request::Query {
        text: "repo:x".into(),
        cursor: None,
        page_size: None,
        why: false,
        session: None,
        fields: vec![],
    };
    let v = serde_json::to_value(&r).unwrap();
    assert_eq!(v["op"], "query");
    let back: Request = serde_json::from_value(v).unwrap();
    assert!(matches!(back, Request::Query { .. }));
}

#[test]
fn context_is_an_internal_plan_and_the_predecessor_wire_operation_is_retired() {
    let predecessor = serde_json::json!({
        "op":"context", "request_id":"r1", "ask":{"ask":"session_start","cwd":"/x","source":"resume"}, "sender":"hook"
    });
    assert!(serde_json::from_value::<Request>(predecessor).is_err());
    let arguments: crate::protocol::context::Brief =
        serde_json::from_value(serde_json::json!({"cwd":"/x","source":"resume"})).unwrap();
    assert_eq!(arguments.source, crate::briefing::StartSource::Resume);
}

#[test]
fn answer_serialization_keeps_source_values_keys_and_empty_fields() {
    let token = format!("sk-ant-api03-{}-AAAAAAAA", "A".repeat(94));
    let data = serde_json::json!({
        "nested": [token.clone(), ""], (token.clone()): {"code": 0, "known": false},
        "cursor": token.clone(), "authored": {"cursor": token.clone()}
    });
    let response = command_surface::Answer::ok(data.clone());
    let encoded = serde_json::to_vec(&response).unwrap();
    let decoded: command_surface::Answer<serde_json::Value> =
        serde_json::from_slice(&encoded).unwrap();
    let command_surface::Answer {
        outcome: command_surface::Outcome::Ok(actual),
        ..
    } = decoded
    else {
        panic!("expected success");
    };
    assert_eq!(actual, data);
    assert_eq!(
        actual["nested"][0].as_str().unwrap().as_bytes(),
        token.as_bytes()
    );
}

#[test]
fn answer_errors_keep_credential_shaped_messages_and_hints() {
    let token = format!("ghp_{}", "q".repeat(36));
    let mut failure = crate::protocol::failure("bad_request", token.clone());
    failure.hint = Some(token.clone());
    let response = command_surface::Answer::<serde_json::Value>::error(failure);
    let encoded = serde_json::to_value(response).unwrap();
    assert_eq!(encoded["error"]["message"], token);
    assert_eq!(encoded["error"]["hint"], token);
}
