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
fn a_context_request_carries_its_ask() {
    let v = serde_json::json!({
        "op": "context",
        "request_id": "r1",
        "ask": {"ask": "session_start", "cwd": "/x", "source": "resume"},
        "sender": "hook"
    });
    let r: Request = serde_json::from_value(v).unwrap();
    assert!(matches!(
        r,
        Request::Context {
            ask: Ask::SessionStart { .. },
            ..
        }
    ));
}

#[test]
fn response_serialization_keeps_source_values_keys_and_empty_fields() {
    let token = format!("sk-ant-api03-{}-AAAAAAAA", "A".repeat(94));
    let data = serde_json::json!({
        "nested": [token.clone(), ""], (token.clone()): {"code": 0, "known": false},
        "cursor": token.clone(), "authored": {"cursor": token.clone()}
    });
    let response = Response::Ok { data: data.clone() };
    let encoded = serde_json::to_vec(&response).unwrap();
    let decoded: Response = serde_json::from_slice(&encoded).unwrap();
    let Response::Ok { data: actual } = decoded else {
        panic!("expected success");
    };
    assert_eq!(actual, data);
    assert_eq!(
        actual["nested"][0].as_str().unwrap().as_bytes(),
        token.as_bytes()
    );
}

#[test]
fn response_errors_keep_credential_shaped_messages_and_details() {
    let token = format!("ghp_{}", "q".repeat(36));
    let response = Response::Error {
        kind: "synthetic".into(),
        message: token.clone(),
        detail: Some(serde_json::json!({"hint":token.clone()})),
    };
    let encoded = serde_json::to_value(response).unwrap();
    assert_eq!(encoded["error"]["message"], token);
    assert_eq!(encoded["error"]["hint"], token);
}
