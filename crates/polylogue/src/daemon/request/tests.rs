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
fn response_boundary_masks_errors_and_nested_details_without_erasing_empty_data() {
    let secret = format!("sk-ant-api03-{}-AAAAAAAA", "A".repeat(94));
    let response = Response::Error {
        kind: "synthetic".into(),
        message: secret.clone(),
        detail: Some(
            serde_json::json!({"nested": [secret.clone(), ""], "code": 0, "known": false}),
        ),
    }
    .masked();
    let value = serde_json::to_value(response).unwrap();
    assert!(!value.to_string().contains(&secret));
    assert_eq!(value["kind"], "synthetic");
    assert_eq!(value["detail"]["nested"][1], "");
    assert_eq!(value["detail"]["code"], 0);
    assert_eq!(value["detail"]["known"], false);
}
