use super::*;

#[test]
fn requests_round_trip_with_an_op_tag() {
    let r = Request::Query {
        text: "repo:x".into(),
        cursor: None,
        page_size: None,
        why: false,
        session: None,
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
