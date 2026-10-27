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

#[test]
fn identifiers_and_cursors_survive_output_masking_but_authored_strings_do_not() {
    let token = "AKIAIOSFODNN7EXAMPLE";
    let id = format!("claude-code:{token}");
    let mut value = serde_json::json!({"session":{"id":id,"title":{"text":token}},"cursor":token,"next":token,"columns":[{"name":"id","reference":"message"},{"name":"text"}],"rows":[{"id":id,"text":token}],"value":{"id":id,"ref":id,"cursor":token}});
    output_values(&mut value, polylogue_render::mask);
    assert_eq!(value["session"]["id"], id);
    assert_eq!(value["rows"][0]["id"], id);
    assert_eq!(value["cursor"], token);
    assert_eq!(value["next"], token);
    assert_ne!(value["session"]["title"]["text"], token);
    assert_ne!(value["rows"][0]["text"], token);
    assert_ne!(value["value"]["id"], id);
    assert_ne!(value["value"]["ref"], id);
    assert_ne!(value["value"]["cursor"], token);
}
