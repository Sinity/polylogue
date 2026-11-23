use super::*;
#[test]
fn parent_deadline_surrounds_the_unchanged_request() {
    let request: command_surface::RunnableRequest = serde_json::from_value(serde_json::json!({
        "operation":"brief", "arguments":{"cwd":"/synthetic"}, "request_id":"bounded-prepare", "deadline":null, "budget_tokens":200
    })).unwrap();
    let wire = serde_json::to_string(&Outgoing {
        request: &request,
        caller: CallerEvidence::default(),
        parent_deadline_ms: Some(123),
    })
    .unwrap();
    let Incoming {
        request: accepted,
        parent_deadline_ms: parent,
        ..
    } = incoming(&wire).unwrap();
    assert_eq!(accepted, request);
    assert_eq!(parent, Some(123));
}

#[test]
fn bare_request_needs_no_caller_details() {
    let request = command_surface::Request {
        operation: "status".into(),
        arguments: serde_json::json!({}),
        request_id: "bare-shared-envelope".into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    let Incoming {
        request: accepted,
        caller,
        parent_deadline_ms,
    } = incoming(&serde_json::to_string(&request).unwrap()).unwrap();
    assert_eq!(accepted, request);
    assert_eq!(parent_deadline_ms, None);
    assert_eq!(caller, CallerEvidence::default());
}

#[test]
fn a_malformed_request_line_says_what_is_wrong() {
    let not_json = incoming("{\"operation\":").err().unwrap().to_string();
    assert!(not_json.contains("EOF"), "{not_json}");
    let no_operation = incoming("{\"arguments\":{}}").err().unwrap().to_string();
    assert!(no_operation.contains("operation"), "{no_operation}");
    assert!(incoming("[1]").is_err());
}
