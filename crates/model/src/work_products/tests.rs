use super::*;
use serde_json::json;

fn message(id: &str, role: &str, kind: &str, blocks: serde_json::Value) -> SelectedMessage {
    SelectedMessage {
        event: format!("event-{id}"),
        cost_nusd: None,
        message: serde_json::from_value(json!({
            "id": format!("codex:{id}"), "session": "codex:morrow", "role": role,
            "prompt_kind": kind, "blocks": blocks
        }))
        .unwrap(),
    }
}
fn call() -> serde_json::Value {
    json!([{"type":"tool_call","call_id":"call-1","tool":"shell","action":"shell","input":{},"command":"true","run_by_script":false}])
}
fn result() -> serde_json::Value {
    json!([{"type":"tool_result","call_id":"call-1","outcome":{"kind":"ok"},"output":"done","reported_duration_ms":27}])
}

#[test]
fn model_authored_prompt_is_not_an_answer_and_unknown_usage_stays_unknown() {
    let (calls, turns) = derive([
        message("task", "user", "model", json!([])),
        message("notice", "user", "harness", json!([])),
    ]);
    assert!(calls.is_empty());
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].facts.answer, None);
    assert!(turns[0].facts.abandoned);
    assert_eq!(turns[0].facts.cost_nusd, None);
    assert_eq!(turns[0].facts.cost_unknown, 0);
    let (_, turns) = derive([
        message("task", "user", "model", json!([])),
        message("answer", "assistant", "model", json!([])),
    ]);
    assert_eq!(
        turns[0].facts.answer.as_ref().map(ToString::to_string),
        Some("codex:answer".into())
    );
    assert_eq!(turns[0].facts.cost_unknown, 1);
    assert_eq!(turns[0].facts.input_unknown, 1);
}

#[test]
fn exact_call_pairing_survives_result_before_call_and_preserves_false_native_checks() {
    let mut invocation = message("call", "assistant", "model", call());
    invocation.message.attrs = serde_json::from_value(
        json!({"revision":"rev-x","check_name":"unit","check_passed":false}),
    )
    .unwrap();
    let (calls, _) = derive([message("result", "user", "harness", result()), invocation]);
    assert_eq!(calls.len(), 1);
    let call = &calls[0];
    assert_eq!(call.facts.outcome, ToolOutcome::Ok);
    assert_eq!(call.facts.reported_duration_ms, Some(27));
    assert_eq!(call.facts.check_passed, Some(false));
    assert_eq!(call.facts.revision.as_deref(), Some("rev-x"));
    assert_eq!(call.parents, ["event-call", "event-result"]);
}

#[test]
fn later_known_model_response_and_explicit_background_notice_complete_calls() {
    let (calls, _) = derive([
        message("call", "assistant", "model", call()),
        message("later", "assistant", "model", json!([])),
    ]);
    assert_eq!(calls[0].facts.outcome, ToolOutcome::NoResult);
    assert_eq!(
        calls[0].facts.outcome_reason.as_deref(),
        Some("passed_without_result")
    );
    let update = crate::BackgroundUpdate {
        session: SessionId::parse("codex:morrow").unwrap(),
        task_id: "task-1".into(),
        call_id: Some("call-1".into()),
        outcome: ToolOutcome::Ok,
        result_at: None,
        result_at_evidence: crate::TimeEvidence::UNKNOWN,
        reported_duration_ms: Some(52),
        output_file: None,
    };
    let (calls, _) = derive_with_background(
        [message("call", "assistant", "model", call())],
        [(update, "event-background".into())],
    );
    assert_eq!(calls[0].facts.outcome, ToolOutcome::Ok);
    assert_eq!(calls[0].facts.reported_duration_ms, Some(52));
    assert!(calls[0].parents.contains(&"event-background".into()));
}

#[test]
fn human_bursts_commands_and_overflow_keep_explicit_meaning() {
    let mut response = message("response", "assistant", "model", json!([]));
    response.message.usage.input = Some(u64::MAX);
    let mut second = message("second", "assistant", "model", json!([]));
    second.message.usage.input = Some(1);
    let (_, turns) = derive([
        message("a", "user", "prose", json!([])),
        message("b", "user", "prose", json!([])),
        response,
        second,
        message("command", "user", "command", json!([])),
    ]);
    assert_eq!(turns.len(), 2);
    assert_eq!(turns[0].facts.prompts.len(), 2);
    assert_eq!(turns[0].facts.input_tokens, None);
    assert_eq!(turns[0].facts.unavailable, ["input_tokens_overflow"]);
    assert!(turns[1].facts.slash_command);
    assert!(turns[1].facts.abandoned);
}

#[test]
fn retired_snapshot_keeps_only_session_identity_and_rejects_erased_metadata() {
    let id = SessionId::parse("codex:retired").unwrap();
    let mut snapshot = WorkSnapshot {
        reducer_version: VERSION,
        position: 9,
        session_id: id.clone(),
        session: None,
        retired: true,
        unavailable: Vec::new(),
        messages: Vec::new(),
        calls: Vec::new(),
        turns: Vec::new(),
        sources: Vec::new(),
        native: Vec::new(),
    };
    assert!(snapshot.validate().is_ok());
    snapshot.session = Some(crate::Session::new(id, crate::SessionKind::Main));
    assert_eq!(
        snapshot.validate(),
        Err("retired snapshot contains source facts")
    );
}

#[test]
fn a_generated_local_call_identity_is_not_native_pairing_evidence() {
    let invocation = message(
        "unkeyed",
        "assistant",
        "model",
        json!([
            {"type":"tool_call","tool":"shell","action":"shell","input":{}}
        ]),
    );
    let result = message(
        "result",
        "user",
        "harness",
        json!([
            {"type":"tool_result","call_id":"codex:unkeyed/0","outcome":{"kind":"ok"},"output":"unrelated"}
        ]),
    );
    for input in [[invocation.clone(), result.clone()], [result, invocation]] {
        let (calls, _) = derive(input);
        assert_eq!(calls.len(), 1);
        assert!(calls[0].facts.result.is_none());
        assert_eq!(calls[0].parents, ["event-unkeyed"]);
    }
}
