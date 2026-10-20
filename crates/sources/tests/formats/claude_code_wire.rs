//! Claude Code wire fields the Python predecessor's tests pinned
//! (`test_claude_code_unread_wire_fields.py`), restated against this
//! parser's model: each fact lands where the model keeps it, and an absent
//! field invents nothing.

use crate::support::{blocks, events, jsonl, message, messages, parse};
use serde_json::{Value, json};

const PATH: &str = ".claude/projects/p/s1.jsonl";

fn record(kind: &str, uuid: &str, message: Value, extra: Value) -> Value {
    let mut r = json!({"type": kind, "uuid": uuid, "sessionId": "s1",
        "timestamp": "2026-03-01T10:00:00Z", "message": message});
    if let (Some(r), Some(extra)) = (r.as_object_mut(), extra.as_object()) {
        r.extend(extra.clone());
    }
    r
}

fn assistant(uuid: &str, content: Value, extra: Value) -> Value {
    record(
        "assistant",
        uuid,
        json!({"role": "assistant", "content": content}),
        extra,
    )
}

fn text(t: &str) -> Value {
    json!([{"type": "text", "text": t}])
}

fn run(records: &[Value]) -> Value {
    parse("claude-code", PATH, &jsonl(records))
}

#[test]
fn stop_reason_lands_on_the_assistant_message_and_is_not_invented() {
    let p = run(&[
        assistant("a1", text("done"), json!({})),
        record(
            "assistant",
            "a2",
            json!({"id": "m2", "role": "assistant", "content": text("x"), "stop_reason": "end_turn"}),
            json!({}),
        ),
    ]);
    assert!(message(&p, ":a1").get("stop_reason").is_none());
    assert_eq!(message(&p, ":a2")["stop_reason"], "end_turn");
}

#[test]
fn an_unfamiliar_stop_reason_is_kept_as_written() {
    // The predecessor nulled values outside a closed column; here the
    // string is the fact and readers interpret it.
    let p = run(&[record(
        "assistant",
        "a1",
        json!({"role": "assistant", "content": text("s"), "stop_reason": "new_provider_reason"}),
        json!({}),
    )]);
    assert_eq!(message(&p, ":a1")["stop_reason"], "new_provider_reason");
}

#[test]
fn the_plan_slug_is_one_event_per_name() {
    let p = run(&[
        record(
            "user",
            "u1",
            json!({"role": "user", "content": "hello"}),
            json!({"slug": "greedy-squishing-hamming"}),
        ),
        record(
            "user",
            "u2",
            json!({"role": "user", "content": "again"}),
            json!({"slug": "greedy-squishing-hamming"}),
        ),
    ]);
    let plans = events(&p, "plan");
    assert_eq!(plans.len(), 1);
    assert_eq!(plans[0]["data"]["slug"], "greedy-squishing-hamming");
    assert!(
        events(
            &run(&[record(
                "user",
                "u1",
                json!({"role": "user", "content": "hi"}),
                json!({})
            )]),
            "plan"
        )
        .is_empty()
    );
}

#[test]
fn the_team_name_is_captured_once_per_session() {
    let p = run(&[
        record(
            "user",
            "u1",
            json!({"role": "user", "content": "hello"}),
            json!({"teamName": "structural-consolidation"}),
        ),
        assistant(
            "a1",
            text("done"),
            json!({"teamName": "structural-consolidation"}),
        ),
    ]);
    let teams = events(&p, "team");
    assert_eq!(teams.len(), 1);
    assert_eq!(teams[0]["data"]["team"], "structural-consolidation");
    assert!(
        p["accounting"]["unread_fields"]
            .as_object()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn the_advisor_model_stays_on_its_turn() {
    let p = run(&[assistant(
        "a1",
        text("advised"),
        json!({"advisorModel": "claude-opus-4-7"}),
    )]);
    assert_eq!(
        message(&p, ":a1")["production"]["advisor_model"],
        "claude-opus-4-7"
    );
}

#[test]
fn the_api_request_id_lands_on_the_message_and_is_not_invented() {
    let p = run(&[assistant(
        "a1",
        text("hi"),
        json!({"requestId": "req_011CPuYvnLASUV8W7nChG4jH"}),
    )]);
    assert_eq!(
        message(&p, ":a1")["production"]["request_id"],
        "req_011CPuYvnLASUV8W7nChG4jH"
    );
    let p = run(&[assistant("a1", text("hi"), json!({}))]);
    assert!(message(&p, ":a1").get("production").is_none());
}

#[test]
fn a_refused_turn_s_quota_limits_are_a_used_up_window() {
    let quota = json!({"status": "rejected", "resetsAt": 1787464800, "rateLimitType": "seven_day"});
    let p = run(&[assistant("a1", text("ok"), json!({"quotaLimits": quota}))]);
    assert_eq!(
        message(&p, ":a1")["production"]["quota"],
        json!({"windows": [{"limit": "seven_day", "minutes": 10080, "used_percent": 100.0,
                            "resets_at": "2026-08-23T06:00:00Z", "status": "rejected"}]})
    );
    // Other states do not say how much is used; the five-hour window
    // starts with the first response after the last one ended.
    let quota =
        json!({"status": "allowed_warning", "resetsAt": 1787464800, "rateLimitType": "five_hour"});
    let p = run(&[assistant("a1", text("ok"), json!({"quotaLimits": quota}))]);
    assert_eq!(
        message(&p, ":a1")["production"]["quota"]["windows"][0],
        json!({"limit": "five_hour", "minutes": 300, "from_first_response": true,
               "resets_at": "2026-08-23T06:00:00Z", "status": "allowed_warning"})
    );
}

#[test]
fn a_file_edit_result_keeps_the_edit_beside_the_result() {
    let edit = json!({"filePath": "/tmp/x.py", "originalFile": "old contents\n", "oldString": "old",
        "newString": "new", "replaceAll": false, "userModified": true,
        "structuredPatch": [{"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 2, "lines": ["+x"]}]});
    let p = run(&[
        assistant(
            "a1",
            json!([{"type": "tool_use", "id": "e1", "name": "Edit", "input": {"file_path": "/tmp/x.py"}}]),
            json!({}),
        ),
        record(
            "user",
            "u1",
            json!({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "e1", "content": "applied", "is_error": false}]}),
            json!({"toolUseResult": edit}),
        ),
    ]);
    let result = blocks(&p)
        .into_iter()
        .find(|b| b["type"] == "tool_result")
        .unwrap();
    assert_eq!(result["data"], edit);
    assert_eq!(result["outcome"]["kind"], "ok");
}

#[test]
fn a_background_launch_is_not_read_as_success() {
    let p = run(&[
        assistant(
            "a1",
            json!([{"type": "tool_use", "id": "b1", "name": "Bash", "input": {"command": "sleep 100 &"}}]),
            json!({}),
        ),
        record(
            "user",
            "u1",
            json!({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "b1", "content": "started", "is_error": false}]}),
            json!({"toolUseResult": {"backgroundTaskId": "bgtask-1"}}),
        ),
    ]);
    let result = blocks(&p)
        .into_iter()
        .find(|b| b["type"] == "tool_result")
        .unwrap();
    assert_eq!(
        result["outcome"],
        json!({"kind": "unknown", "reason": "running_in_background"})
    );
}

#[test]
fn a_denial_keeps_its_kind_beside_the_error() {
    let p = run(&[record(
        "user",
        "u1",
        json!({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": true, "content": "denied"}]}),
        json!({"toolDenialKind": "permission-rule"}),
    )]);
    let result = blocks(&p)
        .into_iter()
        .find(|b| b["type"] == "tool_result")
        .unwrap();
    assert_eq!(
        result["outcome"],
        json!({"kind": "error", "reason": {"kind": "denied", "detail": "permission-rule"}})
    );
}

#[test]
fn capability_attribution_is_read_per_turn() {
    let p = run(&[
        assistant(
            "a1",
            text("replayed prefix"),
            json!({"attributionAgent": "triage",
            "attributionMcpServer": "context7", "attributionMcpTool": "resolve-library-id"}),
        ),
        record(
            "assistant",
            "a2",
            json!({"id": "m2", "role": "assistant", "content": text("divergent tail")}),
            json!({"attributionAgent": "fork", "attributionPlugin": "feature-dev"}),
        ),
    ]);
    let a1 = &message(&p, ":a1")["production"]["attribution"];
    assert_eq!(a1["agent"], "triage");
    assert_eq!(a1["mcp_server"], "context7");
    assert_eq!(a1["mcp_tool"], "resolve-library-id");
    let a2 = &message(&p, ":a2")["production"]["attribution"];
    assert_eq!(a2["agent"], "fork");
    assert_eq!(
        a2["plugin"], "feature-dev",
        "a plugin without a skill is kept"
    );
    assert!(a2.get("skill").is_none());
}

#[test]
fn attribution_survives_an_empty_api_error_turn() {
    let p = run(&[assistant(
        "a1",
        json!([]),
        json!({"attributionSkill": "code-review", "isApiErrorMessage": true}),
    )]);
    assert_eq!(
        message(&p, ":a1")["production"]["attribution"]["skill"],
        "code-review"
    );
}

#[test]
fn a_blank_attribution_is_not_a_capability() {
    let p = run(&[assistant("a1", text("hi"), json!({"attributionSkill": ""}))]);
    assert!(message(&p, ":a1").get("production").is_none());
}

#[test]
fn an_api_error_turn_carries_its_kind_and_status() {
    let p = run(&[record(
        "assistant",
        "a1",
        json!({"role": "assistant", "model": "<synthetic>", "content": text("API Error: 429 rate limited")}),
        json!({"isApiErrorMessage": true, "apiErrorStatus": 429, "apiErrorIsTransient": true,
               "error": "rate_limit_error", "errorDetails": "rate limited"}),
    )]);
    assert_eq!(
        message(&p, ":a1")["production"]["api_error"],
        json!({"kind": "rate_limit_error", "status": 429})
    );
}

#[test]
fn per_turn_effort_lands_as_the_turn_effort() {
    let p = run(&[assistant("a1", text("x"), json!({"perTurnEffort": "high"}))]);
    assert_eq!(message(&p, ":a1")["production"]["effort"], "high");
}

#[test]
fn hook_summaries_keep_their_errors_and_injected_context() {
    let p = run(&[
        json!({"type": "system", "subtype": "stop_hook_summary", "uuid": "sy1", "sessionId": "s1",
        "timestamp": "2026-03-01T10:00:00Z", "hookErrors": ["synthetic hook failed"],
        "preventedContinuation": true, "hookAdditionalContext": ["Synthetic injected context."]}),
    ]);
    assert!(messages(&p).is_empty());
    let hooks = events(&p, "stop_hook_summary");
    assert_eq!(hooks.len(), 1);
    assert_eq!(
        hooks[0]["data"]["hookErrors"],
        json!(["synthetic hook failed"])
    );
    assert_eq!(hooks[0]["data"]["preventedContinuation"], true);
    assert_eq!(
        hooks[0]["data"]["hookAdditionalContext"],
        json!(["Synthetic injected context."])
    );
}

#[test]
fn a_pr_link_record_is_kept_whole_as_an_event() {
    let p = run(&[
        json!({"type": "pr-link", "sessionId": "s1", "timestamp": "2026-03-01T10:00:00Z",
        "prNumber": 3126, "prUrl": "https://github.com/example/repo/pull/3126", "prRepository": "example/repo"}),
    ]);
    let links = events(&p, "pr_link");
    assert_eq!(links.len(), 1);
    assert_eq!(links[0]["data"]["prNumber"], 3126);
    assert_eq!(links[0]["data"]["prRepository"], "example/repo");
}

fn user(uuid: &str, content: Value) -> Value {
    record(
        "user",
        uuid,
        json!({"role": "user", "content": content}),
        json!({}),
    )
}

fn tool_result(id: &str, content: &str, is_error: Option<bool>) -> Value {
    let mut b = json!({"type": "tool_result", "tool_use_id": id, "content": content});
    if let Some(e) = is_error {
        b["is_error"] = json!(e);
    }
    b
}

/// A user-role envelope holding only tool results is the harness writing,
/// not the operator; one with typed text beside them is still the operator.
#[test]
fn tool_result_envelopes_are_not_counted_as_typed_prompts() {
    let p = run(&[
        user("u1", json!([tool_result("t1", "out", None)])),
        user(
            "u2",
            json!([
                tool_result("t1", "out", None),
                tool_result("t2", "out", None)
            ]),
        ),
        user("u3", text("typed prose")),
        user(
            "u4",
            json!([{"type": "text", "text": "hi"}, tool_result("t3", "out", None)]),
        ),
    ]);
    let kind = |s| message(&p, s)["prompt_kind"].clone();
    assert_eq!(kind(":u1"), "harness");
    assert_eq!(kind(":u2"), "harness");
    assert_eq!(kind(":u3"), "prose");
    assert_eq!(kind(":u4"), "prose");
    assert!(messages(&p).iter().all(|m| m["role"] == "user"));
}

/// The outcome comes from the structural field, never from the prose.
#[test]
fn result_prose_never_decides_the_outcome() {
    let p = run(&[user(
        "u1",
        json!([
            tool_result(
                "t1",
                "Done! Success, completed with no errors, all good.",
                Some(true)
            ),
            tool_result(
                "t2",
                "Handled 3 errors gracefully; no failures remain.",
                Some(false)
            ),
            tool_result(
                "t3",
                "Error: command failed with exit code 1, fatal error occurred",
                None
            ),
        ]),
    )]);
    let outcomes: Vec<_> = blocks(&p)
        .iter()
        .map(|b| b["outcome"]["kind"].clone())
        .collect();
    // An absent `is_error` is the API's default, false: Claude Code omits it
    // on successful results and has not been seen omitting it on a failure.
    assert_eq!(outcomes, [json!("error"), json!("ok"), json!("ok")]);
}
