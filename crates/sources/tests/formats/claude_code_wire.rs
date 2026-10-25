//! Claude Code wire fields the Python version's tests pinned
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
    // The Python version nulled values outside a closed column; here the
    // string is the fact and readers decide what it means.
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
    assert!(p["counts"]["unread_fields"].as_object().unwrap().is_empty());
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
    let notice = message(&p, ":a1");
    assert_eq!(notice["role"], "assistant");
    assert_eq!(notice["prompt_kind"], "harness");
    assert!(notice.get("model").is_none());
    assert_eq!(notice["blocks"][0]["text"], "API Error: 429 rate limited");
}

#[test]
fn per_turn_effort_lands_as_the_turn_effort() {
    let p = run(&[assistant("a1", text("x"), json!({"perTurnEffort": "high"}))]);
    assert_eq!(message(&p, ":a1")["production"]["effort"], "high");
}

fn hook_summary(uuid: &str, extra: Value) -> Value {
    let mut native = json!({"type": "system", "subtype": "stop_hook_summary",
        "uuid": uuid, "sessionId": "s1", "timestamp": "2026-03-01T10:00:00Z"});
    native
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    native
}

#[test]
fn hook_summary_preserves_supplied_context_even_when_continuation_is_prevented() {
    let native = hook_summary(
        "sy1",
        json!({
        "hookErrors": ["synthetic hook failed"], "preventedContinuation": true,
        "hookAdditionalContext": ["  Zażółć 🦀\n", "", "\tsecond  "],
        "hookCount": 2, "hasOutput": true, "stopReason": "hook stopped"}),
    );
    let p = run(std::slice::from_ref(&native));
    let hooks = events(&p, "stop_hook_summary");
    assert_eq!(hooks.len(), 1);
    assert_eq!(hooks[0]["data"], native);
    let m = message(&p, ":sy1");
    assert_eq!(m["role"], "system");
    assert_eq!(m["prompt_kind"], "harness");
    assert!(m.get("model").is_none());
    assert_eq!(
        m["blocks"],
        json!([
        {"type": "text", "text": "Stop hook error:\nsynthetic hook failed"},
        {"type": "text", "text": "Stop hook prevented continuation."},
        {"type": "text", "text": "Stop hook reason:\nhook stopped"},
        {"type": "text", "text": "  Zażółć 🦀\n"},
        {"type": "text", "text": ""},
        {"type": "text", "text": "\tsecond  "}])
    );
    assert_eq!(p["counts"]["records"], 1);
}

#[test]
fn hook_context_uses_only_the_summary_carrier_without_content_deduplication() {
    let p = run(&[
        json!({"type": "attachment", "uuid": "attachment", "sessionId": "s1",
            "attachment": {"type": "hook_additional_context", "content": ["same"]}}),
        hook_summary("sy1", json!({"hookAdditionalContext": ["same", "same"]})),
        hook_summary("sy2", json!({"hookAdditionalContext": ["same"]})),
    ]);
    assert_eq!(messages(&p).len(), 2);
    assert_eq!(blocks(&p).len(), 3);
    assert_eq!(events(&p, "attachment").len(), 1);
    assert_eq!(events(&p, "stop_hook_summary").len(), 2);
}

#[test]
fn hook_context_does_not_promote_generic_output_or_another_system_subtype() {
    let p = run(&[
        hook_summary(
            "sy1",
            json!({"stdout": "ordinary output", "hookErrors": [],
            "hasOutput": true, "hookAdditionalContext": []}),
        ),
        hook_summary("sy2", json!({"content": "ordinary content"})),
        record(
            "system",
            "sy3",
            Value::Null,
            json!({"subtype": "informational",
            "hookAdditionalContext": ["unrecognized carrier"]}),
        ),
    ]);
    assert!(messages(&p).is_empty());
    assert_eq!(events(&p, "stop_hook_summary").len(), 2);
}

#[test]
fn hook_context_unsupported_shapes_keep_native_facts_and_diagnose_the_shape() {
    for value in [
        Value::Null,
        json!("scalar"),
        json!(true),
        json!({"text": "object"}),
        json!([null, "valid", 3]),
    ] {
        let native = hook_summary("sy1", json!({"hookAdditionalContext": value}));
        let p = run(std::slice::from_ref(&native));
        assert_eq!(events(&p, "stop_hook_summary")[0]["data"], native);
        assert_eq!(
            p["counts"]["unreadable"]
                .as_object()
                .unwrap()
                .values()
                .map(|v| v.as_u64().unwrap())
                .sum::<u64>(),
            1
        );
        assert_eq!(blocks(&p).len(), usize::from(value.is_array()));
    }
}

#[test]
fn hook_context_keeps_native_identity_and_parent_links_outside_model_responses() {
    let human = user("u1", json!("human prompt"));
    let response = |uuid: &str, text: &str, parent: &str| {
        record(
            "assistant",
            uuid,
            json!({"id": "response", "content": text}),
            json!({"parentUuid": parent}),
        )
    };
    let hook = hook_summary(
        "sy1",
        json!({"parentUuid": "a1", "hookAdditionalContext": ["context"], "hookErrors": ["native error"]}),
    );
    let bytes = jsonl(&[
        human.clone(),
        response("a1", "first", "u1"),
        hook.clone(),
        response("a2", "second", "sy1"),
        record(
            "user",
            "u2",
            json!({"content": "next human"}),
            json!({"parentUuid": "sy1"}),
        ),
    ]);
    let p = parse("claude-code", PATH, &bytes);
    assert_eq!(messages(&p).len(), 4);
    assert_eq!(message(&p, ":a1")["role"], "assistant");
    assert!(message(&p, ":a1").get("prompt_kind").is_none());
    assert_eq!(message(&p, ":a1")["blocks"].as_array().unwrap().len(), 2);
    assert_eq!(message(&p, ":sy1")["parent"], "claude-code:a1");
    assert_eq!(message(&p, ":u2")["parent"], "claude-code:sy1");
    assert_eq!(message(&p, ":u1")["prompt_kind"], "prose");
    let row = p["output"]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["message"]["id"] == "claude-code:sy1")
        .unwrap();
    let offset = jsonl(&[human, response("a1", "first", "u1")]).len();
    assert_eq!(row["offset"], offset);
    assert_eq!(row["end"], offset + jsonl(&[hook]).len() - 1);
}

#[test]
fn hook_errors_and_prevented_continuation_are_one_harness_record_with_context() {
    let native = hook_summary(
        "sy1",
        json!({"hookErrors": ["  synthetic failure λ\n", ""],
        "preventedContinuation": true, "stopReason": "  stopped 🦀\n",
        "hookAdditionalContext": ["  supplied context\n"], "hookCount": 2,
        "hookInfos": [{"command": "synthetic-check", "durationMs": 2}]}),
    );
    let p = run(std::slice::from_ref(&native));
    assert_eq!(messages(&p).len(), 1);
    let m = message(&p, ":sy1");
    assert_eq!(m["role"], "system");
    assert_eq!(m["prompt_kind"], "harness");
    assert!(m.get("model").is_none());
    assert_eq!(
        m["blocks"],
        json!([
        {"type":"text", "text":"Stop hook error:\n  synthetic failure λ\n"},
        {"type":"text", "text":"Stop hook error:\n"},
        {"type":"text", "text":"Stop hook prevented continuation."},
        {"type":"text", "text":"Stop hook reason:\n  stopped 🦀\n"},
        {"type":"text", "text":"  supplied context\n"}])
    );
    assert_eq!(events(&p, "stop_hook_summary")[0]["data"], native);
}

#[test]
fn hook_prevention_and_errors_do_not_require_additional_context_or_each_other() {
    for (extra, expected) in [
        (
            json!({"hookErrors":["synthetic failure"],"preventedContinuation":false}),
            "Stop hook error:\nsynthetic failure",
        ),
        (
            json!({"hookErrors":[],"preventedContinuation":true}),
            "Stop hook prevented continuation.",
        ),
    ] {
        let p = run(&[hook_summary("sy1", extra)]);
        assert_eq!(message(&p, ":sy1")["blocks"][0]["text"], expected);
    }
    let p = run(&[hook_summary(
        "sy1",
        json!({"hookErrors":[], "preventedContinuation":false,
        "stopReason":"error-looking prose", "stdout":"failure", "hasOutput":true,
        "hookInfos":[{"command":"exit 1"}], "hookCount":1}),
    )]);
    assert!(messages(&p).is_empty());
}

#[test]
fn malformed_hook_diagnostics_preserve_valid_context_and_native_evidence() {
    for extra in [
        json!({"hookErrors":"scalar"}),
        json!({"hookErrors":[false,"valid error"]}),
        json!({"preventedContinuation":"true"}),
        json!({"preventedContinuation":null}),
        json!({"preventedContinuation":true,"stopReason":3}),
        json!({"hookAdditionalContext":null,"hookErrors":["valid error"]}),
    ] {
        let native = hook_summary("sy1", extra);
        let p = run(std::slice::from_ref(&native));
        assert_eq!(events(&p, "stop_hook_summary")[0]["data"], native);
        assert_eq!(
            p["counts"]["unreadable"]
                .as_object()
                .unwrap()
                .values()
                .map(|v| v.as_u64().unwrap())
                .sum::<u64>(),
            1
        );
        if let Some(items) = native["hookErrors"].as_array() {
            for error in items.iter().filter_map(Value::as_str) {
                assert!(
                    blocks(&p)
                        .iter()
                        .any(|b| b["text"].as_str().unwrap().contains(error))
                );
            }
        }
    }
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

#[test]
fn a_synthetic_no_response_notice_is_harness_text_without_an_api_error() {
    let parsed = run(&[record(
        "assistant",
        "notice",
        json!({"role": "assistant", "model": "<synthetic>", "content": text("No response requested.")}),
        json!({}),
    )]);
    let notice = message(&parsed, ":notice");
    assert_eq!(notice["role"], "assistant");
    assert_eq!(notice["prompt_kind"], "harness");
    assert!(notice.get("model").is_none());
    assert!(notice["production"].get("api_error").is_none());
    assert_eq!(notice["blocks"][0]["text"], "No response requested.");
}

#[test]
fn ordinary_assistant_text_or_a_missing_model_does_not_mark_a_harness_notice() {
    for model in [
        None,
        Some("actual-model"),
        Some("<synthetic>-suffix"),
        Some("prefix-<synthetic>"),
    ] {
        let mut native = json!({"role": "assistant", "content": text("The prose quotes <synthetic> and API Error.")});
        if let Some(model) = model {
            native["model"] = json!(model);
        }
        let parsed = run(&[record("assistant", "real", native, json!({}))]);
        let real = message(&parsed, ":real");
        assert_eq!(real["role"], "assistant");
        assert!(real.get("prompt_kind").is_none());
        assert_eq!(real.get("model").and_then(Value::as_str), model);
        assert_eq!(
            real["blocks"][0]["text"],
            "The prose quotes <synthetic> and API Error."
        );
    }
}

#[test]
fn synthetic_notices_do_not_join_real_response_fragments_even_without_a_model() {
    for model in [None, Some("actual-model")] {
        let mut real = json!({"id": "shared-response", "role": "assistant", "content": text("Real first fragment.")});
        if let Some(model) = model {
            real["model"] = json!(model);
        }
        let mut later = real.clone();
        later["content"] = text("Real final fragment.");
        let records = [
            record(
                "assistant",
                "first",
                real,
                json!({"requestId": "shared-request"}),
            ),
            record(
                "assistant",
                "notice",
                json!({"id": "shared-response", "role": "assistant", "model": "<synthetic>", "content": text("Harness API notice.")}),
                json!({"requestId": "shared-request", "isApiErrorMessage": true, "error": "rate_limit_error", "apiErrorStatus": 429}),
            ),
            record(
                "assistant",
                "last",
                later,
                json!({"requestId": "shared-request"}),
            ),
        ];
        let parsed = run(&records);
        assert_eq!(messages(&parsed).len(), 2);
        let real = message(&parsed, ":first");
        assert_eq!(real.get("model").and_then(Value::as_str), model);
        assert!(real.get("prompt_kind").is_none());
        assert!(real["production"].get("api_error").is_none());
        assert_eq!(
            real["blocks"],
            json!([
            {"type": "text", "text": "Real first fragment."},
            {"type": "text", "text": "Real final fragment."}])
        );
        let notice = message(&parsed, ":notice");
        assert_eq!(notice["role"], "assistant");
        assert_eq!(notice["prompt_kind"], "harness");
        assert!(notice.get("model").is_none());
        assert_eq!(
            notice["production"]["api_error"],
            json!({"kind": "rate_limit_error", "status": 429})
        );
        let row = parsed["output"]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["message"]["id"] == "claude-code:notice")
            .unwrap();
        let start = jsonl(&records[..1]).len();
        assert_eq!(row["offset"], start);
        assert_eq!(
            row["end"],
            start + serde_json::to_vec(&records[1]).unwrap().len()
        );
    }
}
