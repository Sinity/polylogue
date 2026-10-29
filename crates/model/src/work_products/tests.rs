use super::*;
use serde_json::json;

fn message(id: &str, role: &str, kind: &str, blocks: serde_json::Value) -> SelectedMessage {
    SelectedMessage {
        logical_session: SessionId::parse("codex:morrow").unwrap(),
        projection: None,
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
fn sibling_agent_answers_do_not_complete_the_parent_physical_turn() {
    let parent = message("parent-prompt", "user", "prose", json!([]));
    let mut child = message("child-task", "user", "model", json!([]));
    child.message.session = SessionId::parse("codex:child").unwrap();
    let mut answer = message("child-answer", "assistant", "model", json!([]));
    answer.message.session = child.message.session.clone();
    let (_, turns) = derive([parent, child, answer]);
    assert_eq!(turns.len(), 2);
    assert!(turns[0].facts.abandoned);
    assert!(turns[0].facts.answer.is_none());
    assert_eq!(turns[0].facts.physical_session.to_string(), "codex:morrow");
    assert_eq!(turns[1].facts.physical_session.to_string(), "codex:child");
    assert_eq!(
        turns[1].facts.answer.as_ref().unwrap().to_string(),
        "codex:child-answer"
    );
    assert_eq!(turns[0].facts.session, turns[1].facts.session);
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
    assert_eq!(call.facts.outcome, Some(ToolOutcome::Ok));
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
    assert_eq!(calls[0].facts.outcome, Some(ToolOutcome::NoResult));
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
        [SelectedBackground {
            logical_session: SessionId::parse("codex:morrow").unwrap(),
            update,
            event: "event-background".into(),
        }],
    );
    assert_eq!(calls[0].facts.outcome, Some(ToolOutcome::Ok));
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

#[test]
fn native_and_local_display_collisions_keep_distinct_structural_entries() {
    let local = message(
        "unkeyed",
        "assistant",
        "model",
        json!([
            {"type":"tool_call","tool":"shell","action":"shell","input":{}}
        ]),
    );
    let native = message(
        "unkeyed",
        "assistant",
        "model",
        json!([
            {"type":"tool_call","call_id":"codex:unkeyed/0","tool":"shell","action":"shell","input":{}}
        ]),
    );
    let (local, _) = derive([local]);
    let (native, _) = derive([native]);
    assert_eq!(local[0].facts.call, native[0].facts.call);
    assert_ne!(local[0].facts.entry_id, native[0].facts.entry_id);
    assert_eq!(local[0].facts.native_call_id, None);
    assert_eq!(local[0].facts.outcome, None);
    assert_eq!(
        local[0].facts.outcome_reason.as_deref(),
        Some("native_call_id_unavailable")
    );
    assert_eq!(
        native[0].facts.native_call_id.as_deref(),
        Some("codex:unkeyed/0")
    );
}

#[test]
fn original_slot_identity_survives_sparse_projection_and_source_copy_reselection() {
    for native in [None, Some("native-call")] {
        let full = message(
            "projected",
            "assistant",
            "model",
            json!([
                {"type":"text","text":"before"},
                {"type":"tool_call","call_id":native,"tool":"shell","action":"shell","input":{}},
                {"type":"text","text":"after"}
            ]),
        );
        let mut sparse = full.clone();
        sparse.message.blocks = vec![full.message.blocks[1].clone()];
        sparse.event = "event-selected-sparse-copy".into();
        sparse.projection = Some(crate::bundle::MessageProjection {
            source_version: crate::BlobHash::of(b"complete synthetic message"),
            block_count: 3,
            block_ordinals: vec![1],
        });
        let (complete, _) = derive([full]);
        let (projected, _) = derive([sparse]);
        assert_eq!(complete[0].facts.entry_id, projected[0].facts.entry_id);
        assert_eq!(complete[0].facts.call, projected[0].facts.call);
        assert_eq!(projected[0].facts.original_block, Some(1));
        assert_eq!(projected[0].parents, ["event-selected-sparse-copy"]);
    }
}

#[test]
fn redefined_native_ids_retain_all_definitions_and_exact_ambiguity_evidence() {
    let first = message("first", "assistant", "model", call());
    let second = message("second", "assistant", "model", call());
    let result = message("result", "user", "harness", result());
    for values in [
        [first.clone(), result.clone(), second.clone()],
        [result, second, first],
    ] {
        let (calls, turns) = derive(values);
        assert_eq!(calls.len(), 2);
        assert_ne!(calls[0].facts.entry_id, calls[1].facts.entry_id);
        assert_eq!(calls[0].facts.call, calls[1].facts.call);
        for call in calls {
            assert_eq!(call.facts.outcome, None);
            assert_eq!(call.facts.result, None);
            assert_eq!(call.facts.reported_duration_ms, None);
            assert_eq!(
                call.facts.outcome_reason.as_deref(),
                Some("ambiguous_native_call_id")
            );
            assert_eq!(
                call.parents,
                ["event-first", "event-result", "event-second"]
            );
        }
        assert_eq!(
            turns.iter().map(|turn| turn.facts.tool_calls).sum::<u64>(),
            2
        );
        assert!(turns.iter().any(|turn| {
            turn.facts
                .unavailable
                .contains(&"ambiguous_native_call_id".into())
        }));
    }
    let (calls, turns) = derive([message(
        "same-message",
        "assistant",
        "model",
        json!([
            {"type":"tool_call","call_id":"call-1","tool":"shell","action":"shell","input":{}},
            {"type":"tool_call","call_id":"call-1","tool":"shell","action":"shell","input":{}}
        ]),
    )]);
    assert_eq!(calls.len(), 2);
    assert_ne!(calls[0].facts.entry_id, calls[1].facts.entry_id);
    assert_eq!(
        (calls[0].facts.original_block, calls[1].facts.original_block),
        (Some(0), Some(1))
    );
    assert_eq!(turns[0].facts.tool_calls, 2);
}

#[test]
fn multiple_native_results_do_not_overwrite_each_other_or_choose_by_arrival() {
    let invocation = message("call", "assistant", "model", call());
    let first = message("result-a", "user", "harness", result());
    let second = message(
        "result-b",
        "user",
        "harness",
        json!([
            {"type":"tool_result","call_id":"call-1","outcome":{"kind":"error"},"output":"different evidence"}
        ]),
    );
    for values in [
        [first.clone(), invocation.clone(), second.clone()],
        [second, invocation, first],
    ] {
        let (calls, _) = derive(values);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].facts.result, None);
        assert_eq!(calls[0].facts.outcome, None);
        assert_eq!(
            calls[0].facts.outcome_reason.as_deref(),
            Some("ambiguous_result_evidence")
        );
        assert_eq!(
            calls[0].parents,
            ["event-call", "event-result-a", "event-result-b"]
        );
    }
}

fn scope(mut selected: SelectedMessage, logical: &str, physical: &str) -> SelectedMessage {
    selected.logical_session = SessionId::parse(logical).unwrap();
    selected.message.session = SessionId::parse(physical).unwrap();
    selected
}

#[test]
fn physical_agents_reusing_native_ids_pair_only_with_their_own_results() {
    let root = "codex:logical";
    let (calls, _) = derive([
        scope(
            message("call-a", "assistant", "model", call()),
            root,
            "codex:agent-a",
        ),
        scope(
            message("call-b", "assistant", "model", call()),
            root,
            "codex:agent-b",
        ),
        scope(
            message("result-b", "user", "harness", result()),
            root,
            "codex:agent-b",
        ),
        scope(
            message("result-a", "user", "harness", result()),
            root,
            "codex:agent-a",
        ),
    ]);
    assert_eq!(calls.len(), 2);
    for (call, physical, result) in [
        (&calls[0], "codex:agent-a", "codex:result-a"),
        (&calls[1], "codex:agent-b", "codex:result-b"),
    ] {
        assert_eq!(call.facts.session.to_string(), root);
        assert_eq!(call.facts.physical_session.to_string(), physical);
        assert_eq!(call.facts.result.as_ref().unwrap().to_string(), result);
        assert_eq!(call.facts.outcome, Some(ToolOutcome::Ok));
        assert!(call.facts.call.starts_with(&format!("tool:{physical}#")));
    }
    assert_ne!(calls[0].facts.entry_id, calls[1].facts.entry_id);
}

#[test]
fn background_evidence_uses_physical_scope_and_redefinitions_remain_ambiguous() {
    let update = crate::BackgroundUpdate {
        session: SessionId::parse("codex:agent-a").unwrap(),
        task_id: "task".into(),
        call_id: Some("call-1".into()),
        outcome: ToolOutcome::Ok,
        result_at: None,
        result_at_evidence: crate::TimeEvidence::UNKNOWN,
        reported_duration_ms: Some(19),
        output_file: None,
    };
    let (calls, turns) = derive_with_background(
        [
            scope(
                message("a", "assistant", "model", call()),
                "codex:root",
                "codex:agent-a",
            ),
            scope(
                message("b", "assistant", "model", call()),
                "codex:root",
                "codex:agent-b",
            ),
        ],
        [SelectedBackground {
            logical_session: SessionId::parse("codex:root").unwrap(),
            update: update.clone(),
            event: "background-a".into(),
        }],
    );
    assert_eq!(calls[0].facts.outcome, Some(ToolOutcome::Ok));
    assert_eq!(calls[1].facts.outcome, Some(ToolOutcome::NoResult));
    assert!(!calls[1].parents.contains(&"background-a".into()));
    assert!(turns[0].parents.contains(&"background-a".into()));
    let (calls, _) = derive_with_background(
        [
            scope(
                message("a", "assistant", "model", call()),
                "codex:root",
                "codex:agent-a",
            ),
            scope(
                message("b", "assistant", "model", call()),
                "codex:root",
                "codex:agent-a",
            ),
        ],
        [SelectedBackground {
            logical_session: SessionId::parse("codex:root").unwrap(),
            update,
            event: "background-a".into(),
        }],
    );
    for call in calls {
        assert_eq!(call.facts.outcome, None);
        assert_eq!(call.parents, ["background-a", "event-a", "event-b"]);
    }
}

#[test]
fn invalid_projection_keeps_unknown_origin_entries_without_fabricated_slots() {
    let mut selected = message(
        "sparse",
        "assistant",
        "model",
        json!([
            {"type":"tool_call","call_id":"one","tool":"shell","action":"shell","input":{}},
            {"type":"tool_call","call_id":"two","tool":"shell","action":"shell","input":{}}
        ]),
    );
    selected.projection = Some(crate::bundle::MessageProjection {
        source_version: crate::BlobHash::of(b"synthetic partial"),
        block_count: 20,
        block_ordinals: vec![9],
    });
    let mut other_copy = selected.clone();
    other_copy.event = "event-other-copy".into();
    let (calls, turns) = derive([selected]);
    let (other, _) = derive([other_copy]);
    assert_eq!(calls.len(), 2);
    assert_ne!(calls[0].facts.entry_id, calls[1].facts.entry_id);
    assert_ne!(calls[0].facts.entry_id, other[0].facts.entry_id);
    for call in calls {
        assert_eq!(call.facts.original_block, None);
        assert_eq!(call.facts.unavailable, ["original_block_unavailable"]);
        assert_eq!(call.parents, ["event-sparse"]);
    }
    assert!(
        turns[0]
            .facts
            .unavailable
            .contains(&"original_block_unavailable".into())
    );
}

#[test]
fn structural_scope_does_not_alias_delimiter_display_refs_or_logical_roots() {
    let first = scope(
        message(
            "same",
            "assistant",
            "model",
            json!([
                {"type":"tool_call","call_id":"c","tool":"shell","action":"shell","input":{}}
            ]),
        ),
        "codex:logical",
        "codex:a#b",
    );
    let second = scope(
        message(
            "same",
            "assistant",
            "model",
            json!([
                {"type":"tool_call","call_id":"b#c","tool":"shell","action":"shell","input":{}}
            ]),
        ),
        "codex:logical",
        "codex:a",
    );
    let mut other_root = first.clone();
    other_root.logical_session = SessionId::parse("codex:other-root").unwrap();
    let (a, _) = derive([first]);
    let (b, _) = derive([second]);
    let (c, _) = derive([other_root]);
    assert_eq!(a[0].facts.call, b[0].facts.call);
    assert_ne!(a[0].facts.entry_id, b[0].facts.entry_id);
    assert_ne!(a[0].facts.entry_id, c[0].facts.entry_id);
    assert_eq!(a[0].facts.physical_session, c[0].facts.physical_session);
}

#[test]
fn snapshot_validation_requires_reducer_two_unique_entries_and_unavailable_reasons() {
    let (calls, _) = derive([
        message("first", "assistant", "model", call()),
        message("second", "assistant", "model", call()),
    ]);
    let mut snapshot = WorkSnapshot {
        reducer_version: VERSION,
        position: 4,
        session_id: SessionId::parse("codex:morrow").unwrap(),
        session: None,
        retired: false,
        unavailable: Vec::new(),
        messages: Vec::new(),
        calls: calls.into_iter().map(|call| call.facts).collect(),
        turns: Vec::new(),
        sources: Vec::new(),
        native: Vec::new(),
    };
    assert!(snapshot.validate().is_ok());
    let encoded = serde_json::to_value(&snapshot).unwrap();
    assert!(encoded["calls"][0].get("outcome").is_none());
    let original = snapshot.calls[1].entry_id.clone();
    snapshot.calls[1].entry_id = snapshot.calls[0].entry_id.clone();
    assert_eq!(
        snapshot.validate(),
        Err("snapshot call entry identity is missing or repeated")
    );
    snapshot.calls[1].entry_id = original;
    snapshot.calls[0].unavailable.clear();
    assert_eq!(
        snapshot.validate(),
        Err("unavailable call outcome has no reason and evidence coverage")
    );
    snapshot.calls[0]
        .unavailable
        .push("ambiguous_native_call_id".into());
    snapshot.reducer_version = 1;
    assert_eq!(snapshot.validate(), Err("unsupported work reducer version"));
}

#[test]
fn a_sibling_agent_response_does_not_advance_another_physical_calls_boundary() {
    let (calls, _) = derive([
        scope(
            message("call", "assistant", "model", call()),
            "codex:root",
            "codex:agent-a",
        ),
        scope(
            message("answer", "assistant", "model", json!([])),
            "codex:root",
            "codex:agent-b",
        ),
    ]);
    assert_eq!(calls[0].facts.outcome, Some(ToolOutcome::NoResult));
    assert_eq!(calls[0].facts.outcome_reason.as_deref(), Some("no_result"));
}

#[test]
fn consistent_native_observations_preserve_known_values_without_inventing_final_order() {
    let update = crate::BackgroundUpdate {
        session: SessionId::parse("codex:morrow").unwrap(),
        task_id: "task".into(),
        call_id: Some("call-1".into()),
        outcome: ToolOutcome::Ok,
        result_at: None,
        result_at_evidence: crate::TimeEvidence::UNKNOWN,
        reported_duration_ms: None,
        output_file: None,
    };
    let (calls, _) = derive_with_background(
        [
            message("call", "assistant", "model", call()),
            message("result", "user", "harness", result()),
        ],
        [SelectedBackground {
            logical_session: SessionId::parse("codex:morrow").unwrap(),
            update,
            event: "background".into(),
        }],
    );
    assert_eq!(calls[0].facts.outcome, Some(ToolOutcome::Ok));
    assert_eq!(
        calls[0].facts.result.as_ref().unwrap().to_string(),
        "codex:result"
    );
    assert_eq!(calls[0].facts.reported_duration_ms, Some(27));
    assert!(calls[0].facts.unavailable.is_empty());
    assert_eq!(
        calls[0].parents,
        ["background", "event-call", "event-result"]
    );
}
