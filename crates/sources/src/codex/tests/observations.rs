use super::*;

fn counters(input: u64, output: u64) -> Json {
    json!({"input_tokens": input, "cached_input_tokens": 0,
        "output_tokens": output, "reasoning_output_tokens": 0, "total_tokens": input+output})
}
fn count(o: u64, total: Json, last: Json) -> Json {
    env(
        o,
        "event_msg",
        json!({"type":"token_count", "info":{
        "total_token_usage":total, "last_token_usage":last}, "rate_limits":null}),
    )
}
fn item(o: u64, id: &str, ty: &str) -> Json {
    let p = match ty {
        "reasoning" => {
            json!({"id":id,"type":ty,"summary":[{"type":"summary_text","text":"synthetic thought"}]})
        }
        "function_call" => {
            json!({"id":id,"type":ty,"call_id":id,"name":"synthetic_tool","arguments":"{}"})
        }
        _ => {
            json!({"id":id,"type":"function_call_output","call_id":id,"output":"synthetic output"})
        }
    };
    env(o, "response_item", p)
}

#[test]
fn identical_rollouts_cannot_distinguish_one_response_from_a_retry() {
    // Producer OutputItemDone persists rA before a stream can fail. A retry
    // can persist rB+cB; the durable success trace is identical to one response.
    let trace = vec![
        meta(json!({})),
        item(2, "rA", "reasoning"),
        item(3, "rB", "reasoning"),
        item(4, "cB", "function_call"),
        item(5, "cB-result", "result"),
        count(6, counters(10, 2), counters(10, 2)),
    ];
    let step = run(&rollout(), &mut Carry::default(), &jsonl(&trace), 0).unwrap();
    let readings = readings(&step);
    assert_eq!(readings.len(), 1);
    assert_eq!(readings[0].completion_observed, Some(true));
    assert_eq!(readings[0].response, None);
    let messages = final_messages(&[step]);
    for id in ["rA", "rB", "cB"] {
        let m = messages
            .iter()
            .find(|m| m.id.native().as_str() == id)
            .unwrap();
        assert_eq!(m.production.response_boundary_unknown, Some(true));
        assert!(m.usage.is_empty());
    }
}

#[test]
fn tools_or_absent_items_do_not_drop_usage_or_poison_the_next_delta() {
    let trace = vec![
        meta(json!({})),
        count(2, counters(10, 2), counters(10, 2)),
        item(3, "call1", "function_call"),
        item(4, "call2", "function_call"),
        item(5, "call1-out", "result"),
        item(6, "call2-out", "result"),
        count(7, counters(30, 5), counters(20, 3)),
    ];
    let step = run(&rollout(), &mut Carry::default(), &jsonl(&trace), 0).unwrap();
    let r = readings(&step);
    assert_eq!(r.len(), 2);
    assert!(
        r.iter()
            .all(|r| r.completion_observed == Some(true) && r.response.is_none())
    );
    assert_eq!(r.iter().filter_map(|r| r.usage.input).sum::<u64>(), 30);
}

#[test]
fn repeated_rate_only_recomputed_and_full_context_reports_are_not_completions() {
    let mut trace = vec![
        meta(json!({})),
        count(2, counters(10, 2), counters(10, 2)),
        count(3, counters(10, 2), counters(10, 2)),
        env(
            4,
            "event_msg",
            json!({"type":"token_count","info":null,"rate_limits":{
            "primary":{"window_minutes":300,"used_percent":12.0,"resets_in_seconds":60}}}),
        ),
    ];
    let mut context = counters(0, 0);
    context["total_tokens"] = json!(500);
    trace.push(count(5, counters(10, 2), context.clone()));
    trace.push(count(6, counters(30, 5), counters(20, 3)));
    trace.push(count(7, context.clone(), context));
    trace.push(count(8, counters(0, 0), counters(0, 0)));
    let step = run(&rollout(), &mut Carry::default(), &jsonl(&trace), 0).unwrap();
    let r = readings(&step);
    assert_eq!(
        r.iter()
            .filter(|r| r.completion_observed == Some(true))
            .count(),
        2
    );
    assert_eq!(r.iter().filter_map(|r| r.usage.input).sum::<u64>(), 30);
    assert_eq!(r.last().unwrap().usage.input, Some(0));
    assert_eq!(r.last().unwrap().completion_observed, None);
    let q: polylogue_model::QuotaObservation = serde_json::from_value(
        step.output
            .events
            .iter()
            .find(|e| e.kind == "quota_observation")
            .unwrap()
            .data
            .clone(),
    )
    .unwrap();
    assert_eq!(q.reading.windows[0].resets_at, None);
    assert_eq!(q.observed_at, None);
}

#[test]
fn malformed_and_partial_reports_keep_numeric_suppliedness_without_saturation() {
    let trace = vec![
        meta(json!({})),
        count(2, counters(10, 2), counters(10, 2)),
        count(
            3,
            json!({"input_tokens":true}),
            json!({"input_tokens":7,"cached_input_tokens":9,"output_tokens":"4"}),
        ),
        count(4, counters(30, 5), counters(20, 3)),
    ];
    let step = run(&rollout(), &mut Carry::default(), &jsonl(&trace), 0).unwrap();
    let r = readings(&step);
    assert_eq!(r.len(), 3);
    assert_eq!(r[1].input_including_cache, Some(7));
    assert_eq!(r[1].usage.input, None);
    assert_eq!(r[1].usage.cached_input, None);
    assert_eq!(r[1].usage.output, None);
    assert_eq!(r[2].completion_observed, None);
}

#[test]
fn native_child_tools_are_harness_observations_without_a_model() {
    let trace = vec![
        meta(json!({})),
        env(2, "turn_context", json!({"model":"gpt-6.1-sol"})),
        env(
            3,
            "event_msg",
            json!({"type":"item_completed","item":{"type":"CommandExecution",
            "id":"native-child","command":["echo","synthetic"],"status":"completed",
            "exit_code":0,"aggregated_output":"synthetic output"}}),
        ),
    ];
    let step = run(&rollout(), &mut Carry::default(), &jsonl(&trace), 0).unwrap();
    let m = &step.output.messages[0].message;
    assert_eq!(m.id.native().as_str(), "native-child");
    assert_eq!(m.prompt_kind, Some(PromptKind::Harness));
    assert_eq!(m.model, None);
    assert_eq!(m.role, Role::Assistant);
}

#[test]
fn observation_state_survives_every_eof_restart_without_holding_item_bodies() {
    let trace = jsonl(&[
        meta(json!({})),
        item(2, "r1", "reasoning"),
        item(3, "c1", "function_call"),
        item(4, "out1", "result"),
        count(5, counters(10, 2), counters(10, 2)),
        count(6, counters(20, 4), counters(10, 2)),
    ]);
    let whole = run(&rollout(), &mut Carry::default(), &trace, 0).unwrap();
    let expected = readings(&whole);
    for cut in 0..=trace.len() {
        let mut carry = Carry::default();
        let first = run(&rollout(), &mut carry, &trace[..cut], 0).unwrap();
        let used = first.consumed;
        let mut carry: Carry =
            serde_json::from_value(serde_json::to_value(carry).unwrap()).unwrap();
        let second = run(&rollout(), &mut carry, &trace[used as usize..], used as u64).unwrap();
        let mut got = readings(&first);
        got.extend(readings(&second));
        assert_eq!(got, expected, "split {cut}");
    }
}

#[test]
fn copied_token_records_keep_their_parent_scope_and_original_observation_identity() {
    let token = count(3, counters(10, 2), counters(10, 2));
    let parent = vec![
        env(1, "session_meta", json!({"id":PARENT})),
        item(2, "parent-item", "reasoning"),
        token.clone(),
    ];
    let child = vec![
        meta(json!({"parent_thread_id":PARENT,"subagent_history_start_ordinal":4})),
        item(2, "parent-item", "reasoning"),
        token,
    ];
    let parent = run(
        &Rollout {
            thread: Some(PARENT.into()),
        },
        &mut Carry::default(),
        &jsonl(&parent),
        0,
    )
    .unwrap();
    let child = run(&rollout(), &mut Carry::default(), &jsonl(&child), 0).unwrap();
    assert_eq!(
        readings(&parent)[0].observation,
        readings(&child)[0].observation
    );
    assert_eq!(readings(&child)[0].response, None);
    assert!(
        child
            .output
            .events
            .iter()
            .filter(|e| e.kind != "response_boundaries_unknown")
            .all(|e| e.session.native().as_str() == PARENT)
    );
}

#[test]
fn header_only_and_rate_only_rollouts_keep_the_unknown_boundary_scope_fact() {
    for rows in [
        vec![meta(json!({}))],
        vec![
            meta(json!({})),
            env(
                2,
                "event_msg",
                json!({"type":"token_count","info":null,
        "rate_limits":{"primary":{"window_minutes":300,"used_percent":1}}}),
            ),
        ],
    ] {
        let step = run(&rollout(), &mut Carry::default(), &jsonl(&rows), 0).unwrap();
        assert!(step.output.messages.is_empty());
        let fact = step
            .output
            .events
            .iter()
            .find(|e| e.kind == "response_boundaries_unknown")
            .unwrap();
        assert_eq!(fact.data, json!({}));
        assert_eq!(fact.at, None);
        assert_eq!(fact.after, None);
        assert_eq!(fact.range, None);
    }
}

#[test]
fn regressive_snapshots_cannot_recharge_an_old_delta_or_prove_a_new_counter_epoch() {
    let trace = vec![
        meta(json!({})),
        count(2, counters(10, 2), counters(10, 2)),
        count(3, counters(30, 5), counters(20, 3)),
        count(4, counters(10, 2), counters(10, 2)),
        count(5, counters(30, 5), counters(20, 3)),
        count(6, counters(50, 8), counters(20, 3)),
    ];
    let step = run(&rollout(), &mut Carry::default(), &jsonl(&trace), 0).unwrap();
    let r = readings(&step);
    assert_eq!(
        r.iter()
            .filter(|r| r.completion_observed == Some(true))
            .count(),
        2
    );
    assert_eq!(
        r.iter()
            .filter(|r| r.completion_observed == Some(true))
            .filter_map(|r| r.usage.input)
            .sum::<u64>(),
        30
    );
    assert_eq!(r.last().unwrap().usage.input, Some(20));
    assert_eq!(r.last().unwrap().completion_observed, None);
}
