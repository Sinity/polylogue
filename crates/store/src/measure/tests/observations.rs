use super::*;
use crate::fixture::writer::{WriterWorld, message, sid};
use polylogue_model::{
    Block, Event, MessageRow, Output, PromptKind, QuotaObservation, Role, Session, SessionKind,
    Timestamp, UsageObservation,
};

fn event(kind: &str, at: i64, data: impl serde::Serialize) -> Event {
    Event {
        session: sid("observed"),
        kind: kind.into(),
        at: Timestamp::from_unix_ms(at),
        after: None,
        range: None,
        data: serde_json::to_value(data).unwrap(),
    }
}
fn measured(id: &str, input: u64) -> UsageObservation {
    UsageObservation {
        observation: id.into(),
        usage: tokens(input, 0, Some(0), 2),
        model: Some("gpt-6.1-sol".into()),
        response: None,
        completion_observed: Some(true),
        observed_at: None,
        input_including_cache: Some(input),
        output_including_reasoning: Some(2),
    }
}

#[test]
fn independent_usage_is_measured_once_without_fabricating_response_or_missing_usage_counts() {
    let mut world = WriterWorld::new();
    let mut item = message(
        "item",
        "observed",
        Role::Assistant,
        T0,
        vec![Block::Text {
            text: "synthetic output item".into(),
        }],
    );
    item.model = Some("gpt-6.1-sol".into());
    item.production.response_boundary_unknown = Some(true);
    let mut child = message(
        "native-tool",
        "observed",
        Role::Assistant,
        T0 + 1,
        vec![Block::Text {
            text: "synthetic tool output".into(),
        }],
    );
    child.prompt_kind = Some(PromptKind::Harness);
    let mut unconfirmed = measured("uncertain", 90);
    unconfirmed.completion_observed = None;
    world.write_output(
        "input",
        "codex",
        "/synthetic/rollout.jsonl",
        Output {
            sessions: vec![Session::new(sid("observed"), SessionKind::Main)],
            messages: vec![
                MessageRow {
                    offset: 0,
                    end: 1,
                    message: item,
                },
                MessageRow {
                    offset: 1,
                    end: 2,
                    message: child,
                },
            ],
            events: vec![
                event("usage_observation", T0 + 2, measured("original", 10)),
                event("usage_observation", T0 + 3, measured("original", 10)),
                event("usage_observation", T0 + 4, unconfirmed),
            ],
            ..Output::default()
        },
        5,
    );
    world.writer.derive().unwrap();
    let pool = crate::ReadPool::open(&world.archive).unwrap();
    let reader = pool.reader().unwrap();
    let report = usage(&reader, &Scope::default()).unwrap();
    assert_eq!(report.messages, None);
    assert_eq!(report.known_response_messages, 0);
    assert_eq!(report.unknown_boundary_items, 1);
    assert_eq!(report.api_cost.unpriced_messages, 0);
    assert_eq!(report.tokens.input.unknown_messages, 0);
    assert_eq!(report.recorded_usage.observations, 2);
    assert_eq!(report.recorded_usage.measured_completion_observations, 1);
    assert_eq!(report.recorded_usage.unconfirmed_observations, 1);
    assert_eq!(report.recorded_usage.known_usage.input, Some(10));
    assert_eq!(report.recorded_usage.untimed_observations, 2);
    let sessions = super::super::sessions(&reader, &Scope::default()).unwrap();
    assert_eq!(sessions[0].physical.messages, None);
    assert_eq!(sessions[0].logical.messages, None);
    assert!(sessions[0].logical.recorded_usage.is_none());
    assert_eq!(
        world.rows(
            "SELECT written_by,model FROM message WHERE id='claude-code:observed-native-tool'"
        )[0]["written_by"],
        "harness"
    );
}

#[test]
fn delayed_rate_only_snapshots_preserve_state_without_provider_time_or_window_spend() {
    let mut world = WriterWorld::new();
    let hour = 3_600_000;
    let reading: QuotaReading =
        serde_json::from_value(reading("codex", 300, false, Some(12.0), T0 + hour, None)).unwrap();
    let q = QuotaObservation {
        observation: "quota-state".into(),
        reading,
        observed_at: None,
    };
    world.write_output(
        "input",
        "codex",
        "/synthetic/rollout.jsonl",
        Output {
            sessions: vec![Session::new(sid("observed"), SessionKind::Main)],
            events: vec![
                event("quota_observation", T0, q.clone()),
                event("quota_observation", T0 + hour / 2, q),
                event("usage_observation", T0 + 2, measured("usage", 10)),
            ],
            ..Output::default()
        },
        3,
    );
    let pool = crate::ReadPool::open(&world.archive).unwrap();
    let reader = pool.reader().unwrap();
    let report = quota::quota(&reader, T0 + hour / 2, &[]).unwrap();
    let tool = &report.tools[0];
    let w = &tool.windows[0];
    assert_eq!(w.used_percent, Some(12.0));
    assert_eq!(w.resets_at, T0 + hour);
    assert_eq!(w.observed_at, None);
    assert_eq!(w.burn_percent_per_hour, None);
    assert_eq!(w.exhausted_at, None);
    assert_eq!(w.tokens_per_hour, None);
    assert_eq!(w.api_usd_per_hour, None);
    assert_eq!(w.usage, None);
    assert_eq!(tool.recorded_usage.known_usage.input, Some(10));
}

#[test]
fn unresolved_item_and_harness_followups_do_not_prove_a_call_was_passed_without_a_result() {
    let mut world = WriterWorld::new();
    let mut call =
        crate::fixture::writer::bash("pending", 1, T0, "synthetic", true, "unused").remove(0);
    call.production.response_boundary_unknown = Some(true);
    let mut later = message(
        "later",
        "pending",
        Role::Assistant,
        T0 + 1,
        vec![Block::Text {
            text: "synthetic child event".into(),
        }],
    );
    later.prompt_kind = Some(PromptKind::Harness);
    world.write("pending", vec![call, later]);
    assert_eq!(
        world.rows("SELECT outcome_reason FROM tool_call")[0]["outcome_reason"],
        "no_result"
    );
}

#[test]
fn boundary_scope_is_not_discarded_by_a_later_time_window_with_no_recorded_items() {
    let mut world = WriterWorld::new();
    world.write_output(
        "unknown",
        "codex",
        "/synthetic/rollout.jsonl",
        Output {
            sessions: vec![Session::new(sid("observed"), SessionKind::Main)],
            events: vec![event("response_boundaries_unknown", T0, json!({}))],
            ..Output::default()
        },
        1,
    );
    let known = message(
        "known",
        "known",
        Role::Assistant,
        T0,
        vec![Block::Text {
            text: "known response".into(),
        }],
    );
    world.write("known", vec![known]);
    let pool = crate::ReadPool::open(&world.archive).unwrap();
    let reader = pool.reader().unwrap();
    let pk = |id: &str| {
        reader
            .conn()
            .unwrap()
            .query_row(
                "SELECT pk FROM session WHERE id=?1",
                [sid(id).to_string()],
                |r| r.get::<_, i64>(0),
            )
            .unwrap()
    };
    let scope = |id: &str| Scope {
        sessions: Some(vec![pk(id)]),
        since: Some(T0 + 1),
        ..Scope::default()
    };
    let unknown = usage(&reader, &scope("observed")).unwrap();
    assert_eq!(unknown.messages, None);
    assert_eq!(unknown.recorded_model_items, 0);
    assert_eq!(unknown.unknown_boundary_items, 0);
    assert_eq!(unknown.recorded_usage.observations, 0);
    assert!(unknown.tokens.input.response_total_unknown);
    let known = usage(&reader, &scope("known")).unwrap();
    assert_eq!(known.messages, Some(0));
    assert_eq!(known.recorded_model_items, 0);
    assert!(!known.tokens.input.response_total_unknown);
}
