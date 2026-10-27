use super::*;
use crate::fixture::writer::{WriterArchive, bash, message, sid};
use crate::fixture::{Fixture, MessageFacts, SessionFacts};
use polylogue_model::{Block, Role, Session, SessionKind, Timestamp};

#[test]
fn tool_time_is_separate_from_clustered_time_and_latency_needs_pairs() {
    let mut f = WriterArchive::new();
    let mut calls = bash("long", 1, 0, "synthetic", true, "ok");
    calls[1].at = Timestamp::from_unix_ms(720_000);
    f.write("long", calls);
    f.writer.derive().unwrap();
    let pool = crate::ReadPool::open(&f.archive).unwrap();
    let r = pool.reader().unwrap();
    let rows = r.session_profiles(&Selection::default()).unwrap();
    let p = &rows[0].facts;
    assert_eq!(p.tool_active_duration_ms, 720_000);
    assert_eq!(p.engaged_duration_ms, 0);
    assert_eq!(p.tool_latency.median_ms, Some(720_000));
    assert_eq!(p.tool_latency.measured_pairs, 1);
    assert_eq!(p.rule_version, RULE_VERSION);
    assert_eq!(p.cluster_gap_ms, 300_000);
}

#[test]
fn zero_latency_round_trips_and_distribution_keeps_missing_sessions_apart() {
    let mut f = WriterArchive::new();
    for i in 0..13 {
        let name = format!("s{i}");
        let mut calls = bash(&name, 1, 0, "synthetic", true, "ok");
        if i < 10 {
            calls[1].at = Timestamp::from_unix_ms(if i == 9 { 100 } else { 0 });
        }
        if i == 10 {
            calls.pop();
        }
        if i == 11 {
            for message in &mut calls {
                message.at = None;
            }
        }
        if i == 12 {
            calls = vec![message(
                "a",
                &name,
                Role::Assistant,
                0,
                vec![Block::Text {
                    text: "done".into(),
                }],
            )];
        }
        f.write(&name, calls);
    }
    f.writer.derive().unwrap();
    let pool = crate::ReadPool::open(&f.archive).unwrap();
    let r = pool.reader().unwrap();
    let x =
        super::super::comparison::distribution(&r, &Selection::default(), "median_tool_call_ms")
            .unwrap();
    assert_eq!(x.total_sessions, 13);
    assert_eq!(x.measured_sessions, 10);
    assert_eq!(x.missing_sessions, 3);
    assert_eq!(x.median, Some(0.0));
    assert_eq!(x.p90, Some(0.0));
    assert_eq!(x.max, Some(100.0));
    let correlation = super::super::comparison::correlation(
        &r,
        &Selection::default(),
        "wall_duration_ms:median_tool_call_ms",
    )
    .unwrap();
    assert_eq!(correlation.sample_count, 10);
    assert_eq!(correlation.missing_pairs, 3);
    assert_eq!(correlation.pearson_r, Some(1.0));
    let profiles = r.session_profiles(&Selection::default()).unwrap();
    let zero = profiles
        .iter()
        .find(|p| p.session == "claude-code:s0")
        .unwrap();
    let restored: Facts =
        serde_json::from_value(serde_json::to_value(&zero.facts).unwrap()).unwrap();
    assert_eq!(restored.tool_latency.median_ms, Some(0));
}

#[test]
fn topic_comes_from_authored_prose_raw_title_and_usage_are_preserved() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let s = f
        .session(SessionFacts {
            id: "codex:topic",
            title: Some("raw title"),
            repo: Some("https://example.test/team/project.git"),
            ..Default::default()
        })
        .unwrap();
    f.message(
        s,
        MessageFacts {
            id: "protocol",
            role: "user",
            written_by: "harness",
            text: Some("generated wrapper"),
            ..Default::default()
        },
    )
    .unwrap();
    f.message(
        s,
        MessageFacts {
            id: "prompt",
            role: "user",
            written_by: "human",
            prompt_kind: Some("prose"),
            text: Some("Design the parser"),
            ..Default::default()
        },
    )
    .unwrap();
    let reply = f
        .message(
            s,
            MessageFacts {
                id: "reply",
                role: "assistant",
                written_by: "model",
                model: Some("gpt-6.1-sol"),
                text: Some("Finished successfully"),
                ..Default::default()
            },
        )
        .unwrap();
    f.usage(
        reply,
        &polylogue_model::Usage {
            input: Some(40_000),
            output: Some(7500),
            cached_input: Some(200_000),
            cache_write: Some(0),
            reasoning: Some(0),
            ..Default::default()
        },
        None,
    )
    .unwrap();
    f.derive().unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let rows = r.session_profiles(&Selection::default()).unwrap();
    let p = &rows[0];
    assert_eq!(p.title.as_deref(), Some("raw title"));
    assert_eq!(p.facts.topic.as_deref(), Some("project: Design the parser"));
    assert_eq!(p.facts.topic_source, "repo_plus_first_user_turn");
    assert_eq!(p.facts.workflow_shape, "chat");
    assert_eq!(p.facts.terminal_state, "unknown");
    assert_eq!(p.facts.timestamp_source, "absent");
    assert_eq!(p.facts.wall_duration_ms, None);
    assert_eq!(p.facts.agent_response_latency.median_ms, None);
    let expected = super::super::sessions(&r, &Selection::default())
        .unwrap()
        .remove(0);
    assert_eq!(p.usage.as_ref().unwrap().own, expected.own);
    assert!(p.usage.as_ref().unwrap().own.api_cost.known_usd.unwrap() > 0.0);
    let value = serde_json::to_value(p).unwrap();
    assert_eq!(value["usage"]["own"]["tokens"]["input"]["tokens"], 40_000);
}

#[test]
fn session_only_clock_never_invents_latency_or_clustered_time() {
    let mut f = WriterArchive::new();
    let mut session = Session::new(sid("clock"), SessionKind::Main);
    session.started = Timestamp::from_unix_ms(5_000);
    let mut m = message(
        "u",
        "clock",
        Role::User,
        0,
        vec![Block::Text {
            text: "Start".into(),
        }],
    );
    m.at = None;
    f.write_session(session, vec![m]);
    f.writer.derive().unwrap();
    let pool = crate::ReadPool::open(&f.archive).unwrap();
    let r = pool.reader().unwrap();
    let p = r
        .session_profiles(&Selection::default())
        .unwrap()
        .remove(0)
        .facts;
    assert_eq!(p.timestamp_source, "session_timestamp_fallback");
    assert_eq!(p.first_at, Some(5_000));
    assert_eq!(p.timestamped_messages, 0);
    assert_eq!(p.engaged_duration_ms, 0);
    assert_eq!(p.tool_latency.median_ms, None);
    assert_eq!(p.terminal_state, "question_left");
}

#[test]
fn sparse_user_messages_are_neither_clustered_nor_tool_active() {
    let mut f = WriterArchive::new();
    let rows = (0..10)
        .map(|i| {
            message(
                &format!("u{i}"),
                "sparse",
                Role::User,
                i * 360_000,
                vec![Block::Text {
                    text: "Prompt".into(),
                }],
            )
        })
        .collect();
    f.write("sparse", rows);
    f.writer.derive().unwrap();
    let pool = crate::ReadPool::open(&f.archive).unwrap();
    let r = pool.reader().unwrap();
    let p = r
        .session_profiles(&Selection::default())
        .unwrap()
        .remove(0)
        .facts;
    assert_eq!(p.engaged_duration_ms, 0);
    assert_eq!(p.tool_active_duration_ms, 0);
    assert_eq!(p.wall_duration_ms, Some(9 * 360_000));
}

#[test]
fn structural_errors_pending_calls_and_recovery_decide_final_state() {
    let mut f = WriterArchive::new();
    let mut error = bash("error", 1, 0, "synthetic", false, "failed");
    error.push(message(
        "end",
        "error",
        Role::Assistant,
        2,
        vec![Block::Text {
            text: "It worked".into(),
        }],
    ));
    f.write("error", error);
    let mut recovered = bash("recovered", 1, 0, "synthetic", false, "failed");
    recovered.extend(bash("recovered", 2, 2, "synthetic", true, "ok"));
    recovered.push(message(
        "end",
        "recovered",
        Role::Assistant,
        4,
        vec![Block::Text {
            text: "done".into(),
        }],
    ));
    f.write("recovered", recovered);
    let mut pending = bash("pending", 1, 0, "synthetic", true, "ok");
    pending.pop();
    f.write("pending", pending);
    f.writer.derive().unwrap();
    let pool = crate::ReadPool::open(&f.archive).unwrap();
    let r = pool.reader().unwrap();
    let rows = r.session_profiles(&Selection::default()).unwrap();
    let state = |id: &str| {
        rows.iter()
            .find(|p| p.session == id)
            .unwrap()
            .facts
            .terminal_state
            .as_str()
    };
    assert_eq!(state("claude-code:error"), "error_left");
    assert_eq!(state("claude-code:recovered"), "unknown");
    assert_eq!(state("claude-code:pending"), "tool_left");
}

#[test]
fn agent_and_user_response_gaps_preserve_zero() {
    let mut f = WriterArchive::new();
    f.write(
        "zero",
        vec![
            message(
                "u",
                "zero",
                Role::User,
                0,
                vec![Block::Text {
                    text: "prompt".into(),
                }],
            ),
            message(
                "a",
                "zero",
                Role::Assistant,
                0,
                vec![Block::Text {
                    text: "reply".into(),
                }],
            ),
            message(
                "u2",
                "zero",
                Role::User,
                0,
                vec![Block::Text {
                    text: "prompt".into(),
                }],
            ),
        ],
    );
    f.writer.derive().unwrap();
    let pool = crate::ReadPool::open(&f.archive).unwrap();
    let r = pool.reader().unwrap();
    let p = r
        .session_profiles(&Selection::default())
        .unwrap()
        .remove(0)
        .facts;
    assert_eq!(p.agent_response_latency.median_ms, Some(0));
    assert_eq!(p.user_response_latency.median_ms, Some(0));
}

#[test]
fn event_clock_is_labeled_and_does_not_fabricate_turn_latency() {
    use polylogue_model::{Event, MessageRow, Output};
    let mut f = WriterArchive::new();
    let mut prompt = message(
        "u",
        "events",
        Role::User,
        0,
        vec![Block::Text {
            text: "Start this work".into(),
        }],
    );
    prompt.at = None;
    f.write_output(
        "events",
        "claude-code",
        "/synthetic/events.jsonl",
        Output {
            sessions: vec![Session::new(sid("events"), SessionKind::Main)],
            messages: vec![MessageRow {
                offset: 0,
                end: 1,
                message: prompt,
            }],
            events: [(0, "compaction"), (100_000, "synthetic")]
                .into_iter()
                .map(|(at, kind)| Event {
                    session: Some(sid("events")),
                    kind: kind.into(),
                    at: Timestamp::from_unix_ms(at),
                    after: None,
                    range: None,
                    data: serde_json::json!({}),
                })
                .collect(),
            ..Default::default()
        },
        3,
    );
    f.writer.derive().unwrap();
    let pool = crate::ReadPool::open(&f.archive).unwrap();
    let r = pool.reader().unwrap();
    let p = r
        .session_profiles(&Selection::default())
        .unwrap()
        .remove(0)
        .facts;
    assert_eq!(p.timestamp_source, "session_events_fallback");
    assert_eq!(p.wall_duration_ms, Some(100_000));
    assert_eq!(p.engaged_duration_ms, 100_000);
    assert_eq!(p.timestamped_messages, 0);
    assert_eq!(p.compaction_count, 1);
    assert_eq!(p.agent_response_latency.median_ms, None);
}

#[test]
fn workflow_rules_preserve_precedence_and_thresholds() {
    use crate::fixture::CallFacts;
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for (name, messages, actions) in [
        ("chat", 2, vec![]),
        ("subagent_dispatch", 1, vec!["subagent"]),
        ("batch_review", 2, vec!["file_read"; 3]),
        ("agentic_loop", 10, vec!["shell"; 3]),
        ("exploratory", 2, vec!["file_read"]),
        ("unknown", 0, vec![]),
    ] {
        let id = format!("codex:{name}");
        let pk = f
            .session(SessionFacts {
                id: &id,
                ..Default::default()
            })
            .unwrap();
        let mut last = None;
        for i in 0..messages {
            last = Some(
                f.message(
                    pk,
                    MessageFacts {
                        id: &format!("{name}-{i}"),
                        role: if i == 0 { "user" } else { "assistant" },
                        written_by: if i == 0 { "human" } else { "model" },
                        text: Some("Substantive synthetic text"),
                        ..Default::default()
                    },
                )
                .unwrap(),
            );
        }
        for action in actions {
            f.tool_call(
                pk,
                last.unwrap(),
                CallFacts {
                    tool: "synthetic",
                    action,
                    outcome: "unknown",
                    ..Default::default()
                },
            )
            .unwrap();
        }
    }
    f.derive().unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    for profile in r.session_profiles(&Selection::default()).unwrap() {
        assert_eq!(
            profile.facts.workflow_shape,
            profile.session.strip_prefix("codex:").unwrap()
        );
    }
}

#[test]
fn stop_reasons_are_observed_states_and_topic_prefers_substantive_prompt() {
    let mut f = WriterArchive::new();
    for (name, reason) in [("refused", "refusal"), ("truncated", "max_tokens")] {
        let first = message(
            "short",
            name,
            Role::User,
            0,
            vec![Block::Text { text: "yes".into() }],
        );
        let prompt = message(
            "prompt",
            name,
            Role::User,
            1,
            vec![Block::Text {
                text: "Design the parser".into(),
            }],
        );
        let mut end = message(
            "end",
            name,
            Role::Assistant,
            2,
            vec![Block::Text {
                text: "An observed reply".into(),
            }],
        );
        end.stop_reason = Some(reason.into());
        f.write(name, vec![first, prompt, end]);
    }
    f.writer.derive().unwrap();
    let pool = crate::ReadPool::open(&f.archive).unwrap();
    let r = pool.reader().unwrap();
    for profile in r.session_profiles(&Selection::default()).unwrap() {
        assert_eq!(
            profile.facts.terminal_state,
            profile.session.strip_prefix("claude-code:").unwrap()
        );
        assert_eq!(profile.facts.topic.as_deref(), Some("Design the parser"));
        assert_eq!(profile.facts.substantive_count, 2);
    }
}

#[test]
fn primary_model_uses_output_share_rather_than_input_or_first_model() {
    let mut f = WriterArchive::new();
    let mut first = message(
        "first",
        "models",
        Role::Assistant,
        0,
        vec![Block::Text {
            text: "First reply".into(),
        }],
    );
    first.model = Some("gpt-6.1-sol".into());
    first.usage.input = Some(5000);
    first.usage.output = Some(50);
    let mut dominant = message(
        "dominant",
        "models",
        Role::Assistant,
        1,
        vec![Block::Text {
            text: "Dominant reply".into(),
        }],
    );
    dominant.model = Some("claude-opus-5-5".into());
    dominant.usage.input = Some(100);
    dominant.usage.output = Some(2000);
    f.write("models", vec![first, dominant]);
    let pool = crate::ReadPool::open(&f.archive).unwrap();
    let r = pool.reader().unwrap();
    assert_eq!(
        r.session_profiles(&Selection::default()).unwrap()[0]
            .primary_model
            .as_deref(),
        Some("claude-opus-5-5")
    );
}

#[test]
fn title_topic_masks_before_truncation_without_rewriting_recorded_title() {
    let title = format!(
        "{}sk-ant-api03-{}-{}",
        "x ".repeat(55),
        "A".repeat(94),
        "A".repeat(8)
    );
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    f.session(SessionFacts {
        id: "codex:title",
        title: Some(&title),
        ..Default::default()
    })
    .unwrap();
    f.derive().unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let profile = r.session_profiles(&Selection::default()).unwrap().remove(0);
    assert_eq!(profile.title.as_deref(), Some(title.as_str()));
    assert!(!profile.facts.topic.unwrap().contains("sk-ant"));
}
