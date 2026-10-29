use super::*;
use crate::fixture::writer::{WriterArchive, bash, message, prose};
use polylogue_model::{Block, Role, Timestamp};

const MINUTE: i64 = 60_000;
fn m(id: &str, at: Option<i64>, author: &str, kind: &str, end: bool) -> Message {
    Message {
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        id: id.into(),
        at,
        author: author.into(),
        prompt: (author == "human").then(|| "prose".into()),
        stop: end.then(|| "end_turn".into()),
        last_block: Some(kind.into()),
        open_call: kind == "tool_call",
    }
}
fn lane(lanes: &[Lane], name: &str) -> i64 {
    lanes.iter().filter(|l| l.lane == name).map(|l| l.ms).sum()
}

#[test]
fn span_and_lanes_match_a_hand_computed_session() {
    // Morrow #2's 18 own message times, including harness tool results.
    let seconds = [
        0, 390, 780, 1170, 1590, 1814, 2038, 2262, 2485, 2709, 2933, 3157, 3381, 3605, 3828, 4052,
        4276, 4500,
    ];
    let messages: Vec<_> = seconds
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let author = if [0, 12, 16].contains(&i) {
                "human"
            } else if [2, 4, 6, 8, 10, 14].contains(&i) {
                "harness"
            } else {
                "model"
            };
            let kind = if author == "harness" {
                "tool_result"
            } else if [1, 3, 5, 7, 9, 13].contains(&i) {
                "tool_call"
            } else {
                "text"
            };
            m(
                &format!("m{i}"),
                Some(s * 1000),
                author,
                kind,
                [11, 15, 17].contains(&i),
            )
        })
        .collect();
    let mut tools = vec![];
    for (start, end, n, key) in [
        (390, 780, 2, "file_read"),
        (1170, 1590, 1, "subagent"),
        (1814, 2038, 2, "file_edit"),
        (2262, 2485, 1, "cargo test"),
        (2709, 2933, 2, "edit and test"),
        (3605, 3828, 5, "recorded calls"),
    ] {
        for _ in 0..n {
            tools.push(Tool {
                message_index: messages
                    .iter()
                    .position(|m| m.at == Some(start * 1000))
                    .unwrap(),
                key: key.into(),
                start: Some(start * 1000),
                end: Some(end * 1000),
                paired: true,
                failed: start == 2262,
            });
        }
    }
    let (span, lanes) = calculate(
        &messages,
        &tools,
        Some(0),
        Some(4_500_000),
        "messages",
        CLUSTER_GAP_MS,
    );
    assert_eq!(span.wall_ms, Some(75 * MINUTE));
    assert_eq!(span.clustered_ms, Some(62 * MINUTE));
    assert_eq!(span.tool_sum_ms, Some(3_434_000));
    assert_eq!(span.tool_union_ms, Some(1_704_000));
    assert_eq!(span.tool_overlap_ms, Some(1_730_000));
    assert_eq!(
        (
            lane(&lanes, "tool"),
            lane(&lanes, "model"),
            lane(&lanes, "human"),
            lane(&lanes, "unattributed")
        ),
        (1_704_000, 2_348_000, 448_000, 0)
    );
    assert_eq!(
        lanes.iter().map(|l| l.ms).sum::<i64>(),
        span.wall_ms.unwrap()
    );
    assert_eq!(
        (RULE_VERSION, serde_json::to_value(&span).unwrap()),
        (
            1,
            serde_json::json!({
                "first_at":0,"last_at":4500000,"time_source":"messages","wall_ms":4500000,"clustered_ms":3720000,"tool_sum_ms":3434000,"tool_union_ms":1704000,"tool_overlap_ms":1730000,
                "tool_untimed":0,"tool_no_result":0,"tool_same_timestamp":0,"tool_reversed":0,"timed_messages":18,"untimed_messages":0,"unordered_gaps":0,"first_failure_at":2262000,"prompts_to_failure":1,"prompted":true,"ends_with":"model_text","ends_with_message":"m17","last_stop_reason":"end_turn"
            })
        )
    );
    assert_eq!(span.first_failure_at, Some(2_262_000));
    assert_eq!(span.prompts_to_failure, 1);
    assert_eq!(span.ends_with, "model_text");
    assert_eq!(span.last_stop_reason.as_deref(), Some("end_turn"));
    let alternate = calculate(
        &messages,
        &tools,
        Some(0),
        Some(4_500_000),
        "messages",
        600_000,
    )
    .0;
    assert_eq!(alternate.clustered_ms, Some(75 * MINUTE));
}

#[test]
fn clustering_sensitivity_matches_the_morrow_shaped_sample() {
    let messages: Vec<_> = [0, 2, 4, 12, 15, 27, 29]
        .iter()
        .map(|n| m(&format!("{n}"), Some(n * MINUTE), "model", "text", false))
        .collect();
    let tool = Tool {
        message_index: 0,
        key: "cargo test".into(),
        start: Some(15 * MINUTE),
        end: Some(27 * MINUTE),
        paired: true,
        failed: false,
    };
    for (gap, without, with) in [(5, 9, 21), (10, 17, 29)] {
        assert_eq!(
            calculate(
                &messages,
                &[],
                Some(0),
                Some(29 * MINUTE),
                "messages",
                gap * MINUTE
            )
            .0
            .clustered_ms,
            Some(without * MINUTE)
        );
        assert_eq!(
            calculate(
                &messages,
                std::slice::from_ref(&tool),
                Some(0),
                Some(29 * MINUTE),
                "messages",
                gap * MINUTE
            )
            .0
            .clustered_ms,
            Some(with * MINUTE)
        );
    }
}

#[test]
fn tool_intervals_extend_clusters_and_share_overlap() {
    let messages = vec![
        m("a", Some(0), "model", "tool_call", false),
        m("b", Some(15 * MINUTE), "harness", "tool_result", false),
    ];
    let tools = vec![
        Tool {
            message_index: 0,
            key: "a".into(),
            start: Some(0),
            end: Some(10 * MINUTE),
            paired: true,
            failed: false,
        },
        Tool {
            message_index: 0,
            key: "b".into(),
            start: Some(5 * MINUTE),
            end: Some(15 * MINUTE),
            paired: true,
            failed: false,
        },
    ];
    let (span, lanes) = calculate(
        &messages,
        &tools,
        Some(0),
        Some(15 * MINUTE),
        "messages",
        CLUSTER_GAP_MS,
    );
    assert_eq!(
        (span.tool_sum_ms, span.tool_union_ms, span.tool_overlap_ms),
        (Some(20 * MINUTE), Some(15 * MINUTE), Some(5 * MINUTE))
    );
    assert_eq!(span.clustered_ms, Some(15 * MINUTE));
    let tool: Vec<_> = lanes.iter().filter(|l| l.lane == "tool").collect();
    assert_eq!((tool[0].ms, tool[1].ms), (450_000, 450_000));
    assert_eq!(lane(&lanes, "human"), 0);
}

#[test]
fn time_lanes_sum_to_wall_for_overlapping_intervals() {
    for start in 0..8 {
        for end in start..9 {
            for offset in 0..8 {
                let tools = vec![
                    (start, end, "first".into()),
                    (offset, offset + 5, "second".into()),
                ];
                let lanes = split(0, 12, &tools, &[(0, 6, "model"), (4, 12, "human")]);
                assert_eq!(lanes.iter().map(|l| l.ms).sum::<i64>(), 12);
                assert!(lanes.iter().all(|l| l.ms >= 0));
                assert_eq!(
                    lane(&lanes, "tool"),
                    union_ms(tools.iter().map(|(a, b, _)| (*a, *b)).collect())
                );
            }
        }
    }
}

#[test]
fn invalid_tool_times_and_missing_message_clocks_stay_unknown() {
    let messages = vec![m("last", None, "model", "text", true)];
    let tools = vec![
        Tool {
            message_index: 0,
            key: "none".into(),
            start: Some(0),
            end: None,
            paired: false,
            failed: false,
        },
        Tool {
            message_index: 0,
            key: "missing".into(),
            start: None,
            end: Some(1),
            paired: true,
            failed: false,
        },
        Tool {
            message_index: 0,
            key: "equal".into(),
            start: Some(0),
            end: Some(0),
            paired: true,
            failed: false,
        },
        Tool {
            message_index: 0,
            key: "reversed".into(),
            start: Some(2),
            end: Some(1),
            paired: true,
            failed: false,
        },
    ];
    let (span, lanes) = calculate(&messages, &tools, None, None, "absent", CLUSTER_GAP_MS);
    assert_eq!(
        (
            span.tool_untimed,
            span.tool_no_result,
            span.tool_same_timestamp,
            span.tool_reversed
        ),
        (4, 1, 1, 1)
    );
    assert_eq!(span.tool_sum_ms, None);
    assert_eq!(span.clustered_ms, None);
    assert_eq!(span.wall_ms, None);
    assert_eq!(span.ends_with, "unknown");
    assert!(lanes.is_empty());
}

#[test]
fn writer_pairs_tools_and_behind_reads_equal_the_next_derivation() {
    let mut f = WriterArchive::new();
    let mut messages = prose("clock", 1, 0, "Please run a test");
    let mut calls = bash("clock", 1, MINUTE, "cargo test", true, "ok");
    calls[1].at = Timestamp::from_unix_ms(13 * MINUTE);
    messages.extend(calls);
    let mut reply = message(
        "done",
        "clock",
        Role::Assistant,
        14 * MINUTE,
        vec![Block::Text {
            text: "Done".into(),
        }],
    );
    reply.stop_reason = Some("end_turn".into());
    messages.push(reply);
    f.write("clock", messages);
    let pool = crate::ReadPool::open(&f.archive).unwrap();
    let reader = pool.reader().unwrap();
    let row = &times(&reader, &Selection::default()).unwrap()[0];
    assert_eq!(row.span.as_ref().unwrap().tool_sum_ms, Some(12 * MINUTE));
    assert_eq!(row.span.as_ref().unwrap().clustered_ms, Some(14 * MINUTE));
    assert_eq!(lane(&row.lanes, "human"), 0);
    drop(reader);
    drop(pool);
    let conn = crate::archive::open_write(&f.archive.index_db(), 64).unwrap();
    conn.execute(
        "UPDATE message SET at=at+60000 WHERE id='claude-code:clock-done'",
        [],
    )
    .unwrap();
    conn.execute("UPDATE session SET changed=changed+1", [])
        .unwrap();
    let pool = crate::ReadPool::open(&f.archive).unwrap();
    let reader = pool.reader().unwrap();
    let computed = times(&reader, &Selection::default())
        .unwrap()
        .pop()
        .unwrap();
    assert!(computed.computed_at_read);
    drop(reader);
    drop(pool);
    f.writer.derive().unwrap();
    let pool = crate::ReadPool::open(&f.archive).unwrap();
    let reader = pool.reader().unwrap();
    let stored = times(&reader, &Selection::default())
        .unwrap()
        .pop()
        .unwrap();
    assert!(!stored.computed_at_read);
    assert_eq!(computed.span, stored.span);
    assert_eq!(computed.lanes, stored.lanes);
}

#[test]
fn notes_settings_and_read_clocks_do_not_change_stored_time_products() {
    use crate::fixture::{Fixture, MessageFacts, SessionFacts};
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let s = f
        .session(SessionFacts {
            id: "codex:independent",
            ..Default::default()
        })
        .unwrap();
    for (id, at, by) in [("h", 0, "human"), ("m", 60000, "model")] {
        f.message(
            s,
            MessageFacts {
                id,
                role: if by == "human" { "user" } else { "assistant" },
                written_by: by,
                prompt_kind: (by == "human").then_some("prose"),
                at: Some(at),
                ..Default::default()
            },
        )
        .unwrap();
    }
    let before = {
        let pool = f.reads().unwrap();
        let r = pool.reader().unwrap();
        compute(r.conn().unwrap(), s, CLUSTER_GAP_MS).unwrap()
    };
    f.note("tag", None, Some("new"), "session:codex:independent")
        .unwrap();
    f.setting("day.zone", "Pacific/Auckland").unwrap();
    f.setting("plan.example", "irrelevant setting").unwrap();
    let after = {
        let pool = f.reads().unwrap();
        let r = pool.reader().unwrap();
        compute(r.conn().unwrap(), s, CLUSTER_GAP_MS).unwrap()
    };
    assert_eq!(before, after);
}

#[test]
fn last_recorded_item_and_stop_reason_do_not_claim_task_outcomes() {
    for (author, kind, stop, expected) in [
        ("human", "text", false, "human_prompt"),
        ("model", "text", true, "model_text"),
        ("harness", "tool_result", false, "tool_result"),
        ("model", "tool_call", false, "tool_call_open"),
        ("harness", "text", false, "harness"),
    ] {
        let message = m("last", Some(0), author, kind, stop);
        let (span, _) = calculate(
            &[message],
            &[],
            Some(0),
            Some(0),
            "messages",
            CLUSTER_GAP_MS,
        );
        assert_eq!(span.ends_with, expected);
    }
    let mut command = m("command", Some(0), "human", "text", false);
    command.prompt = Some("command".into());
    assert_eq!(command.ends_with(), "human_command");
}
