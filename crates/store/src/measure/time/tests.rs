use super::*;
use crate::fixture::writer::{WriterArchive, bash, message, prose};
use polylogue_model::Json;
use polylogue_model::{Block, Role, Timestamp};
use serde_json::json;

const MINUTE: i64 = 60_000;
fn m(id: &str, at: Option<i64>, author: &str, kind: &str, end: bool) -> Message {
    Message {
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
    let (span, lanes) = calculate(&messages, &tools, CLUSTER_GAP_MS);
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
            // 2: the human lane is unknown without a provider turn end.
            // 3: a stored prefix ends at a session order of the message.
            3,
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
    let alternate = calculate(&messages, &tools, 600_000).0;
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
            calculate(&messages, &[], gap * MINUTE).0.clustered_ms,
            Some(without * MINUTE)
        );
        assert_eq!(
            calculate(&messages, std::slice::from_ref(&tool), gap * MINUTE)
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
    let (span, lanes) = calculate(&messages, &tools, CLUSTER_GAP_MS);
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
    let (span, lanes) = calculate(&messages, &tools, CLUSTER_GAP_MS);
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
    f.write_underived("clock", prose("clock", 2, 15 * MINUTE, "Thanks"));
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
fn assertions_settings_and_read_clocks_do_not_change_stored_time_products() {
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
    f.note("tag", Some("new"), None, "session:codex:independent")
        .unwrap();
    f.setting("browser.actions", "on").unwrap();
    f.setting("reading.variants", r#"["summary"]"#).unwrap();
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
        let (span, _) = calculate(&[message], &[], CLUSTER_GAP_MS);
        assert_eq!(span.ends_with, expected);
    }
    let mut command = m("command", Some(0), "human", "text", false);
    command.prompt = Some("command".into());
    assert_eq!(command.ends_with(), "human_command");
}

#[test]
fn waiting_human_is_unknown_without_a_provider_turn_end() {
    let prompt = |id, at| m(id, Some(at), "human", "text", false);
    let answer = |id, at, end| m(id, Some(at), "model", "text", end);
    // A provider that marks turn ends: the human lane is measured.
    let (_, lanes) = calculate(
        &[
            prompt("p1", 0),
            answer("a1", MINUTE, true),
            prompt("p2", 3 * MINUTE),
        ],
        &[],
        CLUSTER_GAP_MS,
    );
    assert_eq!(lane(&lanes, "human"), 2 * MINUTE);
    // One that reports none: the same gap is unattributed and the human
    // lane has no value, rather than 0.
    let (_, lanes) = calculate(
        &[
            prompt("p1", 0),
            answer("a1", MINUTE, false),
            prompt("p2", 3 * MINUTE),
        ],
        &[],
        CLUSTER_GAP_MS,
    );
    assert!(lanes.iter().all(|l| l.lane != "human"), "{lanes:?}");
    assert_eq!(lane(&lanes, "unattributed"), 2 * MINUTE);
    assert_eq!(lane(&lanes, "model"), MINUTE);
}

/// The stored span and lanes of the session, and the full calculation over
/// its rows.
fn stored_and_computed(conn: &rusqlite::Connection, session: &str) -> (Json, Json) {
    let pk: i64 = conn
        .query_row(
            "SELECT pk FROM session WHERE id = ?1",
            [format!("claude-code:{session}")],
            |r| r.get(0),
        )
        .unwrap();
    let lanes: Vec<Lane> = conn
        .prepare("SELECT lane,key,ms FROM session_time WHERE session_pk=?1 ORDER BY lane,key")
        .unwrap()
        .query_map([pk], |r| {
            Ok(Lane {
                lane: r.get(0)?,
                key: r.get(1)?,
                ms: r.get(2)?,
            })
        })
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    let stored = json!([stored(conn, pk).unwrap(), lanes]);
    let (span, lanes) = compute(conn, pk, CLUSTER_GAP_MS).unwrap();
    (stored, json!([Some(span), lanes]))
}

fn index(w: &WriterArchive) -> rusqlite::Connection {
    rusqlite::Connection::open(w.archive.index_db()).unwrap()
}

fn answer(n: u32, at: i64) -> polylogue_model::Message {
    let mut answer = message(
        &format!("a{n}"),
        "s",
        Role::Assistant,
        at,
        vec![Block::Text {
            text: "an answer".into(),
        }],
    );
    answer.stop_reason = Some("end_turn".into());
    answer
}

#[test]
fn span_fold_after_append_equals_full_recompute() {
    use crate::fixture::writer::sid;
    use polylogue_model::{
        BackgroundUpdate, Event, Output, Session, SessionKind, TimeQuality, ToolAction, ToolOutcome,
    };
    let mut w = WriterArchive::new();
    let mut seen = Vec::new();
    let mut check = |w: &WriterArchive, stage: &str| {
        let (stored, computed) = stored_and_computed(&index(w), "s");
        assert_eq!(stored, computed, "{stage}");
        seen.push(stored);
    };
    let late = |mut message: polylogue_model::Message, at: i64| {
        message.at = Timestamp::from_unix_ms(at);
        vec![message]
    };
    w.write("s", prose("s", 1, 1_000, "please look"));
    check(&w, "first prompt");
    w.write("s", vec![answer(1, 2_000)]);
    check(&w, "answer");
    let mut first = bash("s", 1, 3_000, "cargo build", true, "built");
    let result = first.pop().unwrap();
    w.write("s", first);
    check(&w, "call without its result");
    w.write("s", late(result, 4_000));
    check(&w, "late result");
    // Two calls in separate appends, each answered later: the first call's
    // interval crosses the second call's message.
    let mut second = bash("s", 2, 5_000, "cargo test", true, "ok");
    let second_result = second.pop().unwrap();
    let mut third = bash("s", 3, 5_500, "cargo clippy", true, "ok");
    let third_result = third.pop().unwrap();
    w.write("s", second);
    check(&w, "second call");
    w.write("s", third);
    check(&w, "third call while the second runs");
    w.write("s", late(second_result, 6_000));
    check(&w, "second result");
    w.write("s", late(third_result, 6_500));
    check(&w, "third result");
    w.write("s", bash("s", 4, 7_000, "cargo test", false, "exit 101"));
    check(&w, "a failed call");
    w.write("s", prose("s", 2, 7_500, "and now"));
    check(&w, "a plain append");
    // A call left waiting once a model message follows it, answered later.
    let mut fifth = bash("s", 5, 7_600, "cargo bench", true, "done");
    let fifth_result = fifth.pop().unwrap();
    w.write("s", fifth);
    check(&w, "a call waiting for its result");
    w.write("s", vec![answer(3, 7_700)]);
    check(&w, "a model message after the waiting call");
    w.write("s", late(fifth_result, 7_800));
    check(&w, "the waiting call's late result");
    // A code call, then the session's first command a script ran: code
    // calls stop counting as actions.
    let code = |n: u32, at: i64, run_by_script: bool, action: ToolAction| {
        message(
            &format!("k{n}"),
            "s",
            Role::Assistant,
            at,
            vec![Block::ToolCall {
                written_text: None,
                run_by_script,
                call_id: Some(format!("k{n}")),
                tool: Some("python".into()),
                action,
                input: Json::Null,
                command: Some("run".into()),
                path: None,
            }],
        )
    };
    let output = message(
        "k1-output",
        "s",
        Role::User,
        7_850,
        vec![Block::ToolResult {
            reported_duration_ms: None,
            call_id: Some("k1".into()),
            outcome: ToolOutcome::Ok,
            output: Some("ran".into()),
            data: None,
            attachments: vec![],
        }],
    );
    w.write(
        "s",
        vec![
            code(1, 7_820, false, ToolAction::Code),
            output,
            answer(4, 7_870),
        ],
    );
    check(&w, "a code call");
    w.write("s", vec![code(2, 7_900, true, ToolAction::Shell)]);
    check(&w, "a command a script ran");
    // A message timed before the latest one arrives after it, between two
    // later ones in one append: no message before it may settle, since the
    // gap after it reaches back before them.
    let mut notes = vec![answer(5, 7_960)];
    notes.extend(prose("s", 3, 2_500, "an earlier note"));
    notes.push(answer(6, 8_100));
    w.write("s", notes);
    check(&w, "a message out of time order");
    w.write("s", vec![answer(2, 8_000)]);
    check(&w, "append after the out-of-order message");
    // The failed call's outcome is reported again outside its messages.
    let offset = w.offset();
    w.write_output(
        "input-s",
        "claude-code",
        "/p/s",
        Output {
            sessions: vec![Session::new(sid("s"), SessionKind::Main)],
            background: vec![BackgroundUpdate {
                result_at: None,
                result_at_evidence: Default::default(),
                reported_duration_ms: None,
                session: sid("s"),
                task_id: "task".into(),
                call_id: Some("t4".into()),
                outcome: ToolOutcome::Ok,
                output_file: None,
            }],
            ..Output::default()
        },
        1,
    );
    assert!(w.offset() > offset);
    w.writer.derive().unwrap();
    check(&w, "an outcome reported outside the call's messages");
    // A rewrite: a snapshot states the first answer again, later.
    w.write_snapshot("s", vec![answer(1, 2_200)]);
    check(&w, "rewrite");
    w.write_from(
        "input-copy",
        Session::new(sid("s"), SessionKind::Main),
        prose("s", 9, 1_500, "a prompt only the copy has"),
    );
    check(&w, "new input");
    // A call left waiting at the end, then a compaction and a marker naming
    // it: every earlier message leaves the path in the derive step, while
    // the writer changed only messages after the settled one.
    w.write("s", {
        let mut call = bash("s", 6, 9_500, "cargo doc", true, "built");
        call.pop();
        call
    });
    check(&w, "a call left waiting at the end");
    let event = |kind: &str, after: Option<&str>| Event {
        session: Some(sid("s")),
        kind: kind.into(),
        at: None,
        at_quality: TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        after: after.map(|id| polylogue_model::MessageId::parse(id).unwrap()),
        range: None,
        data: Json::Null,
    };
    w.write_output(
        "input-s",
        "claude-code",
        "/p/s",
        Output {
            sessions: vec![Session::new(sid("s"), SessionKind::Main)],
            events: vec![
                event("compaction", None),
                event("active_leaf", Some("claude-code:s-c6")),
            ],
            ..Output::default()
        },
        2,
    );
    w.writer.derive().unwrap();
    check(&w, "compaction and new marker");
    seen.dedup();
    assert_eq!(seen.len(), 22, "every stage changes the span");
}

#[test]
fn an_append_derives_the_span_from_the_messages_after_the_settled_one() {
    let mut w = WriterArchive::new();
    let mut messages = Vec::new();
    for n in 0..100 {
        messages.extend(prose("s", n, 1_000 + i64::from(n) * 20, "a question"));
        messages.push(answer(n, 1_010 + i64::from(n) * 20));
    }
    w.write("s", messages);
    w.write_underived("s", prose("s", 100, 5_000, "one more"));
    let mut conn = index(&w);
    // The fixture's writer stays alive and may still commit to the index
    // (search entries, a log checkpoint); taking the write lock first keeps
    // this transaction's snapshot current until its own writes.
    let tx = conn
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .unwrap();
    let pk: i64 = tx
        .query_row("SELECT pk FROM session WHERE id='claude-code:s'", [], |r| {
            r.get(0)
        })
        .unwrap();
    let (derived, read) = derive(&tx, pk, false).unwrap();
    assert_eq!(read, 1);
    write_derived(&tx, pk, &derived).unwrap();
    let (stored, computed) = stored_and_computed(&tx, "s");
    assert_eq!(stored, computed);
    // From the start, the same derivation reads all 201 messages.
    let (derived, read) = derive(&tx, pk, true).unwrap();
    assert_eq!(read, 201);
    write_derived(&tx, pk, &derived).unwrap();
    assert_eq!(stored, stored_and_computed(&tx, "s").0);
}

#[test]
fn a_stored_prefix_changes_shape_only_with_a_new_rule_version() {
    // A stored prefix is read back by the version that wrote it; a prefix of
    // another shape would not decode. Bumping RULE_VERSION derives every
    // session from its start, so a change here needs a new version.
    let prefix = Prefix {
        end: Some((("1order".into(), "m".into()), 3)),
        script: true,
        first: Some(4),
        last: Some(5),
        timed: 6,
        untimed: 7,
        unordered_gaps: 8,
        previous: Some((Some(9), "model".into())),
        tools: 10,
        intervals: 11,
        tool_untimed: 12,
        tool_no_result: 13,
        tool_same_timestamp: 14,
        tool_reversed: 15,
        tool_sum_ms: 16,
        tool_union_ms: 17,
        clustered_ms: 18,
        prompts: 19,
        failure: Some((20, 21)),
        stop_reported: true,
        last_model_stop: Some(Some("end_turn".into())),
        ends: Some(("model_text".into(), "m".into())),
        lanes: vec![Lane {
            lane: "tool".into(),
            key: "bash".into(),
            ms: 22,
        }],
    };
    assert_eq!(
        (RULE_VERSION, serde_json::to_value(&prefix).unwrap()),
        (
            3,
            json!({
                "end":[["1order","m"],3],"script":true,"first":4,"last":5,"timed":6,"untimed":7,
                "unordered_gaps":8,"previous":[9,"model"],"tools":10,"intervals":11,"tool_untimed":12,
                "tool_no_result":13,"tool_same_timestamp":14,"tool_reversed":15,"tool_sum_ms":16,
                "tool_union_ms":17,"clustered_ms":18,"prompts":19,"failure":[20,21],"stop_reported":true,
                "last_model_stop":"end_turn","ends":["model_text","m"],
                "lanes":[{"lane":"tool","key":"bash","ms":22}]
            })
        )
    );
}
