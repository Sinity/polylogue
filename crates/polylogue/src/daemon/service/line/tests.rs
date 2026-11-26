mod read_after_write;
mod saved_queries;

use crate::test_daemon::{self, Call, TestDaemon};
use serde_json::{Value, json};

/// A daemon holding two Claude Code sessions, `s1` and `s2`.
fn two_sessions() -> TestDaemon {
    test_daemon::start_with_sources_and_workers(
        "first prompt",
        &|home| {
            std::fs::write(
                home.join(".claude/projects/-p/s2.jsonl"),
                polylogue_testkit::claude_user("u2", "s2", "second prompt"),
            )
            .unwrap();
        },
        1,
        2,
    )
}

fn line(d: &TestDaemon, id: &str, rows: Value) -> command_surface::Answer<Value> {
    Call::new("rows", rows).id(id).answer(&d.daemon)
}

#[track_caller]
fn steps(d: &TestDaemon, id: &str, rows: Value) -> Vec<Value> {
    let answer = crate::test_answers::ok(line(d, id, rows));
    assert_eq!(answer["kind"], "steps", "{answer}");
    answer["reply"]["steps"].as_array().unwrap().clone()
}

#[track_caller]
fn refusal(answer: command_surface::Answer<Value>) -> command_surface::Failure {
    match answer.outcome {
        command_surface::Outcome::Error(error) => error,
        other => panic!("{other:?}"),
    }
}

#[track_caller]
fn count(d: &TestDaemon, query: &str) -> u64 {
    let answer = Call::new("rows", json!({ "selection": query })).answer(&d.daemon);
    let value = answer
        .value()
        .cloned()
        .unwrap_or_else(|| panic!("{answer:?}"));
    value["reply"]["count"].as_u64().unwrap()
}

#[test]
fn rows_listing_preserves_fields_and_pages() {
    let d = two_sessions();
    let listed = crate::test_answers::ok(line(
        &d,
        "listing",
        json!({"selection":"sessions | sort by id", "fields":["id"], "page_size":1}),
    ));
    assert_eq!(listed["kind"], "listing");
    assert_eq!(listed["reply"]["rows"], json!([{"id":"claude-code:s1"}]));
    assert!(listed["reply"]["next"].is_string());
    let next = listed["reply"]["next"].as_str().unwrap();
    let second = Call::new(
        "rows",
        json!({"selection":"sessions | sort by id", "fields":["id"], "page_size":1}),
    )
    .id("listing-second")
    .cursor(Some(next))
    .answer(&d.daemon);
    let second = crate::test_answers::ok(second);
    assert_eq!(second["reply"]["rows"], json!([{"id":"claude-code:s2"}]));
    assert!(second["reply"]["next"].is_null(), "{second}");
}

#[test]
fn malformed_line_shapes_are_refused_before_anything_resolves() {
    let d = two_sessions();
    for (rows, message) in [
        (json!({"selection":" "}), "selection must not be empty"),
        (
            json!({"selection":"sessions","references":["session:claude-code:s1"]}),
            "a line takes a selection or references, not both",
        ),
        (json!({}), "a listing needs a selection"),
        (
            json!({"steps":[{"verb":"read"}]}),
            "read needs rows: name a selection; on a terminal the command line opens the picker",
        ),
        (
            json!({"steps":[{"verb":"tag","name":"x"}]}),
            "tag needs rows first: name a selection",
        ),
    ] {
        let refused = refusal(line(&d, "shape", rows.clone()));
        assert_eq!(refused.code.as_deref(), Some("bad_request"), "{rows}");
        assert_eq!(refused.message, message, "{rows}");
    }
}

#[test]
fn selection_resolved_once_across_steps() {
    let d = test_daemon::start("a prompt about the clock");
    let steps = steps(
        &d,
        "seen",
        json!({
            "selection": "sessions where NOT tag:seen",
            "steps": [{"verb": "tag", "name": "seen"}, {"verb": "read", "view": "dialogue"}],
        }),
    );
    assert_eq!(
        steps[0]["outcome"]["reply"]["reply"]["targets"], 1,
        "{steps:?}"
    );
    let page: polylogue_model::read::ReadPage =
        serde_json::from_value(steps[1]["outcome"]["reply"]["reply"].clone()).unwrap();
    assert!(
        page.document.items.iter().any(|item| matches!(
            &item.kind,
            polylogue_model::read::ItemKind::Text { text } if text.contains("a prompt about the clock")
        )),
        "{page:?}"
    );
}

#[test]
fn failing_step_stops_line_and_prints_undo() {
    let d = test_daemon::start("first prompt");
    let steps = steps(
        &d,
        "line",
        json!({
            "selection": "session:claude-code:s1",
            "steps": [
                {"verb": "note", "text": "flaky because the clock is real"},
                {"verb": "edit", "text": "steady"},
                {"verb": "read"},
            ],
        }),
    );
    let statuses: Vec<&Value> = steps
        .iter()
        .map(|step| &step["outcome"]["status"])
        .collect();
    assert_eq!(statuses, ["done", "failed", "not_run"]);
    assert_eq!(
        steps[0]["outcome"]["undo"],
        json!([
            r"printf '%s\n' 'annotation:line-1@line-1' | polylogue --request-id line-1-undo - withdraw"
        ])
    );
    assert_eq!(steps[1]["outcome"]["error"]["kind"], "bad_request");
    assert_eq!(steps[1]["request_id"], "line-2");
}

#[test]
fn query_selection_write_needs_all() {
    let d = two_sessions();
    let tag = |all: bool| json!({"selection": "sessions", "steps": [{"verb": "tag", "name": "later"}], "all": all});
    let refused = refusal(line(&d, "without", tag(false)));
    assert_eq!(refused.code.as_deref(), Some("needs_all"));
    assert_eq!(
        refused.message,
        "2 sessions of \"sessions\" match; a write or act on all of them needs --all"
    );
    assert_eq!(refused.candidates().len(), 2);
    // The refusal carries the same line with `all` set, ready to run.
    let Some(command_surface::Suggestion::Request(retry)) = refused.suggestion() else {
        panic!("{refused:?}");
    };
    assert_eq!(retry.arguments, tag(true));
    assert_eq!(count(&d, "sessions where tag:later"), 0);
    let answer = crate::test_answers::ok(Call::from_request(retry).answer(&d.daemon));
    assert_eq!(
        answer["reply"]["steps"][0]["outcome"]["reply"]["reply"]["targets"],
        2
    );
    assert_eq!(count(&d, "sessions where tag:later"), 2);
}

#[test]
fn edit_without_text_gives_the_lines_that_edit_by_hand() {
    let d = two_sessions();
    let steps = steps(
        &d,
        "edit-without-text",
        json!({"selection": "annotation:some-note", "steps": [{"verb": "edit"}]}),
    );
    assert_eq!(steps[0]["outcome"]["status"], "failed");
    let refused = &steps[0]["outcome"]["error"];
    assert_eq!(refused["code"], "bad_request");
    assert_eq!(
        refused["message"],
        "edit needs its new text; with no text a terminal opens your editor. Without one, write the text out with `polylogue annotation:some-note read --to FILE.md`, change the file, and check it in with `polylogue import FILE.md`"
    );
}

/// docs/agent-context.md: three or more sessions named together inject an
/// index, one line per session, never a takeover of each.
#[test]
fn three_named_sessions_brief_as_one_index() {
    let d = test_daemon::start_with_sources_and_workers(
        "first prompt",
        &|home| {
            for (message, session) in [("u2", "s2"), ("u3", "s3")] {
                std::fs::write(
                    home.join(format!(".claude/projects/-p/{session}.jsonl")),
                    polylogue_testkit::claude_user(message, session, "another prompt"),
                )
                .unwrap();
            }
        },
        2,
        2,
    );
    let steps = steps(
        &d,
        "index",
        json!({
            "references": ["session:claude-code:s1", "session:claude-code:s2", "session:claude-code:s3"],
            "steps": [{"verb": "brief"}],
        }),
    );
    let text = steps[0]["outcome"]["reply"]["reply"]["prepared"]["text"]
        .as_str()
        .unwrap();
    assert!(
        text.starts_with("[polylogue session index delivery:"),
        "{text}"
    );
    assert!(
        text.contains("index of 3 sessions named together"),
        "{text}"
    );
    for session in ["s1", "s2", "s3"] {
        assert_eq!(
            text.matches(&format!("\"session:claude-code:{session}\""))
                .count(),
            1,
            "{session}: {text}"
        );
    }
    assert!(!text.contains("takeover"), "{text}");
    assert!(
        text.trim_end()
            .ends_with("`polylogue REF brief` for any of these."),
        "{text}"
    );
}

#[test]
fn named_references_need_no_all() {
    let d = two_sessions();
    let steps = steps(
        &d,
        "named",
        json!({
            "references": ["session:claude-code:s1", "session:claude-code:s2"],
            "steps": [{"verb": "tag", "name": "later"}],
        }),
    );
    assert_eq!(steps[0]["outcome"]["reply"]["reply"]["targets"], 2);
    assert_eq!(count(&d, "sessions where tag:later"), 2);
}

#[test]
fn dry_run_line_writes_nothing() {
    let d = test_daemon::start("first prompt");
    let steps = steps(
        &d,
        "dry",
        json!({
            "selection": "session:claude-code:s1",
            "steps": [{"verb": "note", "text": "flaky"}],
            "dry_run": true,
        }),
    );
    assert_eq!(steps[0]["outcome"]["status"], "would_do");
    assert_eq!(
        steps[0]["outcome"]["would"],
        "would write a note about each of session:claude-code:s1"
    );
    assert_eq!(count(&d, "annotations where kind:note"), 0);
}

#[test]
fn dry_run_tag_names_the_tag_and_writes_nothing() {
    let d = test_daemon::start("first prompt");
    let steps = steps(
        &d,
        "dry-tag",
        json!({
            "selection": "session:claude-code:s1",
            "steps": [{"verb": "tag", "name": "later"}],
            "dry_run": true,
        }),
    );
    assert_eq!(steps[0]["outcome"]["status"], "would_do");
    assert_eq!(
        steps[0]["outcome"]["would"],
        "would tag session:claude-code:s1 later"
    );
    assert_eq!(count(&d, "sessions where tag:later"), 0);
}

#[test]
fn grouped_selection_refuses_write_before_running() {
    let d = test_daemon::start("first prompt");
    let refused = refusal(line(
        &d,
        "grouped",
        json!({
            "selection": "sessions | group by repo",
            "steps": [{"verb": "note", "text": "first"}, {"verb": "tag", "name": "later"}],
        }),
    ));
    assert_eq!(
        refused.message,
        "note needs archive rows; the selection is grouped or measured"
    );
}

#[test]
fn follow_must_be_last() {
    let d = test_daemon::start("first prompt");
    let refused = refusal(line(
        &d,
        "follow",
        json!({
            "selection": "session:claude-code:s1",
            "steps": [{"verb": "read", "follow": true}, {"verb": "tag", "name": "later"}],
        }),
    ));
    assert_eq!(refused.message, "read --follow must be the last step");
    assert_eq!(count(&d, "sessions where tag:later"), 0);
}

#[test]
fn retried_line_writes_once() {
    let d = test_daemon::start("first prompt");
    let rows = json!({
        "selection": "session:claude-code:s1",
        "steps": [{"verb": "note", "text": "flaky because the clock is real"}],
    });
    let first = steps(&d, "again", rows.clone());
    let second = steps(&d, "again", rows);
    assert_eq!(first[0]["outcome"]["reply"]["reply"]["repeat"], false);
    assert_eq!(second[0]["outcome"]["reply"]["reply"]["repeat"], true);
    assert_eq!(count(&d, "annotations where kind:note"), 1);
}

#[test]
fn daemon_refuses_row_numbers() {
    let d = test_daemon::start("first prompt");
    for selection in ["@1", "@1.2", "@2-5", "@1,4", "@all"] {
        let failure = refusal(line(&d, "row", json!({"selection": selection})));
        assert_eq!(failure.code.as_deref(), Some("row_numbers_off_terminal"));
        assert_eq!(
            failure.message,
            "row numbers are resolved by the command line; use the references printed above"
        );
    }
    let failure = refusal(line(
        &d,
        "references",
        json!({"references": ["@1"], "steps": [{"verb": "read"}]}),
    ));
    assert_eq!(failure.code.as_deref(), Some("row_numbers_off_terminal"));
}

#[test]
fn answer_reference_refused_with_run_line() {
    let d = test_daemon::start("first prompt");
    let answer = line(
        &d,
        "answer-reference",
        json!({"selection": "answer:sample:run"}),
    );
    let failure = refusal(answer.clone());
    assert_eq!(failure.kind, command_surface::FailureKind::BadRequest);
    assert_eq!(
        failure.message,
        "answer references are query runs; use polylogue run:sample:run read"
    );
    let next = &answer.envelope.next[0].request;
    assert_eq!(next.operation, "rows");
    assert_eq!(
        next.arguments,
        json!({"selection": "run:sample:run", "steps": [{"verb": "read", "view": "transcript"}]})
    );
    assert_eq!(
        failure.suggestion(),
        Some(&command_surface::Suggestion::Request(next.clone()))
    );
}

#[test]
fn mcp_follow_refused_with_read_call() {
    let d = test_daemon::start("first prompt");
    let rows = json!({"selection": "session:claude-code:s1", "steps": [{"verb": "read", "follow": true, "view": "dialogue"}]});
    let answer = Call::new("rows", rows)
        .channel(command_surface::Via::Mcp)
        .answer(&d.daemon);
    assert_eq!(
        refusal(answer.clone()).message,
        "a model cannot hold a stream; read again to continue"
    );
    let next = &answer.envelope.next[0].request;
    assert_eq!(next.operation, "rows");
    assert_eq!(
        next.arguments,
        json!({"selection": "session:claude-code:s1", "steps": [{"verb": "read", "view": "dialogue"}]})
    );
    let read = crate::test_answers::ok(
        Call::from_request(next)
            .channel(command_surface::Via::Mcp)
            .answer(&d.daemon),
    );
    assert_eq!(read["reply"]["steps"][0]["outcome"]["status"], "done");
}

#[test]
fn listing_wait_returns_when_rows_match() {
    let d = test_daemon::start("first prompt");
    std::thread::scope(|scope| {
        let waiting = scope.spawn(|| {
            line(
                &d,
                "wait",
                json!({"selection": "sessions where tag:ready", "wait": "1m"}),
            )
        });
        // The listing is waiting, or about to: either way the tag below is
        // the change it answers with.
        while d.daemon.requests.running() == 0 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let tagged = steps(
            &d,
            "ready",
            json!({"selection": "session:claude-code:s1", "steps": [{"verb": "tag", "name": "ready"}]}),
        );
        assert_eq!(tagged[0]["outcome"]["status"], "done");
        let answer = crate::test_answers::ok(waiting.join().unwrap());
        assert_eq!(answer["reply"]["rows"].as_array().unwrap().len(), 1);
        assert_eq!(answer["reply"]["rows"][0]["id"], "claude-code:s1");
    });
}

/// A waiting listing on an ID no session has yet keeps waiting, and returns
/// the session once it is imported (docs/query.md, References).
#[test]
fn a_waiting_listing_on_an_absent_id_returns_when_the_session_appears() {
    let d = test_daemon::start("first prompt");
    std::thread::scope(|scope| {
        let waiting = scope.spawn(|| {
            // The import takes longer than a test request's default deadline,
            // which would otherwise cut the wait short.
            let mut call = Call::new(
                "rows",
                json!({"selection": "sessions where id:claude-code:s2", "wait": "1m"}),
            )
            .id("wait-for-s2");
            call.deadline = command_surface::Deadline::None;
            call.answer(&d.daemon)
        });
        // The listing is waiting, or about to: either way the import below
        // is the change it answers with.
        while d.daemon.requests.running() == 0 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        std::fs::write(
            d.home.join(".claude/projects/-p/s2.jsonl"),
            polylogue_testkit::claude_user("u2", "s2", "second prompt"),
        )
        .unwrap();
        // The new session's derivation may still be running: the listing
        // then answers degraded, with the row.
        let answer = waiting.join().unwrap();
        let listing = answer
            .value()
            .unwrap_or_else(|| panic!("{answer:?}"))
            .clone();
        assert_eq!(
            listing["reply"]["rows"].as_array().unwrap().len(),
            1,
            "{listing}"
        );
        assert_eq!(listing["reply"]["rows"][0]["id"], "claude-code:s2");
    });
}

/// A reference selection that names no session is `not_found`; only a
/// query's condition on an absent ID answers empty.
#[test]
fn a_reference_that_names_nothing_answers_not_found() {
    let d = test_daemon::start("first prompt");
    let read = steps(
        &d,
        "absent-reference",
        json!({"selection": "session:claude-code:absent", "steps": [{"verb": "read"}]}),
    );
    assert_eq!(read[0]["outcome"]["status"], "failed", "{read:?}");
    assert_eq!(read[0]["outcome"]["error"]["kind"], "not_found", "{read:?}");
    assert_eq!(read[0]["outcome"]["error"]["code"], "not_found", "{read:?}");
}

#[test]
fn listing_wait_caps_long_mcp_requests() {
    use std::time::Duration;

    assert_eq!(
        super::listing_wait("1h", polylogue_model::authorship::Channel::Mcp).unwrap(),
        Duration::from_secs(50)
    );
    assert_eq!(
        super::listing_wait("25ms", polylogue_model::authorship::Channel::Mcp).unwrap(),
        Duration::from_millis(25)
    );
    assert_eq!(
        super::listing_wait("1h", polylogue_model::authorship::Channel::Cli).unwrap(),
        Duration::from_secs(3600)
    );
}

#[test]
fn mcp_wait_returns_empty_with_same_call() {
    let d = test_daemon::start("first prompt");
    let request = command_surface::RunnableRequest {
        operation: "rows".into(),
        arguments: json!({"selection": "sessions where tag:later", "wait": "25ms", "fields": [], "all": false}),
        request_id: "wait-retry".into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    };
    let answer = Call::from_request(&request)
        .channel(command_surface::Via::Mcp)
        .answer(&d.daemon);
    assert!(
        matches!(answer.outcome, command_surface::Outcome::Empty(_)),
        "{answer:?}"
    );
    assert_eq!(answer.envelope.next.len(), 1);
    assert_eq!(answer.envelope.next[0].request, request);
}

#[test]
fn listing_wait_answers_before_request_deadline() {
    let d = test_daemon::start("first prompt");
    let answer = Call::new(
        "rows",
        json!({"selection": "sessions where tag:later", "wait": "1h"}),
    )
    .within(&polylogue_store::read::limit::ReadLimit::new(
        std::time::Duration::from_secs(5),
        Default::default(),
    ))
    .answer(&d.daemon);
    let empty = crate::test_answers::empty(answer.clone());
    assert_eq!(empty["reply"]["rows"], json!([]));
    assert_eq!(answer.envelope.next.len(), 1);
    assert_eq!(
        answer.envelope.next[0].request.arguments,
        json!({"selection": "sessions where tag:later", "wait": "1h"})
    );
}

fn socket_events(d: &TestDaemon, id: &str, arguments: Value) -> Vec<(String, Value)> {
    let request = crate::client::request("rows", arguments, id.into());
    let mut events = Vec::new();
    let answer = crate::client::call_streaming(
        &d.daemon.archive,
        &request,
        Some(std::time::Duration::from_secs(10)),
        |frame| {
            events.push((frame.event, serde_json::from_str(&frame.data).unwrap()));
            Ok(false)
        },
    )
    .unwrap()
    .expect("the line answers");
    events.push(("answer".into(), serde_json::to_value(answer).unwrap()));
    events
}

#[test]
fn socket_chain_streams_each_read_at_its_step_and_retries_writes_once() {
    let d = test_daemon::start("ordered stream prompt");
    let arguments = json!({
        "selection": "sessions where NOT tag:done",
        "steps": [
            {"verb":"tag", "name":"done"},
            {"verb":"read", "view":"transcript", "delivery":"stream"},
            {"verb":"note", "text":"after-note"},
            {"verb":"read", "view":"transcript", "delivery":"stream"}
        ]
    });
    for retry in [false, true] {
        let events = socket_events(&d, "ordered", arguments.clone());
        let names: Vec<_> = events.iter().map(|(event, _)| event.as_str()).collect();
        assert_eq!(
            names.iter().filter(|event| **event == "begin").count(),
            2,
            "{events:?}"
        );
        assert_eq!(
            names.iter().filter(|event| **event == "complete").count(),
            2
        );
        assert_eq!(names.last(), Some(&"answer"));
        let reports: Vec<_> = events
            .iter()
            .filter(|(name, _)| name == "steps" || name == "answer")
            .flat_map(|(_, answer)| answer["value"]["reply"]["steps"].as_array().unwrap().iter())
            .collect();
        assert_eq!(
            reports
                .iter()
                .map(|report| report["verb"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["tag", "read", "note", "read"]
        );
        for index in [0, 2] {
            assert_eq!(reports[index]["outcome"]["reply"]["reply"]["repeat"], retry);
        }
        let completions: Vec<_> = events
            .iter()
            .filter(|(event, _)| event == "complete")
            .map(|(_, value)| value)
            .collect();
        for (report, complete) in [reports[1], reports[3]].into_iter().zip(completions) {
            assert_eq!(report["outcome"]["reply"]["kind"], "streamed");
            assert_eq!(&report["outcome"]["reply"]["reply"], complete);
        }
        let mut reads = vec![String::new(), String::new()];
        let mut index = 0;
        for (name, value) in &events {
            if name == "record" {
                reads[index].push_str(&value.to_string());
            }
            if name == "complete" {
                index += 1;
            }
        }
        assert!(
            reads
                .iter()
                .all(|read| read.contains("ordered stream prompt")),
            "{reads:?}"
        );
        assert!(
            reads.iter().all(|read| read.contains("\"key\":\"done\"")),
            "{reads:?}"
        );
        if !retry {
            assert!(!reads[0].contains("after-note"), "{reads:?}");
            assert!(reads[1].contains("after-note"), "{reads:?}");
        }
        assert_eq!(count(&d, "annotations where kind:tag OR kind:note"), 2);
    }
    let changed = json!({"selection":"sessions", "steps":[{"verb":"note", "text":"another"}]});
    assert_eq!(
        refusal(line(&d, "ordered", changed)).kind,
        command_surface::FailureKind::Conflict
    );
}

#[test]
fn stream_prefix_failure_keeps_undo_and_does_not_start_a_read() {
    let d = test_daemon::start("prefix failure prompt");
    for retry in [false, true] {
        let events = socket_events(
            &d,
            "prefix",
            json!({
                "references":["session:claude-code:s1"],
                "steps":[{"verb":"note","text":"kept-note"},{"verb":"edit","text":"cannot-edit-session"},
                    {"verb":"read","delivery":"stream"}]
            }),
        );
        assert!(
            events
                .iter()
                .all(|(name, _)| name == "steps" || name == "answer"),
            "{events:?}"
        );
        let reports: Vec<_> = events
            .iter()
            .flat_map(|(_, answer)| answer["value"]["reply"]["steps"].as_array().unwrap().iter())
            .collect();
        assert_eq!(
            reports
                .iter()
                .map(|step| step["outcome"]["status"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["done", "failed", "not_run"]
        );
        assert_eq!(
            reports[0]["outcome"]["undo"],
            json!([
                r"printf '%s\n' 'annotation:prefix-1@prefix-1' | polylogue --request-id prefix-1-undo - withdraw"
            ])
        );
        assert_eq!(reports[0]["outcome"]["reply"]["reply"]["repeat"], retry);
        assert_eq!(count(&d, "annotations where kind:note"), 1);
    }
}

#[test]
fn empty_fixed_selection_streams_no_records_and_keeps_empty_outcome() {
    let d = test_daemon::start("not selected");
    let events = socket_events(
        &d,
        "empty",
        json!({
            "selection":"sessions where 0 = 1", "steps":[
                {"verb":"tag","name":"no-target"}, {"verb":"read","delivery":"stream"}]
        }),
    );
    assert!(
        events.iter().all(|(name, _)| name != "record"),
        "{events:?}"
    );
    assert_eq!(events.last().unwrap().1["outcome"], "empty", "{events:?}");
    assert_eq!(count(&d, "annotations"), 0);
}

#[test]
fn streamed_read_keeps_the_fixed_query_order() {
    let d = two_sessions();
    let events = socket_events(
        &d,
        "ordered-population",
        json!({
            "selection":"sessions | sort by id desc", "all":true,
            "steps":[{"verb":"tag","name":"ordered"},{"verb":"read","delivery":"stream","view":"transcript"}]
        }),
    );
    let sessions: Vec<String> = events
        .iter()
        .filter(|(event, _)| event == "record")
        .filter_map(|(_, value)| {
            let record: polylogue_model::read::ReadRecord =
                serde_json::from_value(value.clone()).unwrap();
            match record.record() {
                Some(polylogue_model::bundle::Record::Session(session)) => {
                    Some(session.reference.clone())
                }
                _ => None,
            }
        })
        .collect();
    assert_eq!(
        sessions,
        ["session:claude-code:s2", "session:claude-code:s1"]
    );
}

#[test]
fn streamed_read_keeps_message_grain_in_a_fixed_reference_population() {
    let d = two_sessions();
    let events = socket_events(
        &d,
        "message-population",
        json!({
            "references":["message:claude-code:u2","message:claude-code:u1"],
            "steps":[{"verb":"tag","name":"message-only"},{"verb":"read","delivery":"stream","view":"transcript"}]
        }),
    );
    let mut messages = Vec::new();
    for (_, value) in events.iter().filter(|(event, _)| event == "record") {
        let record: polylogue_model::read::ReadRecord =
            serde_json::from_value(value.clone()).unwrap();
        match record.record() {
            Some(polylogue_model::bundle::Record::Session(_)) => {
                panic!("message selection became session grain")
            }
            Some(polylogue_model::bundle::Record::Message(message)) => {
                messages.push(message.reference.clone())
            }
            _ => {}
        }
    }
    assert_eq!(
        messages,
        ["message:claude-code:u2", "message:claude-code:u1"]
    );
}

#[test]
fn session_notes_are_visible_in_pages_and_hide_or_withdraw_removes_them() {
    use polylogue_model::read::{Card, ItemKind, ReadPage};
    let daemon = test_daemon::start("session note visibility");
    let written = steps(
        &daemon,
        "page-note",
        json!({"selection":"session:claude-code:s1",
        "steps":[{"verb":"note","text":"current session note"},{"verb":"read"}]}),
    );
    let page: ReadPage =
        serde_json::from_value(written[1]["outcome"]["reply"]["reply"].clone()).unwrap();
    let notes = |page: &ReadPage| {
        page.document
            .items
            .iter()
            .flat_map(|item| match &item.kind {
                ItemKind::Card { card } => match card.as_ref() {
                    Card::Session { annotations, .. } => annotations.clone(),
                    _ => vec![],
                },
                _ => vec![],
            })
            .collect::<Vec<_>>()
    };
    let shown = notes(&page);
    assert_eq!(shown.len(), 1);
    assert_eq!(shown[0].value, json!("current session note"));
    let hidden = steps(
        &daemon,
        "hide-note",
        json!({"selection":"session:claude-code:s1",
        "steps":[{"verb":"read","hide":["annotations"]}]}),
    );
    let hidden: ReadPage =
        serde_json::from_value(hidden[0]["outcome"]["reply"]["reply"].clone()).unwrap();
    assert!(notes(&hidden).is_empty());
    let withdrawn = steps(
        &daemon,
        "withdraw-note",
        json!({"selection":"annotation:page-note-1",
        "steps":[{"verb":"withdraw"}]}),
    );
    assert_eq!(withdrawn[0]["outcome"]["status"], "done");
    let read = steps(
        &daemon,
        "read-after-withdraw",
        json!({"selection":"session:claude-code:s1",
        "steps":[{"verb":"read"}]}),
    );
    let read: ReadPage =
        serde_json::from_value(read[0]["outcome"]["reply"]["reply"].clone()).unwrap();
    assert!(notes(&read).is_empty());
}

#[test]
fn line_deadline_still_applies_after_resolution_journal_handoff() {
    use polylogue_store::read::limit::ReadLimit;
    use std::time::{Duration, Instant};
    let d = test_daemon::start("whole line deadline");
    let blocker = std::fs::File::open(d.daemon.archive.archive_db()).unwrap();
    blocker.lock_shared().unwrap();
    let limit = ReadLimit::new(Duration::from_secs(2), Default::default());
    let request = Call::new(
        "rows",
        json!({
            "selection":"session:claude-code:s1", "steps":[
                {"verb":"note","text":"must not be written after the deadline"},
                {"verb":"read"}
            ]
        }),
    )
    .id("line-deadline");
    let (answer, handed_off) = std::thread::scope(|scope| {
        let worker = scope.spawn(|| request.within(&limit.clone()).answer(&d.daemon));
        let until = Instant::now() + Duration::from_secs(5);
        while limit.stage() != Some("write") && !worker.is_finished() && Instant::now() < until {
            std::thread::yield_now();
        }
        let handed_off = limit.stage() == Some("write");
        if handed_off {
            while limit.check().is_ok() {
                std::thread::yield_now();
            }
        }
        drop(blocker);
        (worker.join().unwrap(), handed_off)
    });
    assert!(
        handed_off,
        "line did not hand off its resolution journal: {answer:?}"
    );
    let answer = crate::test_answers::ok(answer);
    let reports = answer["reply"]["steps"].as_array().unwrap();
    assert_eq!(reports[0]["outcome"]["status"], "failed");
    assert_eq!(reports[0]["outcome"]["error"]["kind"], "timeout");
    assert_eq!(reports[1]["outcome"]["status"], "not_run");
    let reader = d.daemon.reads.reader().unwrap();
    assert!(reader.user_request("line-deadline").unwrap().is_some());
    assert!(reader.user_request("line-deadline-1").unwrap().is_none());
    drop(reader);
    assert_eq!(count(&d, "annotations"), 0);
}

#[test]
fn ordered_read_stream_preserves_degraded_read_outcome() {
    let d = test_daemon::start("readable session beside an incomplete source");
    let bytes = polylogue_testkit::claude_user("u-pending", "s-pending", "not parsed yet");
    let mut batch = polylogue_store::fixture::writer::step(
        "pending-completeness",
        "synthetic-unregistered",
        "/home/op/pending.jsonl",
        Default::default(),
        0,
        0,
    );
    batch.length = bytes.len() as u64;
    batch.parse.clear();
    let prepared = d.daemon.archive.prepare_bytes(bytes.as_bytes()).unwrap();
    d.daemon
        .writes
        .as_ref()
        .unwrap()
        .store_bytes(prepared, batch)
        .unwrap();
    let completeness = d
        .daemon
        .reads
        .reader()
        .unwrap()
        .completeness(&polylogue_engine::reading_contracts())
        .unwrap();
    assert!(
        !completeness.caught_up(),
        "captured source has no declared reading"
    );
    let finite = line(
        &d,
        "finite-degraded",
        json!({
            "selection":"session:claude-code:s1", "steps":[{"verb":"read"}]
        }),
    );
    let (_, notes) = crate::test_answers::degraded(finite);
    assert!(!notes.is_empty());
    let events = socket_events(
        &d,
        "stream-degraded",
        json!({
            "selection":"session:claude-code:s1", "steps":[
                {"verb":"read","delivery":"stream"},
                {"verb":"read","delivery":"stream"}
            ]
        }),
    );
    assert_eq!(
        events
            .iter()
            .filter(|(event, _)| event == "complete")
            .count(),
        2
    );
    let (event, final_answer) = events.last().unwrap();
    assert_eq!(event, "answer");
    let final_answer: command_surface::Answer<Value> =
        serde_json::from_value(final_answer.clone()).unwrap();
    let (_, streamed_notes) = crate::test_answers::degraded(final_answer);
    for note in notes {
        assert!(streamed_notes.contains(&note), "{streamed_notes:?}");
    }
}

fn listing_value(d: &TestDaemon, selection: &str) -> Value {
    let answer = line(
        d,
        &uuid::Uuid::now_v7().to_string(),
        json!({"selection":selection}),
    );
    let value = answer.value().unwrap_or_else(|| panic!("{answer:?}"));
    assert_eq!(value["kind"], "listing", "{value}");
    value["reply"].clone()
}

fn kept_result(d: &TestDaemon, report: &Value) -> Value {
    assert_eq!(report["outcome"]["status"], "done", "{report}");
    let result = report["outcome"]["reply"]["reply"]["kept"]["result"]
        .as_str()
        .unwrap();
    listing_value(d, result)
}

#[test]
fn empty_entity_population_stays_empty_across_two_saves_and_retry() {
    let mut d = test_daemon::start("fixed empty save population");
    let expected = listing_value(&d, "annotations");
    assert_eq!(expected["count"], 0);
    let request = json!({"selection":"annotations", "steps":[
        {"verb":"save","name":"first-empty"}, {"verb":"save","name":"second-empty"}
    ]});
    for restart in [false, true] {
        if restart {
            d.restart();
        }
        let reports = steps(&d, "fixed-empty-save", request.clone());
        for report in reports {
            let result = kept_result(&d, &report);
            for field in ["unit", "columns", "rows", "count"] {
                assert_eq!(result[field], expected[field], "{field}: {result}");
            }
        }
    }
    assert_eq!(count(&d, "annotations where kind:saved_query"), 2);
}

#[test]
fn keyed_save_hydrates_earlier_title_and_read_sees_earlier_annotations() {
    let d = test_daemon::start("original synthetic title");
    let reports = steps(
        &d,
        "keyed-current-save",
        json!({
            "selection":"sessions", "steps":[
                {"verb":"tag","name":"before-save"},
                {"verb":"note","as":"title","text":"Title from earlier step"},
                {"verb":"save","name":"current-entity"},
                {"verb":"read","show":["annotations"]}
            ]
        }),
    );
    let kept = kept_result(&d, &reports[2]);
    assert_eq!(
        kept["rows"][0]["title"], "Title from earlier step",
        "{kept}"
    );
    let read = reports[3]["outcome"]["reply"]["reply"].to_string();
    assert!(read.contains("before-save"), "{read}");
    assert!(read.contains("Title from earlier step"), "{read}");
}

#[test]
fn unkeyed_aggregate_is_unchanged_by_its_own_saves() {
    let d = test_daemon::start("aggregate save population");
    let query = "annotations | count";
    let expected = listing_value(&d, query);
    let reports = steps(
        &d,
        "aggregate-save",
        json!({"selection":query,"steps":[
            {"verb":"save","name":"first-count","denominator":"annotations"}, {"verb":"save","name":"second-count","denominator":"annotations"}
        ]}),
    );
    for report in reports {
        let kept = kept_result(&d, &report);
        assert_eq!(kept["rows"], expected["rows"], "{kept}");
        assert_eq!(kept["columns"], expected["columns"], "{kept}");
    }
    assert_eq!(count(&d, "annotations where kind:saved_query"), 2);
}

#[test]
fn projected_native_relation_is_fixed_while_a_later_read_is_current() {
    let d = test_daemon::start("original projected title");
    let query = "sessions | fields id, title";
    let expected = listing_value(&d, query);
    let reports = steps(
        &d,
        "projected-native-save",
        json!({"selection":query,"steps":[
            {"verb":"note","as":"title","text":"Current title after projection"},
            {"verb":"save","name":"fixed-projection"},
            {"verb":"read"}
        ]}),
    );
    let kept = kept_result(&d, &reports[1]);
    assert_eq!(kept["rows"], expected["rows"], "{kept}");
    assert!(
        reports[2]["outcome"]["reply"]["reply"]
            .to_string()
            .contains("Current title after projection"),
        "{:?}",
        reports[2]
    );
}

#[test]
fn saving_a_saved_kept_relation_retains_nested_output_schemas() {
    let mut d = test_daemon::start("Nested kept source title");
    let original = steps(
        &d,
        "kept-source",
        json!({"selection":"session:claude-code:s1","steps":[{"verb":"save"}]}),
    );
    let source = original[0]["outcome"]["reply"]["reply"]["kept"]["result"]
        .as_str()
        .unwrap();
    let expected = kept_result(&d, &original[0]);
    let named = steps(
        &d,
        "name-kept-source",
        json!({"selection":source,"steps":[{"verb":"save","name":"kept-source"}]}),
    );
    assert_eq!(kept_result(&d, &named[0])["rows"], expected["rows"]);
    let request =
        json!({"selection":"query:kept-source","steps":[{"verb":"save","name":"nested-kept"}]});
    let nested = steps(&d, "nested-kept-source", request.clone());
    assert_eq!(kept_result(&d, &nested[0])["rows"], expected["rows"]);
    d.restart();
    let repeated = steps(&d, "nested-kept-source", request);
    assert_eq!(kept_result(&d, &repeated[0])["rows"], expected["rows"]);
}

#[test]
fn kept_paged_measure_follows_every_page_with_its_original_population() {
    let d = test_daemon::start_with_sources_and_workers(
        "First measurement session",
        &|home| {
            for index in 0..30 {
                let id = format!("measure-{index}");
                std::fs::write(
                    home.join(format!(".claude/projects/-p/{id}.jsonl")),
                    polylogue_testkit::claude_user(
                        &format!("u-{index}"),
                        &id,
                        "Synthetic measured session",
                    ),
                )
                .unwrap();
            }
        },
        30,
        2,
    );
    let query = "sessions | measure compare:origin";
    let first = listing_value(&d, query);
    assert!(
        first["next"].is_string(),
        "fixture must exceed ordinary measure page size: {first}"
    );
    let request = json!({"selection":query,"steps":[{"verb":"save","name":"complete-measure","denominator":"sessions"}]});
    let reports = steps(&d, "complete-paged-measure", request.clone());
    let kept = kept_result(&d, &reports[0]);
    assert_eq!(kept["shape"], "summary", "{kept}");
    assert_eq!(kept["rows"].as_array().unwrap().len(), 1, "{kept}");
    assert_eq!(kept["rows"][0]["measure"], "compare:origin");
    let sessions = kept["rows"][0]["data"]["sessions"].as_array().unwrap();
    assert_eq!(sessions.len(), 31, "{kept}");
    let expected: std::collections::BTreeSet<_> = std::iter::once("claude-code:s1".to_owned())
        .chain((0..30).map(|index| format!("claude-code:measure-{index}")))
        .collect();
    let actual: std::collections::BTreeSet<_> = sessions
        .iter()
        .map(|session| session["session"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(actual, expected);
    let repeated = steps(&d, "complete-paged-measure", request);
    assert_eq!(kept_result(&d, &repeated[0])["rows"], kept["rows"]);
}

#[test]
fn requested_picker_population_retains_order_and_retry_after_prefix_changes_predicate() {
    let mut d = two_sessions();
    let request = json!({
        "selection":"sessions where NOT tag:prefix | sort by id",
        "all":true,"return_selection":true,
        "steps":[{"verb":"tag","name":"prefix"}]
    });
    for restart in [false, true] {
        if restart {
            d.restart();
        }
        let answer = crate::test_answers::ok(line(&d, "accepted-picker-prefix", request.clone()));
        let population = &answer["reply"]["resolved"]["population"]["value"];
        assert_eq!(
            population["references"],
            json!(["session:claude-code:s1", "session:claude-code:s2"]),
            "{answer}"
        );
        assert_eq!(population["rows"].as_array().unwrap().len(), 2, "{answer}");
        assert_eq!(population["count"], 2, "{answer}");
        assert_eq!(count(&d, "sessions where NOT tag:prefix"), 0);
        assert_eq!(count(&d, "annotations where kind:tag"), 2);
    }
    let conflict = refusal(line(
        &d,
        "accepted-picker-prefix",
        json!({
            "selection":"sessions","all":true,"return_selection":true,
            "steps":[{"verb":"tag","name":"prefix"}]
        }),
    ));
    assert_eq!(conflict.code.as_deref(), Some("conflict"));
}

#[test]
fn dry_run_picker_describes_steps_without_writing() {
    let d = two_sessions();
    let reports = steps(
        &d,
        "dry-picker",
        json!({"selection":"sessions","all":true,"dry_run":true,
        "steps":[{"verb":"tag","name":"prefix"},{"verb":"pick"},{"verb":"tag","name":"suffix"}]}),
    );
    assert_eq!(reports.len(), 3);
    assert!(
        reports[1]["outcome"]["would"]
            .as_str()
            .unwrap()
            .contains("picker"),
        "{reports:?}"
    );
    assert_eq!(count(&d, "annotations where kind:tag"), 0);
}

#[test]
fn newly_saved_names_cannot_be_captured_by_commands_or_verbs() {
    let d = test_daemon::start("saved name grammar");
    for name in [
        "status", "read", "sessions", "latest", "self", "here", "query", "result", "deadbeef",
    ] {
        let failure = crate::test_answers::refusal(line(
            &d,
            &format!("reserved-{name}"),
            json!({
                "selection":"sessions", "steps":[{"verb":"save","name":name}]
            }),
        ));
        assert_eq!(failure.code.as_deref(), Some("bad_value"));
        assert!(failure.message.contains(name), "{failure:?}");
    }
    assert_eq!(count(&d, "annotations where kind:saved_query"), 0);
    steps(
        &d,
        "ordinary-saved-name",
        json!({"selection":"sessions", "steps":[{"verb":"save","name":"grammar-control"}]}),
    );
    assert_eq!(count(&d, "query:grammar-control"), 1);
}

#[test]
fn a_save_retry_retains_original_child_revisions_and_undo_after_definition_changes() {
    let d = test_daemon::start("Save child identities");
    let save = |query: &str| json!({"selection":query,"steps":[{"verb":"save","name":"child-name","watch":true,"hold":"20m"}]});
    let first = steps(&d, "first-save", save("sessions"));
    let second = steps(&d, "second-save", save("sessions"));
    let part = |report: &Value| report["outcome"]["reply"]["reply"].clone();
    let first_parts = part(&first[0]);
    let second_parts = part(&second[0]);
    // A new Save keeps new rows but reuses an unchanged question and watch.
    for (field, created) in [("saved", "new_revision"), ("watch", "new_rule")] {
        assert_eq!(second_parts[field]["id"], first_parts[field]["id"]);
        assert_eq!(first_parts[field][created], true);
        assert_eq!(second_parts[field][created], false);
    }
    assert_eq!(second_parts["repeat"], false);
    assert!(
        second[0]["outcome"].get("undo").is_none(),
        "a keep-only Save has no undo"
    );
    assert_eq!(second_parts["kept"]["result"], "result:second-save-1");
    steps(&d, "changed-definition", save("messages"));
    for (id, original) in [("first-save", &first[0]), ("second-save", &second[0])] {
        let retry = steps(&d, id, save("sessions"));
        for field in ["saved", "watch", "kept"] {
            assert_eq!(part(&retry[0])[field], part(original)[field]);
        }
        assert_eq!(part(&retry[0])["repeat"], true);
        assert_eq!(retry[0]["outcome"]["undo"], original["outcome"]["undo"]);
    }
    let undo = first[0]["outcome"]["undo"].as_array().unwrap();
    assert_eq!(undo.len(), 1);
    let command = undo[0].as_str().unwrap();
    assert!(
        command.contains("annotation:first-save-1.1@first-save-1.1"),
        "{command}"
    );
    assert!(
        command.contains("annotation:first-save-1.2@first-save-1.2"),
        "{command}"
    );
    assert!(command.contains("first-save-1-undo"), "{command}");
}

#[test]
fn an_owned_save_finishes_atomically_after_deadline_then_stops_later_steps() {
    use polylogue_store::{read::limit::ReadLimit, writer::DispatchPhase};
    use std::time::Duration;
    let d = test_daemon::start("atomic Save ownership");
    let pause = d
        .daemon
        .writes
        .as_ref()
        .unwrap()
        .pause_user_dispatch_for_test("owned-save-1", DispatchPhase::Owned);
    let limit = ReadLimit::new(Duration::from_secs(2), Default::default());
    let request = Call::new(
        "rows",
        json!({"selection":"sessions","steps":[
            {"verb":"save","name":"owned-question","watch":true,"hold":"20m"},
            {"verb":"read"},{"verb":"tag","name":"not-after-deadline"}
        ]}),
    )
    .id("owned-save");
    let answer = std::thread::scope(|scope| {
        let worker = scope.spawn(|| request.within(&limit.clone()).answer(&d.daemon));
        pause.wait().expect("Save must reach writer ownership");
        while limit.check().is_ok() {
            std::thread::yield_now();
        }
        pause.release();
        worker.join().unwrap()
    });
    let value = crate::test_answers::ok(answer);
    let reports = value["reply"]["steps"].as_array().unwrap();
    assert_eq!(reports[0]["outcome"]["status"], "done", "{value}");
    assert_eq!(
        reports[0]["outcome"]["undo"].as_array().unwrap().len(),
        1,
        "{value}"
    );
    assert_eq!(reports[1]["outcome"]["error"]["kind"], "timeout", "{value}");
    assert_eq!(reports[2]["outcome"]["status"], "not_run", "{value}");
    let reader = d.daemon.reads.reader().unwrap();
    let accepted = reader.user_request("owned-save-1").unwrap().unwrap();
    let polylogue_model::user::UserWrite::Save(save) = accepted.write else {
        panic!("one atomic Save request")
    };
    assert_eq!(
        save.outcomes.as_ref().unwrap().kept.result,
        "result:owned-save-1"
    );
    for child in ["owned-save-1.1", "owned-save-1.2"] {
        assert!(
            reader.user_request(child).unwrap().is_none(),
            "one parent receipt: {child}"
        );
    }
    drop(reader);
    assert_eq!(count(&d, "annotations where kind:saved_query"), 1);
    assert_eq!(count(&d, "annotations where kind:signal"), 1);
    assert_eq!(count(&d, "annotations where kind:tag"), 0);
    assert_eq!(count(&d, "result:owned-save-1"), 1);
}

#[test]
fn unnamed_save_keeps_only_rows_and_named_only_options_explain_the_refusal() {
    let d = test_daemon::start("unnamed Save");
    let kept = steps(
        &d,
        "only-rows",
        json!({"selection":"sessions", "steps":[{"verb":"save"}]}),
    );
    let report = &kept[0]["outcome"]["reply"]["reply"];
    assert!(
        report["saved"].is_null() && report["watch"].is_null(),
        "{report}"
    );
    assert_eq!(report["kept"]["rows"], 1);
    assert_eq!(report["kept"]["result"], "result:only-rows-1");
    assert!(
        kept[0]["outcome"].get("undo").is_none(),
        "an unnamed Save has no undo"
    );
    assert_eq!(count(&d, "annotations where kind:(saved_query|signal)"), 0);
    let watch = crate::test_answers::refusal(line(
        &d,
        "nameless-watch",
        json!({
            "selection":"sessions where 'a quoted question'", "steps":[{"verb":"save","watch":true}]
        }),
    ));
    assert_eq!(watch.code.as_deref(), Some("bad_request"));
    let query = "sessions where 'a quoted question'";
    let suggested = crate::cli::line::shell_line(&[
        query.into(),
        "save".into(),
        "NAME".into(),
        "--watch".into(),
    ]);
    assert_eq!(watch.message, format!("a watch needs a name: {suggested}"));
    let figure = crate::test_answers::refusal(line(
        &d,
        "nameless-figure",
        json!({
            "selection":"sessions | count", "steps":[{"verb":"save","denominator":"sessions"}]
        }),
    ));
    assert_eq!(figure.code.as_deref(), Some("bad_request"));
    assert_eq!(figure.message, "a figure needs a saved query name");
    for (option, value) in [("hold", json!("20m")), ("notify", json!(true))] {
        let mut step = json!({"verb":"save","name":"refined"});
        step[option] = value;
        let error = crate::test_answers::refusal(line(
            &d,
            &format!("unwatched-{option}"),
            json!({
                "selection":"sessions", "steps":[step]
            }),
        ));
        assert_eq!(error.code.as_deref(), Some("bad_request"));
        assert_eq!(error.message, format!("save --{option} needs --watch"));
    }
    assert_eq!(count(&d, "results"), 1);
}

#[test]
fn save_keeps_the_accepted_membership_after_an_earlier_step_changes_the_predicate() {
    let d = test_daemon::start("accepted Save membership");
    let query = "sessions where NOT tag:handled";
    let before = listing_value(&d, query);
    assert_eq!(before["count"], 1);
    let reports = steps(
        &d,
        "membership",
        json!({"selection":query, "steps":[
            {"verb":"tag", "name":"handled"}, {"verb":"save", "name":"before-handled"}
        ]}),
    );
    assert_eq!(reports[0]["outcome"]["status"], "done");
    assert_eq!(
        count(&d, query),
        0,
        "the original predicate must now be false"
    );
    let kept = kept_result(&d, &reports[1]);
    assert_eq!(kept["count"], 1, "{kept}");
    assert_eq!(kept["rows"][0]["id"], "claude-code:s1", "{kept}");
    assert_eq!(reports[1]["outcome"]["reply"]["reply"]["kept"]["rows"], 1);
}

#[test]
fn run_references_refuse_row_verbs_and_offer_the_recorded_run_reading() {
    let d = test_daemon::start("run reference admission");
    let reference = "run:018f0000-0000-7000-8000-000000000011";
    for verb in [
        json!({"verb":"tag","name":"not-run-rows"}),
        json!({"verb":"save","name":"not-run-rows"}),
    ] {
        let error = crate::test_answers::refusal(line(
            &d,
            "run-refusal",
            json!({
                "selection":reference, "steps":[verb]
            }),
        ));
        assert_eq!(error.code.as_deref(), Some("bad_request"));
        assert_eq!(
            error.message,
            format!("a run holds no rows; use polylogue {reference} read")
        );
        let Some(command_surface::Suggestion::Request(next)) = error.suggestion() else {
            panic!("{error:?}");
        };
        assert_eq!(next.arguments["selection"], reference);
        assert_eq!(next.arguments["steps"].as_array().unwrap().len(), 1);
        assert_eq!(next.arguments["steps"][0]["verb"], "read");
    }
    let pending_read = crate::test_answers::refusal(line(
        &d,
        "run-read-pending",
        json!({
            "selection":reference, "steps":[{"verb":"read"}]
        }),
    ));
    assert_eq!(pending_read.code.as_deref(), Some("not_found"));
    assert_eq!(count(&d, "annotations"), 0);
    assert_eq!(count(&d, "results"), 0);
}
