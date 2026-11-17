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
    let answer = Call::new("query", json!({ "query": query })).answer(&d.daemon);
    let value = answer
        .value()
        .cloned()
        .unwrap_or_else(|| panic!("{answer:?}"));
    value["count"].as_u64().unwrap()
}

#[test]
fn rows_without_steps_lists_like_query() {
    let d = two_sessions();
    let listed = crate::test_answers::ok(line(
        &d,
        "list",
        json!({"selection": "sessions", "fields": ["id"], "page_size": 1}),
    ));
    let queried = crate::test_answers::ok(
        Call::new(
            "query",
            json!({"query": "sessions", "fields": ["id"], "limit": 1}),
        )
        .answer(&d.daemon),
    );
    assert_eq!(listed["kind"], "listing");
    assert_eq!(listed["reply"]["rows"], queried["rows"]);
    assert_eq!(listed["reply"]["rows"].as_array().unwrap().len(), 1);
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
        json!(["polylogue assertion:line-1 withdraw"])
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
        "2 rows match; a write or act on all of them needs --all"
    );
    assert_eq!(refused.candidates().len(), 2);
    assert_eq!(count(&d, "sessions where tag:later"), 0);
    let steps = steps(&d, "with", tag(true));
    assert_eq!(steps[0]["outcome"]["reply"]["reply"]["targets"], 2);
    assert_eq!(count(&d, "sessions where tag:later"), 2);
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
    assert_eq!(count(&d, "assertions where kind:note"), 0);
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
    assert_eq!(count(&d, "assertions where kind:note"), 1);
}

#[test]
fn daemon_refuses_row_numbers() {
    let d = test_daemon::start("first prompt");
    for selection in ["@1", "@1.2", "@2-5", "@1,4", "@all", "@last"] {
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
        .surface(crate::commands::Surface::Mcp)
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
            .surface(crate::commands::Surface::Mcp)
            .answer(&d.daemon),
    );
    assert_eq!(read["reply"]["steps"][0]["outcome"]["status"], "done");
}

#[test]
fn listing_wait_returns_when_rows_match() {
    let d = test_daemon::start("first prompt");
    let reads = d.watch_reads();
    std::thread::scope(|scope| {
        let waiting = scope.spawn(|| {
            line(
                &d,
                "wait",
                json!({"selection": "sessions where tag:ready", "wait": "1m"}),
            )
        });
        let read = reads
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        while read.limit.stage() != Some("listing wait") {
            read.limit.check().unwrap();
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

#[test]
fn mcp_wait_returns_empty_with_same_call_after_cap() {
    let d = test_daemon::start("first prompt");
    let request = command_surface::RunnableRequest {
        operation: "rows".into(),
        arguments: json!({"selection": "sessions where tag:later", "wait": "1h", "fields": [], "all": false}),
        request_id: "wait-cap".into(),
        deadline: command_surface::Deadline::None,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: true,
        format: Default::default(),
    };
    let call = Call::from_request(&request);
    let invocation = d
        .daemon
        .invocation(&call.authorship)
        .with_accepted_request(Some(std::sync::Arc::new(request.clone())));
    let answer = call
        .surface(crate::commands::Surface::Mcp)
        .limit(polylogue_store::read::limit::ReadLimit::without_deadline(
            Default::default(),
        ))
        .answer_with(&d.daemon, &invocation);
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
    .limit(polylogue_store::read::limit::ReadLimit::new(
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
    crate::client::stream::run_call(
        &d.daemon.archive,
        &request,
        Some(std::time::Duration::from_secs(10)),
        |frame| {
            let done = frame.event == "answer";
            events.push((frame.event, serde_json::from_str(&frame.data).unwrap()));
            Ok(done)
        },
    )
    .unwrap();
    events
}

#[test]
fn socket_chain_streams_each_read_at_its_step_and_retries_writes_once() {
    let d = test_daemon::start("ordered stream prompt");
    let arguments = json!({
        "selection": "sessions where NOT tag:done",
        "steps": [
            {"verb":"tag", "name":"done"},
            {"verb":"read", "view":"raw", "delivery":"stream"},
            {"verb":"note", "text":"after-note"},
            {"verb":"read", "view":"raw", "delivery":"stream"}
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
        if !retry {
            assert!(!reads[0].contains("after-note"), "{reads:?}");
            assert!(reads[1].contains("after-note"), "{reads:?}");
        }
        assert_eq!(count(&d, "assertions where kind:tag OR kind:note"), 2);
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
        json!(["polylogue assertion:prefix-1 withdraw"])
    );
    assert_eq!(count(&d, "assertions where kind:note"), 1);
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
    assert_eq!(count(&d, "assertions"), 0);
}
