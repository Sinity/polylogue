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
        "steps": [{"verb": "tag", "name": "later"}],
    });
    let first = steps(&d, "again", rows.clone());
    let second = steps(&d, "again", rows);
    assert_eq!(first[0]["outcome"]["reply"]["reply"]["repeat"], false);
    assert_eq!(second[0]["outcome"]["reply"]["reply"]["repeat"], true);
    assert_eq!(count(&d, "assertions where kind:tag"), 1);
}
