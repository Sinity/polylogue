//! Query plans of the default pages and of conditions on indexed columns,
//! read with `EXPLAIN QUERY PLAN` against the production schema.

use super::*;
use crate::fixture::CallFacts;
use polylogue_model::live::{LiveFacts, OpenCall};

fn plan(reader: &crate::read::Reader<'_>, sql: &str) -> Vec<String> {
    reader
        .query(&format!("EXPLAIN QUERY PLAN {sql}"), &[])
        .unwrap()
        .into_iter()
        .map(|row| row["detail"].as_str().unwrap().to_owned())
        .collect()
}

fn page_plan(f: &Fixture, text: &str) -> Vec<String> {
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let program = crate::query::compile(text, &options()).unwrap();
    plan(
        &reader,
        &program.statements.page(
            &program.order,
            program.params.len(),
            false,
            false,
            crate::query::PageColumns::All,
        ),
    )
}

fn ids(f: &Fixture, text: &str, field: &str) -> Vec<String> {
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut options = options();
    options.page_size = None;
    options.fields = vec![field.into()];
    let selected = reader.prepare_selection(text, &options).unwrap();
    let mut ids = Vec::new();
    reader
        .visit_selection::<SelectionError>(&selected, None, None, |row| {
            ids.push(row.values[field].as_str().unwrap_or_default().to_owned());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    ids
}

/// A default page of messages, tool calls or events reads its unit's rows
/// in index order and stops after the page, instead of sorting every row.
#[test]
fn default_page_plan_has_no_temp_btree() {
    let (_dir, f) = fixture();
    for (text, index) in [
        ("messages", "message_order"),
        ("tool_calls", "tool_call_order"),
        ("events", "event_order"),
    ] {
        let steps = page_plan(&f, text);
        assert!(
            steps
                .iter()
                .any(|step| step.contains(&format!("USING INDEX {index}"))),
            "{text}: {steps:#?}"
        );
        assert!(
            !steps
                .iter()
                .any(|step| step == "USE TEMP B-TREE FOR ORDER BY"),
            "{text}: {steps:#?}"
        );
    }
}

/// Reading in index order keeps the documented order: dated rows oldest
/// first, rows without a time after them.
#[test]
fn undated_messages_follow_dated_ones() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let s = f
        .session(SessionFacts {
            id: "test:order",
            last: Some(3),
            ..Default::default()
        })
        .unwrap();
    for (id, at) in [
        ("test:undated", None),
        ("test:late", Some(2_000)),
        ("test:early", Some(1_000)),
    ] {
        f.message(
            s,
            MessageFacts {
                id,
                role: "user",
                written_by: "human",
                at,
                ..Default::default()
            },
        )
        .unwrap();
    }
    assert_eq!(
        ids(&f, "messages", "id"),
        ["test:early", "test:late", "test:undated"]
    );
}

/// `outcome:` compares the stored column, so with `action:` the
/// `tool_call_action` index answers both conditions.
#[test]
fn outcome_condition_uses_tool_call_action_index() {
    let (_dir, f) = fixture();
    let steps = page_plan(&f, "tool_calls where action:shell outcome:error");
    assert!(
        steps
            .iter()
            .any(|step| step.contains("INDEX tool_call_action (action=? AND <expr>=?)")),
        "{steps:#?}"
    );
}

/// `pending` is the stored `unknown` call the session's live state names as
/// open; `unknown` leaves it out; stored outcomes match as stored.
#[test]
fn pending_outcome_matches_open_live_call() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let s = f
        .session(SessionFacts {
            id: "test:calls",
            last: Some(3),
            ..Default::default()
        })
        .unwrap();
    let mut calls = vec![];
    for (id, outcome) in [
        ("test:open", ""),
        ("test:lost", ""),
        ("test:failed", "error"),
        ("test:done", "ok"),
    ] {
        let m = f
            .message(
                s,
                MessageFacts {
                    id,
                    role: "assistant",
                    written_by: "model",
                    ..Default::default()
                },
            )
            .unwrap();
        f.tool_call(
            s,
            m,
            CallFacts {
                tool: "Bash",
                action: "shell",
                outcome,
                output: (!outcome.is_empty()).then_some("synthetic output"),
                ..Default::default()
            },
        )
        .unwrap();
        calls.push(m);
    }
    // The fixture names each call `c{message_pk}-{idx}`.
    f.live_facts(
        s,
        &LiveFacts {
            open_call: Some(OpenCall {
                tool: "Bash".into(),
                call_id: Some(format!("c{}-0", calls[0])),
                at: 1,
            }),
            ..Default::default()
        },
    )
    .unwrap();
    let messages = |text: &str| {
        let mut found = ids(&f, text, "message");
        found.sort();
        found
    };
    assert_eq!(messages("tool_calls where outcome:pending"), ["test:open"]);
    assert_eq!(messages("tool_calls where outcome:unknown"), ["test:lost"]);
    assert_eq!(messages("tool_calls where outcome:Error"), ["test:failed"]);
    assert_eq!(
        messages("tool_calls where outcome:ok OR outcome:pending"),
        ["test:done", "test:open"]
    );
    assert_eq!(
        messages("tool_calls where NOT outcome:unknown"),
        ["test:done", "test:failed", "test:open"]
    );
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut options = options();
    options.page_size = None;
    options.fields = vec!["message".into(), "outcome".into()];
    let selected = reader
        .prepare_selection("tool_calls where message:test:open", &options)
        .unwrap();
    reader
        .visit_selection::<SelectionError>(&selected, None, None, |row| {
            assert_eq!(row.values["outcome"], "pending");
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
}

/// Parents are found through the `link.child_id` index, and removing a
/// session's links reads both link indexes rather than the whole table.
#[test]
fn parent_links_use_index() {
    let (_dir, f) = fixture();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    for sql in [
        "SELECT parent_id FROM link WHERE child_id = ?1 AND kind = 'continuation' AND how IN ('fork', 'resume')",
        "SELECT pk FROM link WHERE parent_id = ?1 OR child_id = ?1",
    ] {
        let steps = plan(&reader, sql);
        assert!(
            steps
                .iter()
                .any(|step| step.contains("USING INDEX link_child_id")),
            "{sql}: {steps:#?}"
        );
        assert!(
            !steps.iter().any(|step| step == "SCAN link"),
            "{sql}: {steps:#?}"
        );
    }
}

/// `model:` keeps reading `message_model_at` now that `message_model` is gone.
#[test]
fn model_filter_plans_unchanged() {
    let (_dir, f) = fixture();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let steps = plan(
        &reader,
        "SELECT pk FROM message WHERE model = ?1 ORDER BY at, id",
    );
    assert!(
        steps
            .iter()
            .any(|step| step.contains("INDEX message_model_at")),
        "{steps:#?}"
    );
}

/// An attachment reads each page parent's children through the parent's
/// key, however many children other sessions have.
#[test]
fn attachment_reads_children_through_each_parent_key() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    for (n, id) in ["test:big", "test:small"].into_iter().enumerate() {
        let session = f
            .session(SessionFacts {
                id,
                last: Some(1_000 + n as i64),
                ..Default::default()
            })
            .unwrap();
        let count = if n == 0 { 3_000 } else { 3 };
        for m in 0..count {
            let message = format!("{id}:m{m}");
            f.message(
                session,
                MessageFacts {
                    id: &message,
                    role: if m % 2 == 0 { "user" } else { "assistant" },
                    written_by: if m % 2 == 0 { "human" } else { "model" },
                    at: Some(m as i64),
                    ..Default::default()
                },
            )
            .unwrap();
        }
    }
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let program = crate::query::compile(
        "sessions with messages(id, role)[role:assistant, last:1]",
        &options(),
    )
    .unwrap();
    let steps = plan(&reader, &program.attach[0].sql);
    let parents = steps
        .iter()
        .position(|step| step.starts_with("SCAN _parents"));
    let children = steps
        .iter()
        .position(|step| step.starts_with("SEARCH m USING INDEX") && step.contains("session_pk=?"));
    assert!(
        matches!((parents, children), (Some(p), Some(c)) if p < c),
        "{steps:#?}"
    );
    assert!(
        !steps
            .iter()
            .any(|step| step == "SCAN m" || step.starts_with("SCAN m ")),
        "{steps:#?}"
    );
}

/// A grouping with the `first N` or `agg …` after it reads the rows
/// entering the grouping once: the unit's rows are scanned once however
/// many such stages follow.
#[test]
fn per_group_stages_read_the_rows_entering_them_once() {
    let (_dir, f) = fixture();
    for (text, alias) in [
        ("sessions | group by origin | first 1", "s"),
        ("sessions | group by origin | count", "s"),
        (
            "sessions | group by origin | agg count(), min(started)",
            "s",
        ),
        (
            "sessions | group by origin | first 2 | group by repo | first 1 | group by origin | count",
            "s",
        ),
        ("messages | group by role | first 1", "m"),
    ] {
        let steps = page_plan(&f, text);
        let scans = steps
            .iter()
            .filter(|step| {
                step.strip_prefix("SCAN ")
                    .is_some_and(|rest| rest == alias || rest.starts_with(&format!("{alias} ")))
            })
            .count();
        assert_eq!(scans, 1, "{text}: {steps:#?}");
    }
}
