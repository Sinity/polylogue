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
            None,
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
        .visit_selection::<SelectionError>(&selected, crate::read::VisitFrom::First, None, |row| {
            ids.push(row.values[field].as_str().unwrap_or_default().to_owned());
            Ok(ControlFlow::Continue(()))
        })
        .unwrap();
    ids
}

/// The steps of a default page's plan outside correlated subqueries: what
/// the page itself reads, not the per-row values it computes.
fn page_steps(f: &Fixture, text: &str) -> Vec<String> {
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let program = crate::query::compile(text, &options()).unwrap();
    let sql = program.statements.page(
        &program.order,
        program.params.len(),
        None,
        false,
        crate::query::PageColumns::All,
    );
    let rows = reader
        .query(&format!("EXPLAIN QUERY PLAN {sql}"), &[])
        .unwrap();
    let mut inner = std::collections::HashSet::new();
    let mut steps = Vec::new();
    for row in rows {
        let (id, parent) = (row["id"].as_i64().unwrap(), row["parent"].as_i64().unwrap());
        let detail = row["detail"].as_str().unwrap().to_owned();
        if inner.contains(&parent) || detail.starts_with("CORRELATED SCALAR SUBQUERY") {
            inner.insert(id);
            continue;
        }
        steps.push(detail);
    }
    steps
}

/// The default page of every unit reads its rows in the order of an index
/// and stops after the page: no step sorts the whole result, and no step
/// reads every message, event or tool call (docs/query.md, paging). A sort
/// of the last terms orders only rows that share the index's leading terms
/// as the index is walked, so it is allowed. What a correlated subquery does
/// for one row is the cost of a column, not of the page.
#[test]
fn default_pages_read_in_index_order() {
    let (_dir, f) = fixture();
    for unit in polylogue_query::units::Unit::all() {
        let text = unit.name();
        let Ok(program) = crate::query::compile(text, &options()) else {
            continue;
        };
        // A derived relation, and a unit whose rows are an aggregate
        // (`files`), is computed whole before its page.
        if !matches!(
            program.statements,
            crate::query::compile::Statements::Keyed(_)
        ) || unit == polylogue_query::units::Unit::Files
        {
            continue;
        }
        let steps = page_steps(&f, text);
        let mut undated = false;
        for step in &steps {
            undated |= step == "UNION ALL";
            // Tool calls that share a time, the undated ones included, are
            // ordered by their message ID and block position, which the
            // time index cannot hold.
            let shared_time = unit == polylogue_query::units::Unit::ToolCalls && undated;
            assert!(
                !step.starts_with("USE TEMP B-TREE") || step.contains(" LAST ") || shared_time,
                "{text}: {steps:#?}"
            );
            for table in ["message", "event", "tool_call"] {
                // A scan of a whole table reads every row; a scan in an
                // index's order stops after the page.
                assert!(
                    !(step.starts_with("SCAN ") && step.ends_with(&format!(" {table}")))
                        && step != &format!("SCAN {table}"),
                    "{text}: {steps:#?}"
                );
            }
        }
    }
    for (text, index) in [
        ("messages", "message_time"),
        ("events", "event_time"),
        ("sessions", "session_activity"),
    ] {
        let steps = page_plan(&f, text);
        assert!(
            steps
                .iter()
                .any(|step| step.contains(&format!("INDEX {index}"))),
            "{text}: {steps:#?}"
        );
    }
}

/// A time bound compiles to comparisons of the time's columns, so its page
/// seeks a range of the time index and reads no stored time bytes.
#[test]
fn time_bounds_seek_the_time_index() {
    let (_dir, f) = fixture();
    for (text, index) in [
        ("messages where since:2026-01-01", "message_time (at>?)"),
        (
            "messages where until:2026-01-01",
            "message_time (at>? AND at<?)",
        ),
        ("events where since:2026-01-01", "event_time (at>?)"),
    ] {
        let steps = page_plan(&f, text);
        assert!(
            steps.iter().any(|step| step.contains(index)),
            "{text}: {steps:#?}"
        );
    }
    let program = crate::query::compile(
        "messages where since:2026-01-01 until:2026-02-01",
        &options(),
    )
    .unwrap();
    let crate::query::compile::Statements::Keyed(keyed) = &program.statements else {
        panic!("messages are read by key");
    };
    for function in ["pl_time", "at_time", "json_extract", " OR "] {
        assert!(!keyed.keys.contains(function), "{function}: {}", keyed.keys);
    }
}

/// A time condition holds when the whole period of a row's time meets it,
/// fails when none does and is otherwise unknown, which neither it nor its
/// negation matches; `since:` and `until:` select the times that overlap
/// the window (docs/query.md, times).
#[test]
fn time_conditions_test_whole_periods() {
    use polylogue_model::{Precision, TimeAccuracy, TimeQuality, Timestamp};
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let s = f
        .session(SessionFacts {
            id: "test:periods",
            last: Some(3_000),
            ..Default::default()
        })
        .unwrap();
    let ms = |ms| Timestamp::from_unix_ms(ms).unwrap();
    for (id, time) in [
        (
            "test:instant",
            Some((1_000, TimeAccuracy::intrinsic(Precision::Instant))),
        ),
        (
            "test:period",
            Some((
                500,
                TimeAccuracy {
                    quality: TimeQuality::Intrinsic,
                    precision: Some(Precision::Bounded),
                    until: Some(ms(2_500)),
                },
            )),
        ),
        (
            "test:second",
            Some((3_000, TimeAccuracy::intrinsic(Precision::Second))),
        ),
        ("test:undated", None),
    ] {
        let m = f
            .message(
                s,
                MessageFacts {
                    id,
                    role: "user",
                    written_by: "human",
                    ..Default::default()
                },
            )
            .unwrap();
        if let Some((at, accuracy)) = time {
            f.message_time(m, Some(ms(at)), accuracy).unwrap();
        }
    }
    let t = |s: u32| format!("1970-01-01T00:00:0{s}Z");
    for (condition, expected) in [
        (
            format!("since:{}", t(2)),
            &["test:period", "test:second"][..],
        ),
        // `until:` a second ends with that second.
        (
            format!("until:{}", t(1)),
            &["test:period", "test:instant"][..],
        ),
        (format!("until:{}", t(0)), &["test:period"][..]),
        (
            format!("at >= {}", t(1)),
            &["test:instant", "test:second"][..],
        ),
        (format!("NOT at >= {}", t(1)), &[][..]),
        (
            format!("at < {}", t(3)),
            &["test:period", "test:instant"][..],
        ),
        (format!("NOT at < {}", t(3)), &["test:second"][..]),
        (
            format!("at < {} OR at >= {}", t(1), t(3)),
            &["test:second"][..],
        ),
        (
            "at:*".to_owned(),
            &["test:period", "test:instant", "test:second", "test:undated"][..],
        ),
    ] {
        assert_eq!(
            ids(&f, &format!("messages where {condition}"), "id"),
            expected,
            "{condition}"
        );
    }
    // The session's latest activity is the latest period's: [3, 4) s.
    for (condition, expected) in [
        (format!("since:{}", t(3)), &["test:periods"][..]),
        (format!("last_active >= {}", t(3)), &["test:periods"][..]),
        (format!("last_active > {}", t(3)), &[][..]),
    ] {
        assert_eq!(
            ids(&f, &format!("sessions where {condition}"), "id"),
            expected,
            "{condition}"
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

#[test]
fn named_session_drives_a_page_of_shell_calls() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    for (id, calls) in [
        (
            "test:target",
            &[
                ("shell", "cargo test", 1),
                ("shell", "Cargo Test", 2),
                ("shell", "cargo build", 3),
                ("file_read", "cargo test", 4),
            ][..],
        ),
        ("test:target-other", &[("shell", "cargo test", 5)][..]),
    ] {
        let session = fixture
            .session(SessionFacts {
                id,
                ..Default::default()
            })
            .unwrap();
        let message = fixture
            .message(
                session,
                MessageFacts {
                    id: &format!("{id}:m"),
                    role: "assistant",
                    written_by: "model",
                    ..Default::default()
                },
            )
            .unwrap();
        for &(action, command, at) in calls {
            fixture
                .tool_call(
                    session,
                    message,
                    CallFacts {
                        tool: "Bash",
                        action,
                        command: Some(command),
                        outcome: "ok",
                        at: Some(at),
                        ..Default::default()
                    },
                )
                .unwrap();
        }
    }
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut options = options();
    options.page_size = Some(100);
    let program = crate::query::compile(
        "tool_calls where session:test:target action:shell command_norm:\"cargo test\" \
         | fields command,command_norm",
        &options,
    )
    .unwrap();
    let sql = program.statements.page(
        &program.order,
        program.params.len(),
        None,
        false,
        crate::query::PageColumns::All,
    );
    let mut params = crate::selection::values(&program.params);
    params.push(crate::SqlValue::Int(101));
    let plan = reader
        .query(&format!("EXPLAIN QUERY PLAN {sql}"), &params)
        .unwrap();
    assert!(
        plan.iter().any(|step| step["detail"]
            .as_str()
            .is_some_and(|detail| detail.contains("INDEX tool_call_session (session_pk=?"))),
        "{plan:#?}"
    );

    let page = reader.query_page(&program, None).unwrap();
    assert!(!page.more);
    let commands: Vec<_> = page
        .rows
        .into_iter()
        .map(|mut row| {
            crate::query::execute::present(&program, &mut row).unwrap();
            (row["command"].clone(), row["command_norm"].clone())
        })
        .collect();
    assert_eq!(
        commands,
        [
            (
                serde_json::json!("cargo test"),
                serde_json::json!("cargo test")
            ),
            (
                serde_json::json!("Cargo Test"),
                serde_json::json!("Cargo Test")
            ),
        ]
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
        .visit_selection::<SelectionError>(&selected, crate::read::VisitFrom::First, None, |row| {
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
        "SELECT parent_id FROM session_link WHERE child_id = ?1 AND kind <> 'reference' AND (kind = 'fork' OR copies_prefix = 1)",
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
    let steps = plan(
        &reader,
        "SELECT pk FROM link WHERE parent_id = ?1 OR child_id = ?1",
    );
    assert!(
        steps
            .iter()
            .any(|step| step.contains("link_parent") || step.contains("link_statement")),
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
