//! Sequences follow session order, not provider time.

use super::*;

/// A session whose provider times run backwards: two failing and
/// succeeding `cargo test` pairs written in that order, each call stamped
/// earlier than the one before it.
fn backwards() -> TestArchive {
    let dir = tempfile::tempdir().unwrap();
    let mut fx = Fixture::new(&dir.path().join("archive")).unwrap();
    let s = fx
        .session(SessionFacts {
            id: "claude-code:backwards",
            started: at(1, 9),
            last: at(1, 18),
            ..SessionFacts::default()
        })
        .unwrap();
    let calls = [
        ("cargo test", "error", 101, 18),
        ("src/date.rs", "ok", 0, 17),
        ("cargo test", "ok", 0, 16),
        ("cargo test", "error", 101, 15),
        ("cargo test", "ok", 0, 14),
    ];
    for (n, (what, outcome, exit_code, hour)) in calls.into_iter().enumerate() {
        let m = fx
            .message(
                s,
                msg(
                    &scoped("claude-code:backwards", &format!("b{n}")),
                    "assistant",
                    "model",
                    "",
                    at(1, hour),
                ),
            )
            .unwrap();
        let edit = what.starts_with("src/");
        fx.tool_call(
            s,
            m,
            CallFacts {
                tool: if edit { "Edit" } else { "Bash" },
                action: if edit { "file_edit" } else { "shell" },
                command: (!edit).then_some(what),
                path: edit.then_some(what),
                outcome,
                exit_code: Some(exit_code),
                at: at(1, hour),
                output: Some("done"),
            },
        )
        .unwrap();
    }
    TestArchive { _dir: dir, fx }
}

/// `seq(...)` and "a later success of the same command" match by order in
/// the session, and their match rows are listed in that order, however
/// the provider's times run.
#[test]
fn sequences_follow_session_order_when_times_run_backwards() {
    let w = backwards();
    let fail_then_pass = "sessions where seq(command:\"cargo test\" AND outcome:error -> command:\"cargo test\" AND outcome:ok)";
    assert_eq!(ids(&q(&w, fail_then_pass)), ["claude-code:backwards"]);
    let v = q(&w, &format!("{fail_then_pass} | matches"));
    let rows = v["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 4, "{v}");
    let order: Vec<(u64, u64, &str)> = rows
        .iter()
        .map(|r| {
            (
                r["match"].as_u64().unwrap(),
                r["position"].as_u64().unwrap(),
                r["step"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        order,
        [
            (1, 1, "block:claude-code:b0#1"),
            (1, 2, "block:claude-code:b2#1"),
            (2, 1, "block:claude-code:b3#1"),
            (2, 2, "block:claude-code:b4#1"),
        ]
    );
    let time = |row: &Value| {
        row["step.at"]
            .as_str()
            .unwrap()
            .parse::<jiff::Timestamp>()
            .unwrap()
            .as_millisecond()
    };
    assert_eq!(time(&rows[1]) - time(&rows[0]), -2 * 3_600_000);
    // In session order a success never comes before the first failure.
    let pass_then_fail = "sessions where seq(command:\"cargo test\" AND outcome:ok -> [next] command:\"cargo test\" AND outcome:error)";
    let v = q(&w, &format!("{pass_then_fail} | matches"));
    let rows = v["rows"].as_array().unwrap();
    assert_eq!(rows.len(), 2, "{v}");
    assert_eq!(rows[0]["step"], "block:claude-code:b2#1");
    assert_eq!(rows[1]["step"], "block:claude-code:b3#1");
    // A time window needs the later call to be later in time too.
    let within = "sessions where seq(command:\"cargo test\" AND outcome:error -> [within 5h] command:\"cargo test\" AND outcome:ok)";
    assert_eq!(q(&w, within)["outcome"], "empty");
}

#[test]
fn matches_keep_only_entering_parent_columns_with_aliases_and_metadata() {
    let archive = backwards();
    let source = q(&archive, "sessions");
    let started = &source["rows"][0]["started"];
    for predicate in [
        "seq(command:\"cargo test\" AND outcome:error -> command:\"cargo test\" AND outcome:ok)",
        "exists tool_call(outcome:error)",
    ] {
        let text =
            format!("sessions where {predicate} | fields id as source, started as began | matches");
        let matched = q(&archive, &text);
        let rows = matched["rows"].as_array().unwrap();
        assert!(!rows.is_empty(), "{matched}");
        for row in rows {
            assert_eq!(row["parent.source"], "claude-code:backwards");
            assert_eq!(&row["parent.began"], started);
            assert!(row.get("parent.title").is_none(), "{matched}");
            assert!(row.get("parent.id").is_none(), "{matched}");
        }
        let columns = matched["columns"].as_array().unwrap();
        assert!(columns.iter().any(|column| column["name"] == "parent.source"
            && column["reference"] == "session"), "{matched}");
        assert!(
            columns
                .iter()
                .any(|column| column["name"] == "parent.began" && column["kind"] == "time"),
            "{matched}"
        );
        let projected = q(
            &archive,
            &format!("{text} | fields parent.source, parent.began"),
        );
        assert_eq!(projected["count"], matched["count"]);
        for row in projected["rows"].as_array().unwrap() {
            assert_eq!(row["parent.source"], "claude-code:backwards");
            assert_eq!(&row["parent.began"], started);
        }
    }
}
