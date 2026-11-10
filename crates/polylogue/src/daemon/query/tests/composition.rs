//! Membership and selection across query stages.

use super::*;
use std::collections::BTreeSet;

fn row_ids(value: &Value) -> BTreeSet<String> {
    value["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_str().unwrap().to_owned())
        .collect()
}

#[test]
fn in_ignores_unknown_group_values_and_keeps_unknown_left_values_unknown() {
    let w = morrow();
    let set = "sessions where origin:codex | group by cwd";
    // Both Codex sessions have an unknown cwd. Known cwd values are not members.
    assert_eq!(
        q(&w, &format!("sessions where kind:* cwd in ({set})"))["count"],
        0
    );
    assert_eq!(
        ids(&q(&w, &format!("sessions where kind:* NOT cwd in ({set})"))),
        ["#4", "#2", "#1"]
    );
    let empty = "sessions where origin:codex repo:absent | group by cwd";
    assert_eq!(
        ids(&q(
            &w,
            &format!("sessions where kind:* NOT cwd in ({empty})")
        )),
        ["#4", "#2", "#1"]
    );
}

#[test]
fn a_scoped_query_keeps_the_session_selection_and_discards_its_ranking() {
    let w = morrow();
    let expected = q(&w, "messages where session:claude-code:7f3a2c10-0004");
    for query in [
        "origin:claude-code | limit 1 | messages",
        "origin:claude-code | union (origin:codex) | sort by id | limit 1 | messages",
        "origin:claude-code | fuse (origin:claude-code) | sort by last_active desc | limit 1 | messages",
    ] {
        let answer = q(&w, query);
        assert_eq!(answer["ranked"], false, "{query}");
        let selected = if query.contains("sort by id") {
            q(&w, "messages where session:claude-code:7f3a2c10-0001")
        } else {
            expected.clone()
        };
        assert_eq!(row_ids(&answer), row_ids(&selected), "{query}");
    }
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(32))]

    #[test]
    fn equivalent_timestamp_offsets_select_the_same_instant(millis in 0_i64..1000, offset in -12_i32..12) {
        let dir = tempfile::tempdir().unwrap();
        let mut fx = Fixture::new(&dir.path().join("archive")).unwrap();
        let session = fx.session(SessionFacts { id: "claude-code:time", ..SessionFacts::default() }).unwrap();
        let instant = OCT1 + DAY + 30 * 60_000 + 45_000 + millis;
        for (name, time) in [("before", Some(instant - 1)), ("equal", Some(instant)), ("after", Some(instant + 1)), ("unknown", None)] {
            fx.message(session, msg(name, "user", "human", "synthetic", time)).unwrap();
        }
        let w = TestArchive { _dir: dir, fx };
        let utc = format!("2026-10-02T12:30:45.{millis:03}Z");
        let sign = if offset < 0 { '-' } else { '+' };
        let local = format!("2026-10-02T{:02}:30:45.{millis:03}{sign}{:02}:00", 12 + offset, offset.abs());
        for (operator, names) in [("=", vec!["equal"]), ("<", vec!["before"]), (">", vec!["after"]), ("!=", vec!["before", "after"])] {
            let expected = names.into_iter().map(str::to_owned).collect::<BTreeSet<_>>();
            for time in [&utc, &local] {
                let query = format!("messages where at {operator} {time}");
                proptest::prop_assert_eq!(row_ids(&q(&w, &query)), expected.clone(), "{}", query);
            }
        }
    }
}

#[test]
fn membership_refuses_group_values_of_another_type() {
    let w = morrow();
    let query = "sessions where cwd in (sessions | group by cost)";
    let error = run_q(&w, ask(query)).unwrap_err();
    assert_eq!(error.code(), "invalid_query");
    assert!(error.message.contains("same type"), "{}", error.message);
}

#[test]
fn saved_relations_bind_defaults_before_the_caller_narrows_them() {
    let mut w = morrow();
    for (name, text) in [
        ("children", "sessions where kind:subagent"),
        ("child-ref", "session:claude-code:7f3a2c10-0003"),
        ("abandoned", "messages where active:false"),
    ] {
        save_definition(&mut w, name, text);
    }
    for query in [
        "sessions where query:children",
        "sessions where query:children origin:claude-code",
        "sessions where query:child-ref",
    ] {
        assert_eq!(ids(&q(&w, query)), ["#3"], "{query}");
    }
    assert_eq!(
        q(&w, "messages where query:abandoned")["count"],
        q(&w, "messages where active:false")["count"]
    );
    assert_eq!(
        ids(&q(
            &w,
            "sessions where (query:children OR origin:claude-code)"
        )),
        ["#4", "#2", "#1"]
    );
}

fn save(w: &mut TestArchive, name: &str, text: &str) {
    save_definition(w, name, text);
}

#[test]
fn parameterized_saved_references_agree_with_direct_references() {
    let mut w = morrow();
    w.fx.note(
        "hidden",
        None,
        None,
        "session:claude-code:7f3a2c10-0002-4000-8000-000000000002",
    )
    .unwrap();
    save(&mut w, "pick", "$id");
    save(&mut w, "forward", "query:pick(id=$id)");
    save(&mut w, "nested", "query:forward(id=$id)");
    for id in [
        "7f3a2c10-0003-4000-8000-000000000003",
        "7f3a2c10-0002-4000-8000-000000000002",
    ] {
        let expected = row_ids(&q(&w, id));
        assert_eq!(expected.len(), 1);
        for name in ["pick", "forward", "nested"] {
            for query in [
                format!("query:{name}(id={id})"),
                format!("query:{name}(id=\"{id}\")"),
                format!("sessions where query:{name}(id={id})"),
                format!("sessions where query:{name}(id={id}) origin:claude-code"),
            ] {
                assert_eq!(row_ids(&q(&w, &query)), expected, "{query}");
            }
        }
    }
    // Each call binds its own defaults, so two calls side by side select
    // what both of their populations hold (docs/query.md#defaults): the
    // reference lifts the reachability defaults for its own call only.
    let both = row_ids(&q(
        &w,
        "sessions where query:pick(id=chrono) query:pick(id=7f3a2c10-0003)",
    ));
    let chrono = row_ids(&q(&w, "query:pick(id=chrono)"));
    let referenced = row_ids(&q(&w, "query:pick(id=7f3a2c10-0003)"));
    assert_eq!(referenced.len(), 1);
    assert_eq!(
        both,
        chrono
            .intersection(&referenced)
            .cloned()
            .collect::<BTreeSet<_>>()
    );
    // Calls sharing a name may bind different values in the same program.
    assert_eq!(
        row_ids(&q(
            &w,
            "sessions where query:pick(id=7f3a2c10-0003) | union (sessions where query:pick(id=7f3a2c10-0002))"
        )),
        row_ids(&q(&w, "7f3a2c10-0003"))
            .union(&row_ids(&q(&w, "7f3a2c10-0002")))
            .cloned()
            .collect::<BTreeSet<_>>()
    );
}

#[test]
fn saved_references_under_or_and_not_keep_outer_defaults() {
    let mut w = morrow();
    save(&mut w, "pick", "$id");
    save(&mut w, "either", "query:pick(id=$id) OR origin:claude-code");
    save(&mut w, "other", "NOT query:pick(id=$id)");
    let id = "7f3a2c10-0003";
    for (name, condition) in [
        ("either", format!("{id} OR origin:claude-code")),
        ("other", format!("NOT {id}")),
    ] {
        let expected = row_ids(&q(&w, &format!("sessions where {condition}")));
        assert!(!expected.contains("claude-code:7f3a2c10-0003-4000-8000-000000000003"));
        for query in [
            format!("query:{name}(id={id})"),
            format!("sessions where query:{name}(id={id})"),
        ] {
            assert_eq!(row_ids(&q(&w, &query)), expected, "{query}");
        }
    }
    for (condition, direct) in [
        (
            format!("query:pick(id={id}) OR origin:claude-code"),
            format!("{id} OR origin:claude-code"),
        ),
        (format!("NOT query:pick(id={id})"), format!("NOT {id}")),
    ] {
        assert_eq!(
            row_ids(&q(&w, &format!("sessions where {condition}"))),
            row_ids(&q(&w, &format!("sessions where {direct}")))
        );
    }
}

#[test]
fn saved_relation_errors_reach_the_query_error_path() {
    let mut w = morrow();
    save(&mut w, "pick", "$id");
    save(&mut w, "forward", "query:pick(id=$id)");
    save(&mut w, "twice", "origin:codex");
    save(&mut w, "twice", "origin:claude-code");
    for (call, kind, message) in [
        ("absent", "invalid_query", "no saved query"),
        ("pick", "invalid_query", "needs a value"),
        ("forward", "invalid_query", "needs a value"),
        (
            "pick(id=7f3a2c10, extra=1)",
            "invalid_query",
            "has no parameter",
        ),
        ("twice", "ambiguous", "query:twice"),
    ] {
        for query in [
            format!("query:{call}"),
            format!("sessions where query:{call}"),
        ] {
            let failure = run_q(&w, ask(&query)).unwrap_err();
            assert_eq!(failure.code(), kind, "{query}: {}", failure.message);
            if call == "twice" {
                let rows = q(&w, "assertions where kind:saved_query key:twice");
                let expected: std::collections::BTreeSet<_> = rows["rows"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|row| {
                        polylogue_model::Reference::parse(row["reference"].as_str().unwrap())
                            .unwrap()
                            .to_string()
                    })
                    .collect();
                assert_eq!(expected.len(), 2);
                assert!(
                    failure.message.starts_with("query:twice "),
                    "{}",
                    failure.message
                );
                assert_eq!(
                    failure
                        .candidates()
                        .iter()
                        .cloned()
                        .collect::<std::collections::BTreeSet<_>>(),
                    expected
                );
            } else {
                assert!(
                    failure.message.contains(message),
                    "{query}: {}",
                    failure.message
                );
            }
        }
    }
    // Native diagnostic details remain available before any definition is saved.
    for (text, kind, message) in [
        (
            "sessions where invented_field:true",
            "unknown_field",
            "invented_field",
        ),
        (
            "sessions where cost > nonsense",
            "bad_value",
            "amount in dollars",
        ),
    ] {
        let failure = run_q(&w, ask(text)).unwrap_err();
        assert_eq!(failure.code(), kind, "{text}");
        assert!(failure.message.contains(message), "{}", failure.message);
    }
}
