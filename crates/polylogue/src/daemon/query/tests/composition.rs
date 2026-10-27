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
    assert_eq!(q(&w, &format!("sessions where kind:* cwd in ({set})"))["count"], 0);
    assert_eq!(
        ids(&q(&w, &format!("sessions where kind:* NOT cwd in ({set})"))),
        ["#4", "#2", "#1"]
    );
    let empty = "sessions where origin:codex repo:absent | group by cwd";
    assert_eq!(
        ids(&q(&w, &format!("sessions where kind:* NOT cwd in ({empty})"))),
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
