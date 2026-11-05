use super::*;

#[test]
fn every_non_special_field_has_sql() {
    let a = Aliases::plain();
    for f in FIELDS {
        if f.kind != Kind::Special {
            assert!(f.expr(&a, 0).is_some(), "{:?}.{}", f.unit, f.name);
        }
    }
}

#[test]
fn names_are_unique_per_unit() {
    let mut seen = std::collections::HashSet::new();
    for f in FIELDS {
        assert!(seen.insert((f.unit, f.name)), "{:?}.{}", f.unit, f.name);
    }
}

#[test]
fn exactly_the_special_fields_have_a_compile_rule() {
    for f in FIELDS {
        assert_eq!(
            f.kind == Kind::Special,
            f.compile.is_some(),
            "{:?}.{}",
            f.unit,
            f.name
        );
    }
}

/// The catalog declares every field; the store gives each declared field one
/// rule and has no rule for a field the catalog does not declare.
#[test]
fn the_store_has_a_rule_for_exactly_the_catalog_fields() {
    for declared in polylogue_query::fields::FIELDS {
        assert!(
            FIELDS
                .iter()
                .any(|f| f.unit == declared.unit && f.name == declared.name),
            "{:?}.{} has no store rule",
            declared.unit,
            declared.name
        );
    }
    for f in FIELDS {
        assert!(
            polylogue_query::fields::FIELDS
                .iter()
                .any(|declared| declared.unit == f.unit && declared.name == f.name),
            "{:?}.{} is not in the catalog",
            f.unit,
            f.name
        );
    }
}

/// law: live-state-one-rule
/// SQL and the in-memory scope index use the same model through time boundaries.
#[test]
fn active_live_state_agrees_with_the_sql_rule() {
    use polylogue_model::live::{Activity, LIVE_WINDOW_MS, LiveFacts, STALE_FOR_MS, Since, view};
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    crate::sql_functions::register(&conn).unwrap();
    conn.execute_batch("CREATE TABLE s(live_facts TEXT, last_activity_at INTEGER)")
        .unwrap();
    let now = 10 * STALE_FOR_MS;
    for activity in [Activity::Working, Activity::Idle, Activity::Waiting] {
        for at in [
            now + 60_000,
            now,
            now - LIVE_WINDOW_MS,
            now - LIVE_WINDOW_MS - 1,
            now - STALE_FOR_MS,
        ] {
            for grew_at in [None, Some(now - 1_000)] {
                let facts = LiveFacts {
                    hook_activity: Some(Since {
                        value: activity,
                        since: at,
                    }),
                    last_at: Some(at),
                    ..Default::default()
                };
                conn.execute("DELETE FROM s", []).unwrap();
                conn.execute(
                    "INSERT INTO s VALUES (?1, ?2)",
                    rusqlite::params![serde_json::to_string(&facts).unwrap(), grew_at],
                )
                .unwrap();
                let state: String = conn
                    .query_row(
                        &format!("SELECT {} FROM s", live_state("s", now)),
                        [],
                        |row| row.get(0),
                    )
                    .unwrap();
                assert_eq!(state, view(Some(&facts), grew_at, now).state.as_str());
            }
        }
    }
}
