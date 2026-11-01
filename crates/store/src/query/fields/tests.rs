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
/// The scope index's live state agrees with the query language's on every
/// combination of live facts, inside and outside the live window.
#[test]
fn active_live_state_agrees_with_the_sql_rule() {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute_batch(
        "CREATE TABLE s(live_last_event TEXT, live_last_event_at INTEGER,
                        live_waiting_on TEXT, last_activity_at INTEGER)",
    )
    .unwrap();
    let now = 10 * LIVE_DAY_MS;
    let events = [
        None,
        Some("hook_pre_tool_use"),
        Some("hook_stop"),
        Some("hook_notification"),
        Some("hook_session_end"),
        Some("hook_user_prompt_submit"),
    ];
    let times = [
        None,
        Some(now),
        Some(now + 60_000),
        Some(now - LIVE_WINDOW_MS),
        Some(now - LIVE_WINDOW_MS - 1),
        Some(now - LIVE_DAY_MS),
    ];
    let sql = format!("SELECT {} FROM s", live_state("s", now));
    let mut cases = 0;
    for event in events {
        for at in times {
            for waiting in [None, Some("permission")] {
                for activity in [None, Some(now - 1_000)] {
                    conn.execute("DELETE FROM s", []).unwrap();
                    conn.execute(
                        "INSERT INTO s VALUES (?1, ?2, ?3, ?4)",
                        rusqlite::params![event, at, waiting, activity],
                    )
                    .unwrap();
                    let from_sql: String = conn.query_row(&sql, [], |r| r.get(0)).unwrap();
                    let expected = ["waiting", "working", "idle"]
                        .contains(&from_sql.as_str())
                        .then_some(from_sql.as_str());
                    assert_eq!(
                        active_live_state(event, at, waiting, now),
                        expected,
                        "{event:?} at {at:?} waiting {waiting:?} activity {activity:?}"
                    );
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 144);
}
