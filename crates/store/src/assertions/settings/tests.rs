use super::*;
use polylogue_model::authorship::Channel;

fn date(n: i64) -> Timestamp {
    Timestamp::from_unix_ms(n * 1000).unwrap()
}
fn reference(value: &str) -> Reference {
    Reference::parse(value).unwrap()
}
fn db() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(crate::schema::ARCHIVE).unwrap();
    conn
}
fn setting(
    root: &str,
    id: &str,
    key: &str,
    value: bool,
    at: Option<i64>,
    base: Option<&str>,
) -> AssertionFact {
    AssertionFact {
        id: id.into(),
        root: root.into(),
        kind: AssertionKind::Setting,
        key: Some(key.into()),
        value: Some(
            serde_json::to_value(values::Setting {
                machine: None,
                value: settings::Value::Boolean(value),
            })
            .unwrap(),
        ),
        author: Author::Unattributed,
        via: Some(Channel::Import),
        machine: None,
        created_at: at.map(date),
        valid_until: None,
        links: base
            .into_iter()
            .map(|base| AssertionLink {
                role: LinkRole::Revises,
                reference: reference(base),
                seen_hash: None,
            })
            .collect(),
    }
}
fn withdraw(id: &str, target: &str, at: Option<i64>) -> AssertionFact {
    AssertionFact {
        id: id.into(),
        root: id.into(),
        kind: AssertionKind::Withdrawal,
        key: None,
        value: None,
        author: Author::Unattributed,
        via: None,
        machine: None,
        created_at: at.map(date),
        valid_until: None,
        links: vec![AssertionLink {
            role: LinkRole::Withdraws,
            reference: reference(target),
            seen_hash: None,
        }],
    }
}

#[test]
fn historical_setting_uses_dated_graph_with_withdrawal_and_restoration() {
    let conn = db();
    // Receipt order and the unrelated undated setting never date or choose
    // this family's effective value. Restoration follows the graph.
    let facts = [
        withdraw("restore", "assertion:withdraw", Some(4)),
        setting(
            "setting",
            "second",
            "browser.actions",
            false,
            Some(2),
            Some("assertion:setting@first"),
        ),
        setting("other", "other", "drive.ai-studio", true, None, None),
        withdraw("withdraw", "assertion:setting@second", Some(3)),
        setting("setting", "first", "browser.actions", true, Some(1), None),
    ];
    for (i, fact) in facts.iter().enumerate() {
        super::super::apply(&conn, std::slice::from_ref(fact), 100 - i as i64).unwrap();
    }
    conn.execute_batch("BEGIN").unwrap();
    assert_eq!(
        history(&conn, "browser.actions", date(4)).unwrap(),
        vec![
            (date(1), Some(settings::Value::Boolean(true))),
            (date(2), Some(settings::Value::Boolean(false))),
            (date(3), Some(settings::Value::Boolean(true))),
            (date(4), Some(settings::Value::Boolean(false))),
        ]
    );
    assert_eq!(
        history(&conn, "browser.actions", date(1)).unwrap(),
        vec![(date(1), Some(settings::Value::Boolean(true)))]
    );
}

#[test]
fn undated_affecting_fact_refuses_history_but_not_structural_current_currency() {
    for undated in [false, true] {
        let conn = db();
        let first = setting("setting", "first", "browser.actions", true, Some(1), None);
        let effect = if undated {
            withdraw("unknown", "assertion:setting", None)
        } else {
            setting(
                "setting",
                "unknown",
                "browser.actions",
                false,
                None,
                Some("assertion:setting@first"),
            )
        };
        super::super::apply(&conn, &[first, effect], 1).unwrap();
        conn.execute_batch("BEGIN").unwrap();
        assert!(
            matches!(history(&conn,"browser.actions",date(4)),Err(StoreError::ReferenceUnavailable {reference,..}) if reference==if undated {"assertion:unknown@unknown"} else {"assertion:setting@unknown"})
        );
        let current = super::super::heads(&conn, &reference("assertion:setting")).unwrap();
        assert_eq!(current.len(), usize::from(!undated));
        if !undated {
            assert_eq!(current[0], reference("assertion:setting@unknown"));
        }
    }
}

#[test]
fn historical_setting_forks_never_choose_the_latest_writer_clock() {
    let conn = db();
    let facts = [
        setting("setting", "first", "browser.actions", true, Some(1), None),
        setting(
            "setting",
            "left",
            "browser.actions",
            false,
            Some(2),
            Some("assertion:setting@first"),
        ),
        setting(
            "setting",
            "right",
            "browser.actions",
            true,
            Some(3),
            Some("assertion:setting@first"),
        ),
    ];
    super::super::apply(&conn, &facts, 1).unwrap();
    conn.execute_batch("BEGIN").unwrap();
    assert!(
        matches!(history(&conn,"browser.actions",date(3)),Err(StoreError::Ambiguous {candidates,..}) if candidates==vec!["assertion:setting@left","assertion:setting@right"])
    );
    assert_eq!(
        history(&conn, "browser.actions", date(2))
            .unwrap()
            .last()
            .unwrap(),
        &(date(2), Some(settings::Value::Boolean(false)))
    );
}

#[test]
fn undated_restoration_keeps_current_graph_readable_and_history_unknown() {
    let conn = db();
    let facts = [
        setting("setting", "first", "browser.actions", true, Some(1), None),
        withdraw("withdraw", "assertion:setting", Some(2)),
        withdraw("restore", "assertion:withdraw", None),
    ];
    super::super::apply(&conn, &facts, 1).unwrap();
    conn.execute_batch("BEGIN").unwrap();
    assert!(
        matches!(history(&conn,"browser.actions",date(4)),Err(StoreError::ReferenceUnavailable {reference,..}) if reference=="assertion:restore@restore")
    );
    assert_eq!(
        super::super::heads(&conn, &reference("assertion:setting")).unwrap(),
        [reference("assertion:setting@first")]
    );
}
