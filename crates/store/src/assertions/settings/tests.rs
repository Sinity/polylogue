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
        via: Channel::Import,
        machine: "synthetic-host".into(),
        created_at: date(at.unwrap_or(0)),
        valid_until: None,
        links: base
            .into_iter()
            .map(|base| AssertionLink {
                role: LinkRole::Revises,
                reference: reference(base),
                seen_hash: None,
                quote: None,
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
        via: polylogue_model::authorship::Channel::Import,
        machine: "synthetic-host".into(),
        created_at: date(at.unwrap_or(0)),
        valid_until: None,
        links: vec![AssertionLink {
            role: LinkRole::Withdraws,
            reference: reference(target),
            seen_hash: None,
            quote: None,
        }],
    }
}

#[test]
fn setting_history_follows_withdrawal_and_restoration_by_writer_clock() {
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
fn setting_history_forks_never_choose_the_latest_writer_clock() {
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
