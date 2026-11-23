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
) -> AnnotationFact {
    AnnotationFact {
        id: id.into(),
        root: root.into(),
        kind: AnnotationKind::Setting,
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
        created_at: at.map(date),
        valid_until: None,
        links: base
            .into_iter()
            .map(|base| AnnotationLink {
                role: LinkRole::Revises,
                reference: reference(base),
                seen_hash: None,
                quote: None,
            })
            .collect(),
    }
}
fn known(value: bool) -> SettingAt {
    SettingAt::Value(Some(settings::Value::Boolean(value)))
}
fn withdraw(id: &str, target: &str, at: Option<i64>) -> AnnotationFact {
    AnnotationFact {
        id: id.into(),
        root: id.into(),
        kind: AnnotationKind::Withdrawal,
        key: None,
        value: None,
        author: Author::Unattributed,
        via: polylogue_model::authorship::Channel::Import,
        machine: "synthetic-host".into(),
        created_at: at.map(date),
        valid_until: None,
        links: vec![AnnotationLink {
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
    // Commit order and the unrelated undated setting never date or choose
    // this family's effective value. Restoration follows the graph.
    let facts = [
        withdraw("restore", "annotation:withdraw", Some(4)),
        setting(
            "setting",
            "second",
            "browser.actions",
            false,
            Some(2),
            Some("annotation:setting@first"),
        ),
        setting("other", "other", "drive.ai-studio", true, None, None),
        withdraw("withdraw", "annotation:setting@second", Some(3)),
        setting("setting", "first", "browser.actions", true, Some(1), None),
    ];
    for (i, fact) in facts.iter().enumerate() {
        super::super::apply(&conn, std::slice::from_ref(fact), 100 - i as i64).unwrap();
    }
    conn.execute_batch("BEGIN").unwrap();
    assert_eq!(
        history(&conn, "browser.actions", date(4)).unwrap(),
        vec![
            (date(1), known(true)),
            (date(2), known(false)),
            (date(3), known(true)),
            (date(4), known(false)),
        ]
    );
    assert_eq!(
        history(&conn, "browser.actions", date(1)).unwrap(),
        vec![(date(1), known(true))]
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
            Some("annotation:setting@first"),
        ),
        setting(
            "setting",
            "right",
            "browser.actions",
            true,
            Some(3),
            Some("annotation:setting@first"),
        ),
    ];
    super::super::apply(&conn, &facts, 1).unwrap();
    conn.execute_batch("BEGIN").unwrap();
    assert_eq!(
        history(&conn, "browser.actions", date(3)).unwrap(),
        vec![
            (date(1), known(true)),
            (date(2), known(false)),
            (
                date(3),
                SettingAt::Conflict(vec![
                    "annotation:setting@left".into(),
                    "annotation:setting@right".into()
                ])
            ),
        ]
    );
    assert_eq!(
        history(&conn, "browser.actions", date(2)).unwrap(),
        vec![(date(1), known(true)), (date(2), known(false))]
    );
}

#[test]
fn history_reports_conflict_interval_as_unknown() {
    let conn = db();
    // Two hosts each set the key first; a later revision naming both heads
    // resolves it.
    let mut resolved = setting(
        "laptop",
        "settled",
        "browser.actions",
        false,
        Some(3),
        Some("annotation:laptop@laptop"),
    );
    resolved.links.push(AnnotationLink {
        role: LinkRole::Revises,
        reference: reference("annotation:desktop@desktop"),
        seen_hash: None,
        quote: None,
    });
    let facts = [
        setting("laptop", "laptop", "browser.actions", true, Some(1), None),
        setting(
            "desktop",
            "desktop",
            "browser.actions",
            false,
            Some(2),
            None,
        ),
        resolved,
    ];
    for (i, fact) in facts.iter().enumerate() {
        super::super::apply(&conn, std::slice::from_ref(fact), i as i64 + 1).unwrap();
    }
    conn.execute_batch("BEGIN").unwrap();
    assert_eq!(
        history(&conn, "browser.actions", date(3)).unwrap(),
        vec![
            (date(1), known(true)),
            (
                date(2),
                SettingAt::Conflict(vec![
                    "annotation:desktop@desktop".into(),
                    "annotation:laptop@laptop".into()
                ])
            ),
            (date(3), known(false)),
        ]
    );
}
