use crate::archive::{cache, open_write};
use crate::fixture::{Fixture, MessageFacts, SessionFacts};
use polylogue_model::{Catalog, Usage};
use rusqlite::Connection;
use serde_json::json;

fn catalog(version: &str, input: u32) -> Catalog {
    serde_json::from_value(json!({
        "version": version,
        "models": {"meter": {"input": input, "output": 3,
            "cached_input": 1, "cache_write": 4, "cache_write_1h": 8,
            "tiers": {"priority": 2, "fast": 3}}}
    }))
    .unwrap()
}

fn message(f: &mut Fixture, id: &str, model: Option<&str>, usage: &Usage) -> (i64, i64) {
    let session = f
        .session(SessionFacts {
            id,
            ..SessionFacts::default()
        })
        .unwrap();
    let pk = f
        .message(
            session,
            MessageFacts {
                id,
                role: "assistant",
                written_by: "model",
                model,
                ..MessageFacts::default()
            },
        )
        .unwrap();
    f.usage(pk, usage, None).unwrap();
    (session, pk)
}

fn costs(index: &Connection) -> Vec<(String, Option<i64>)> {
    index
        .prepare("SELECT id, cost_nusd FROM message ORDER BY id")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

fn states(index: &Connection, session: Option<i64>) -> Vec<(String, i64, String, i64)> {
    index
        .prepare(
            "SELECT name, session_pk, version, from_changed FROM derive_state
                   WHERE (?1 IS NULL OR session_pk = ?1) AND name <> 'cost'
                   ORDER BY name, session_pk",
        )
        .unwrap()
        .query_map([session], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
        })
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

#[test]
fn catalog_changes_reprice_cost_without_resetting_other_derivations() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let measured = Usage {
        input: Some(1_000_000),
        output: Some(1_000_000),
        ..Usage::default()
    };
    message(&mut f, "test:known", Some("meter"), &measured);
    message(&mut f, "test:unknown", Some("not-in-catalog"), &measured);
    message(&mut f, "test:missing", Some("meter"), &Usage::default());
    message(
        &mut f,
        "test:zero",
        Some("meter"),
        &Usage {
            input: Some(0),
            output: Some(0),
            ..Usage::default()
        },
    );
    f.derive().unwrap();
    let mut index = open_write(&f.archive.index_db(), cache::SETUP).unwrap();
    index
        .execute("UPDATE message SET provider_cost_nusd = 17", [])
        .unwrap();
    let before = states(&index, None);
    for (version, input, expected) in [
        ("first", 2, 5_000_000_000),
        ("second", 4, 7_000_000_000),
        ("second", 7, 10_000_000_000),
    ] {
        let catalog = catalog(version, input);
        assert_eq!(
            super::super::run_with_catalog(&mut index, &catalog).unwrap(),
            4
        );
        assert_eq!(
            costs(&index),
            vec![
                ("test:known".into(), Some(expected)),
                ("test:missing".into(), None),
                ("test:unknown".into(), None),
                ("test:zero".into(), Some(0))
            ]
        );
        let after = states(&index, None);
        assert_eq!(
            after
                .iter()
                .map(|(name, pk, version, _)| (name, pk, version))
                .collect::<Vec<_>>(),
            before
                .iter()
                .map(|(name, pk, version, _)| (name, pk, version))
                .collect::<Vec<_>>()
        );
        assert_eq!(index.query_row("SELECT count(*) FROM derive_state d JOIN session s ON s.pk=d.session_pk WHERE d.from_changed<>s.changed", [], |r|r.get::<_,i64>(0)).unwrap(), 0);
        assert_eq!(
            super::super::run_with_catalog(&mut index, &catalog).unwrap(),
            0
        );
    }
    assert_eq!(
        index
            .query_row(
                "SELECT count(*) FROM message WHERE provider_cost_nusd = 17",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        4
    );
}

#[test]
fn changed_usage_derives_only_its_session_and_preserves_cache_and_tier_fields() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let (changed, pk) = message(&mut f, "test:changed", Some("meter"), &Usage::default());
    let (unrelated, _) = message(&mut f, "test:unrelated", None, &Usage::default());
    let mut index = open_write(&f.archive.index_db(), cache::SETUP).unwrap();
    let catalog = catalog("first", 2);
    super::super::run_with_catalog(&mut index, &catalog).unwrap();
    let before = states(&index, Some(unrelated));
    f.usage(
        pk,
        &Usage {
            input: Some(1_000_000),
            output: Some(1_000_000),
            reasoning: Some(500_000),
            cache_write: Some(1_000_000),
            cache_write_1h: Some(500_000),
            service_tier: Some("priority".into()),
            speed: Some("fast".into()),
            ..Usage::default()
        },
        None,
    )
    .unwrap();
    assert_eq!(
        super::super::run_with_catalog(&mut index, &catalog).unwrap(),
        super::super::ALL.len() as u64
    );
    assert_eq!(
        costs(&index),
        vec![
            ("test:changed".into(), Some(75_000_000_000)),
            ("test:unrelated".into(), None)
        ]
    );
    assert_eq!(states(&index, Some(unrelated)), before);
    let changed_value: i64 = index
        .query_row("SELECT changed FROM session WHERE pk=?1", [changed], |r| {
            r.get(0)
        })
        .unwrap();
    assert!(
        states(&index, Some(changed))
            .iter()
            .all(|(_, _, _, from_changed)| *from_changed == changed_value)
    );
    assert_eq!(
        index
            .query_row(
                "SELECT cache_write_5m_tokens FROM message WHERE pk = ?1",
                [pk],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        500_000
    );
}

#[test]
fn writer_uses_catalog_derivation_and_keeps_provider_reported_cost_separate() {
    use crate::fixture::writer::*;
    let mut w = WriterWorld::new();
    let mut messages = prose("a", 1, 1000, "hello");
    messages[0].model = Some("gpt-5".into());
    messages[0].usage = Usage {
        input: Some(1_000_000),
        output: Some(0),
        ..Usage::default()
    };
    messages[0].reported_cost_usd = Some(serde_json::Number::from_f64(0.25).unwrap());
    w.write("a", messages);
    assert_eq!(
        w.rows("SELECT cost_nusd, provider_cost_nusd FROM message"),
        vec![json!({"cost_nusd": 1_250_000_000i64, "provider_cost_nusd": 250_000_000})]
    );
    assert_eq!(
        w.rows("SELECT count(*) AS n FROM derive_state WHERE name = 'cost'"),
        vec![json!({"n": 1})]
    );
    assert_eq!(w.writer.derive().unwrap(), 0);
}
