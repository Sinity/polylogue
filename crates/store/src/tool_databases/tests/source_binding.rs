use super::*;
use crate::Observation;

fn observation(db: &Path, machine: &str) -> Observation {
    Observation {
        machine: machine.into(),
        host: "synthetic-host".into(),
        path: db.display().to_string(),
        facts: None,
        seen_at_ms: 1,
        import: None,
    }
}

#[test]
fn native_binding_survives_restart_and_ignores_clock_and_host_changes() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("profile/state.db");
    fs::create_dir_all(db.parent().unwrap()).unwrap();
    Connection::open(&db).unwrap().execute_batch("CREATE TABLE rows(id INTEGER PRIMARY KEY,value TEXT); INSERT INTO rows VALUES(1,'first')").unwrap();
    let out = dir.path().join("stored/state.db.jsonl");
    let mut observed = observation(&db, "machine-a");
    Exporter::default()
        .export(&db, &out, Some(&observed))
        .unwrap();
    assert_eq!(
        lines(&out)[0]["source"],
        json!({"machine":"machine-a","path":db.display().to_string()})
    );
    let prefix = fs::read(&out).unwrap();
    observed.host = "renamed-host".into();
    observed.seen_at_ms = 999;
    Exporter::default()
        .export(&db, &out, Some(&observed))
        .unwrap();
    assert_eq!(fs::read(&out).unwrap(), prefix);
    Connection::open(&db)
        .unwrap()
        .execute_batch("UPDATE rows SET value='second'")
        .unwrap();
    let mut fresh = Exporter::default();
    fresh.export(&db, &out, Some(&observed)).unwrap();
    assert!(fs::read(&out).unwrap().starts_with(&prefix));
    assert!(
        lines(&out)
            .iter()
            .any(|line| line.pointer("/r/value") == Some(&json!("second")))
    );
}

#[test]
fn wrong_or_missing_binding_never_changes_a_bound_export() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("state.db");
    Connection::open(&db)
        .unwrap()
        .execute_batch("CREATE TABLE rows(n); INSERT INTO rows VALUES(1)")
        .unwrap();
    let out = dir.path().join("state.db.jsonl");
    let observed = observation(&db, "machine-a");
    Exporter::default()
        .export(&db, &out, Some(&observed))
        .unwrap();
    let before = fs::read(&out).unwrap();
    for wrong in [
        None,
        Some(observation(&db, "machine-b")),
        Some(observation(&dir.path().join("other/state.db"), "machine-a")),
    ] {
        assert!(
            Exporter::default()
                .export(&db, &out, wrong.as_ref())
                .is_err()
        );
        assert_eq!(fs::read(&out).unwrap(), before);
    }
}

#[test]
fn an_unbound_export_is_kept_without_guessing_its_native_profile() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("state.db");
    Connection::open(&db)
        .unwrap()
        .execute_batch("CREATE TABLE rows(n); INSERT INTO rows VALUES(1)")
        .unwrap();
    let out = dir.path().join("state.db.jsonl");
    Exporter::default().export(&db, &out, None).unwrap();
    let before = fs::read(&out).unwrap();
    assert!(
        Exporter::default()
            .export(&db, &out, Some(&observation(&db, "machine-a")))
            .is_err()
    );
    assert_eq!(fs::read(&out).unwrap(), before);
    assert!(lines(&out)[0].get("source").is_none());
}
