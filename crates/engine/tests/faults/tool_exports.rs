use super::*;
use polylogue_store::tool_databases::Captures;
use polylogue_store::tool_databases::Exporter;

#[test]
fn failed_database_capture_keeps_only_the_completed_prefix() {
    fault::clear();
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("state.sqlite");
    let out = dir.path().join("state.jsonl");
    let source = rusqlite::Connection::open(&db).unwrap();
    source
        .execute_batch(
            "CREATE TABLE threads(id TEXT PRIMARY KEY, title TEXT);
        INSERT INTO threads VALUES ('a', 'old'), ('b', 'old');",
        )
        .unwrap();
    let mut exporter = Exporter::default();
    exporter.export(&db, &out, None).unwrap();
    let before = fs::read(&out).unwrap();
    source
        .execute_batch("UPDATE threads SET title = 'new';")
        .unwrap();
    exporter.export(&db, &out, None).unwrap();
    fault::arm("tool_export.after_row", "threads", Action::Fail, 0, 1);
    assert!(exporter.export(&db, &out, None).is_err());
    fault::clear();
    assert_eq!(fs::read(&out).unwrap(), before);
    let done = exporter.export(&db, &out, None).unwrap().unwrap();
    assert_eq!(done.rows, 2);
    let after = fs::read_to_string(&out).unwrap();
    assert_eq!(after.lines().filter(|line| line.contains("new")).count(), 2);
}

#[test]
fn interrupted_capture_recovers_before_reusing_its_output() {
    for point in [
        "tool_export.after_row",
        "tool_export.after_gone",
        "tool_export.after_append",
        "tool_export.before_complete",
        "tool_export.after_complete",
    ] {
        for action in [Action::Fail, Action::Panic] {
            fault::clear();
            let dir = tempfile::tempdir().unwrap();
            let db = dir.path().join("state.sqlite");
            let out = dir.path().join("state.jsonl");
            let source = rusqlite::Connection::open(&db).unwrap();
            source
                .execute_batch(
                    "CREATE TABLE threads(id INTEGER PRIMARY KEY, title TEXT);
                INSERT INTO threads VALUES(1,'old'),(2,'old');",
                )
                .unwrap();
            let mut exporter = Exporter::default();
            exporter.export(&db, &out, None).unwrap();
            let before = fs::read(&out).unwrap();
            source
                .execute_batch(
                    "UPDATE threads SET title='intermediate' WHERE id=1;
                DELETE FROM threads WHERE id=2;",
                )
                .unwrap();
            exporter.export(&db, &out, None).unwrap();
            fault::arm(point, "", action, 0, 1);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                exporter.export(&db, &out, None)
            }));
            assert!(matches!(result, Err(_) | Ok(Err(_))), "{point}/{action:?}");
            fault::clear();
            // A later source revision must not be spliced into the failed scan.
            source
                .execute_batch("UPDATE threads SET title='latest';")
                .unwrap();
            drop(exporter);
            let mut fresh = Exporter::default();
            let done = fresh.export(&db, &out, None).unwrap().unwrap();
            assert_eq!(done.rows, 1, "{point}/{action:?}");
            let after = fs::read(&out).unwrap();
            assert!(after.starts_with(&before));
            let values: Vec<serde_json::Value> = String::from_utf8(after)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            let intermediate = values
                .iter()
                .filter(|row| row["r"]["title"] == "intermediate")
                .count();
            assert_eq!(
                intermediate,
                usize::from(point == "tool_export.after_complete"),
                "{point}/{action:?}"
            );
            assert_eq!(
                values.iter().filter(|row| row["gone"] == true).count(),
                1,
                "{point}/{action:?}"
            );
            assert_eq!(
                values
                    .iter()
                    .rev()
                    .find(|row| row.get("r").is_some())
                    .unwrap()["r"]["title"],
                "latest"
            );
        }
    }
}

#[test]
fn engine_never_collects_a_failed_database_export() {
    fault::clear();
    let w = Morrow::lay_out(Set::Main);
    let db = w.file(".codex/state_99.sqlite");
    fs::create_dir_all(db.parent().unwrap()).unwrap();
    let source = rusqlite::Connection::open(&db).unwrap();
    source
        .execute_batch(
            "CREATE TABLE threads(id TEXT PRIMARY KEY,title TEXT);
        INSERT INTO threads VALUES('7f3a2c10-0042-4000-8000-000000000042','before');",
        )
        .unwrap();
    let (mut engine, _lock) = w.engine();
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let out = fs::read_dir(w.root.join("exports/codex-database"))
        .unwrap()
        .map(|entry| entry.unwrap().path().join("state_99.sqlite.jsonl"))
        .find(|path| path.is_file())
        .unwrap();
    let before = w.stored_bytes(&out).unwrap();
    source
        .execute_batch("UPDATE threads SET title='failed';")
        .unwrap();
    engine.run_pass().unwrap();
    // This fault leaves a complete, synced output but returns an error before
    // the disposable comparison transaction commits. Mere file presence must
    // still not authorize collection by this failed call.
    fault::arm("tool_export.after_complete", "", Action::Fail, 0, 1);
    let pass = engine.run_pass().unwrap();
    fault::clear();
    assert!(
        pass.errors
            .iter()
            .any(|e| e.contains("tool_export.after_complete")),
        "{:?}",
        pass.errors
    );
    assert_eq!(w.stored_bytes(&out).unwrap(), before);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    assert!(
        String::from_utf8(w.stored_bytes(&out).unwrap())
            .unwrap()
            .contains("failed")
    );
}

#[test]
fn capture_worker_errors_and_panics_release_the_attempt_for_retry() {
    for point in [
        "tool_export.after_row",
        "tool_export.after_append",
        "tool_export.after_complete",
    ] {
        for action in [Action::Fail, Action::Panic] {
            fault::clear();
            let dir = tempfile::tempdir().unwrap();
            let db = dir.path().join("state.sqlite");
            let out = dir.path().join("state.jsonl");
            rusqlite::Connection::open(&db)
                .unwrap()
                .execute_batch(
                    "CREATE TABLE t(id INTEGER PRIMARY KEY,n TEXT);
                INSERT INTO t VALUES(1,'kept')",
                )
                .unwrap();
            let mut captures = Captures::default();
            fault::arm(point, "", action, 0, 1);
            let mut turn = captures
                .turn(vec![(db.clone(), out.clone(), None)], 1)
                .unwrap();
            let mut errors = Vec::new();
            loop {
                errors.extend(turn.errors);
                if !turn.pending {
                    break;
                }
                turn = captures.turn(Vec::new(), 1).unwrap();
            }
            fault::clear();
            assert_eq!(errors.len(), 1, "{point}/{action:?}");
            assert!(captures.paths().next().is_none());
            let mut turn = captures.turn(vec![(db, out.clone(), None)], 1).unwrap();
            loop {
                assert!(
                    turn.errors.is_empty(),
                    "{point}/{action:?}: {:?}",
                    turn.errors
                );
                if !turn.pending {
                    break;
                }
                turn = captures.turn(Vec::new(), 1).unwrap();
            }
            assert_eq!(captures.paths().next(), Some(out.as_path()));
            assert_eq!(
                fs::read_to_string(&out)
                    .unwrap()
                    .lines()
                    .filter(|line| line.contains("kept"))
                    .count(),
                1
            );
        }
    }
}
