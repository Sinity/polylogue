use super::*;

fn database(w: &Home) -> std::path::PathBuf {
    let db = w.path(".codex/state_99.sqlite");
    fs::create_dir_all(db.parent().unwrap()).unwrap();
    rusqlite::Connection::open(&db).unwrap().execute_batch("CREATE TABLE kept_rows(id INTEGER PRIMARY KEY,body TEXT); WITH RECURSIVE ids(n) AS(VALUES(1) UNION ALL SELECT n+1 FROM ids WHERE n<12) INSERT INTO kept_rows SELECT n,printf('%01000000d',n) FROM ids;").unwrap();
    db
}
fn row_lines(w: &Home, db: &std::path::Path) -> Vec<serde_json::Value> {
    String::from_utf8(w.stored_bytes(db).unwrap())
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn interrupted_database_read_keeps_committed_batches_and_retries_without_duplicates() {
    for action in [Action::Fail, Action::Panic] {
        fault::clear();
        let w = Home::morrow(Set::Main);
        let db = database(&w);
        fault::arm("database.after_row", "kept_rows", action, 9, 1);
        {
            let (mut engine, _lock) = w.engine(polylogue_testkit::WHOLE_FILES);
            let first = engine.run_pass().unwrap();
            assert!(
                first
                    .errors
                    .iter()
                    .any(|e| e.to_string().contains("database")),
                "{:?}",
                first.errors
            );
        }
        fault::clear();
        w.store();
        let rows = row_lines(&w, &db);
        let keys: Vec<_> = rows
            .iter()
            .filter(|r| r["t"] == "kept_rows")
            .map(|r| r["k"].to_string())
            .collect();
        assert_eq!(keys.len(), 12, "{action:?}");
        assert_eq!(
            keys.iter().collect::<std::collections::BTreeSet<_>>().len(),
            12
        );
        // printf('%01000000d', n): the id, zero-padded to a million digits.
        for record in rows.iter().filter(|r| r["t"] == "kept_rows") {
            let id = record["r"]["id"].as_i64().unwrap();
            assert_eq!(
                record["r"]["body"].as_str().unwrap(),
                format!("{}{id}", "0".repeat(1_000_000 - id.to_string().len())),
                "{action:?}: row {id}"
            );
        }
    }
}

#[test]
fn a_failed_comparison_commit_recovers_stored_database_rows_on_retry() {
    fault::clear();
    let w = Home::morrow(Set::Main);
    let db = database(&w);
    // Original bytes commit first. A failed parse must finish through the
    // stored-input path before another source read compares the same rows.
    fault::arm(
        "tool_database.comparison_commit",
        "kept_rows",
        Action::Fail,
        0,
        2,
    );
    {
        let (mut engine, _lock) = w.engine(polylogue_testkit::WHOLE_FILES);
        let result = engine.run_pass().unwrap();
        assert!(
            result.errors.iter().any(|e| e
                .to_string()
                .contains("injected fault at tool_database.comparison_commit")),
            "{:?}",
            result.errors
        );
    }
    fault::clear();
    w.store();
    let rows = row_lines(&w, &db);
    assert_eq!(rows.iter().filter(|r| r["t"] == "kept_rows").count(), 12);
}

#[test]
fn database_progress_does_not_reload_unrelated_sources() {
    fault::clear();
    let w = Home::morrow(Set::Main);
    let db = database(&w);
    let (mut engine, _lock) = w.engine(polylogue_testkit::WHOLE_FILES);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    rusqlite::Connection::open(&db)
        .unwrap()
        .execute("UPDATE kept_rows SET body='changed' WHERE id=1", [])
        .unwrap();
    fault::arm(
        "read.watched_inputs",
        &engine.config.machine,
        Action::Fail,
        0,
        100,
    );
    let results = [engine.run_pass(), engine.run_pass()];
    fault::clear();
    for result in results {
        let pass = result.unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    }
    assert!(
        row_lines(&w, &db)
            .iter()
            .any(|row| row["r"]["body"] == "changed")
    );
}
