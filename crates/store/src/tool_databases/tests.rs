use super::*;

fn lines(p: &Path) -> Vec<Value> {
    fs::read_to_string(p)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn exports_new_changed_and_gone_rows_only() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("state_5.sqlite");
    let out = dir.path().join("exports/state_5.sqlite.jsonl");
    let c = Connection::open(&db).unwrap();
    c.execute_batch(
        "PRAGMA journal_mode=WAL;
         CREATE TABLE threads (id TEXT PRIMARY KEY, title TEXT, data BLOB, n REAL);
         CREATE TABLE notes (body TEXT);
         INSERT INTO threads VALUES ('a', 'first', x'00ff', 1.5), ('b', 'second', NULL, NULL);
         INSERT INTO notes VALUES ('kept');",
    )
    .unwrap();
    let mut ex = Exporter::default();
    let first = ex.export(&db, &out, None).unwrap().unwrap();
    assert_eq!(first, Exported { rows: 3, gone: 0 });
    let l = lines(&out);
    assert!(
        l[0]["schema"]["threads"]
            .as_str()
            .unwrap()
            .contains("CREATE TABLE threads")
    );
    let a = l.iter().find(|v| v["k"] == json!(["a"])).unwrap();
    assert_eq!(a["r"]["data"]["base64"], "AP8=");
    assert_eq!(a["r"]["n"], 1.5);
    // Unchanged files: nothing is due.
    assert_eq!(ex.export(&db, &out, None).unwrap(), None);
    c.execute_batch(
        "UPDATE threads SET title = 'renamed' WHERE id = 'a'; DELETE FROM threads WHERE id = 'b';",
    )
    .unwrap();
    // Changed: due once quiet, so the first look only notes the change.
    assert_eq!(ex.export(&db, &out, None).unwrap(), None);
    let second = ex.export(&db, &out, None).unwrap().unwrap();
    assert_eq!(second, Exported { rows: 1, gone: 1 });
    let l = lines(&out);
    assert_eq!(l.len(), 6);
    assert_eq!(l[4]["r"]["title"], "renamed");
    let kh = format!("{:016x}", row_key("threads", &json!(["b"])));
    assert_eq!(
        l[5],
        json!({"t": "threads", "k":["b"], "kh": kh, "gone": true})
    );
}

#[test]
fn a_restart_rebuilds_what_was_exported_from_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("goals_1.sqlite");
    let out = dir.path().join("goals_1.sqlite.jsonl");
    let c = Connection::open(&db).unwrap();
    c.execute_batch(
        "CREATE TABLE g (id INTEGER PRIMARY KEY, s TEXT); INSERT INTO g VALUES (1, 'x');",
    )
    .unwrap();
    Exporter::default()
        .export(&db, &out, None)
        .unwrap()
        .unwrap();
    c.execute_batch("INSERT INTO g VALUES (2, 'y');").unwrap();
    let mut fresh = Exporter::default();
    let done = fresh.export(&db, &out, None).unwrap().unwrap();
    assert_eq!(done, Exported { rows: 1, gone: 0 });
    // The schema line is not repeated.
    assert_eq!(
        lines(&out)
            .iter()
            .filter(|v| v.get("schema").is_some())
            .count(),
        1
    );
}

#[test]
fn dropped_tables_commit_exact_removals_and_can_reappear() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("state.sqlite");
    let out = dir.path().join("state.jsonl");
    let source = Connection::open(&db).unwrap();
    source
        .execute_batch(
            "CREATE TABLE t(a TEXT,b INTEGER,PRIMARY KEY(a,b)) WITHOUT ROWID;
        INSERT INTO t VALUES('a',1),('a',2);",
        )
        .unwrap();
    Exporter::default().export(&db, &out, None).unwrap();
    source.execute_batch("DROP TABLE t").unwrap();
    let gone = Exporter::default()
        .export(&db, &out, None)
        .unwrap()
        .unwrap();
    assert_eq!(gone, Exported { rows: 0, gone: 2 });
    assert!(
        lines(&out)
            .iter()
            .any(|v| v["gone"] == true && v["k"] == json!(["a", 1]))
    );
    source
        .execute_batch(
            "CREATE TABLE t(a TEXT,b INTEGER,PRIMARY KEY(a,b)) WITHOUT ROWID;
        INSERT INTO t VALUES('a',1);",
        )
        .unwrap();
    assert_eq!(
        Exporter::default()
            .export(&db, &out, None)
            .unwrap()
            .unwrap(),
        Exported { rows: 1, gone: 0 }
    );
}

#[test]
fn malformed_exports_without_a_marker_are_preserved_and_refused() {
    for suffix in ["broken\n", "{\"t\":\"t\"}\n", "{\"schema\":{}}"] {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("old.jsonl");
        fs::write(&out, suffix).unwrap();
        assert!(
            Exporter::default()
                .export(&dir.path().join("missing.sqlite"), &out, None)
                .is_err()
        );
        assert_eq!(fs::read_to_string(&out).unwrap(), suffix);
        assert!(!synced_prefix::checkpoint(&out).exists());
    }
}

#[test]
fn changed_completed_bytes_are_not_truncated_or_silently_replaced() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("state.sqlite");
    let out = dir.path().join("state.jsonl");
    Connection::open(&db)
        .unwrap()
        .execute_batch("CREATE TABLE t(id); INSERT INTO t VALUES(1)")
        .unwrap();
    let mut exporter = Exporter::default();
    exporter.export(&db, &out, None).unwrap();
    let mut corrupt = fs::read(&out).unwrap();
    corrupt[0] = b'!';
    corrupt.extend_from_slice(b"uncommitted tail");
    fs::write(&out, &corrupt).unwrap();
    assert!(exporter.export(&db, &out, None).is_err());
    assert_eq!(fs::read(&out).unwrap(), corrupt);
}

#[test]
fn a_lost_marker_refuses_existing_bytes_without_changing_them() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("state.sqlite");
    let out = dir.path().join("state.jsonl");
    Connection::open(&db)
        .unwrap()
        .execute_batch("CREATE TABLE t(id); INSERT INTO t VALUES(1)")
        .unwrap();
    Exporter::default().export(&db, &out, None).unwrap();
    let before = fs::read(&out).unwrap();
    fs::remove_file(synced_prefix::checkpoint(&out)).unwrap();
    assert!(Exporter::default().export(&db, &out, None).is_err());
    assert_eq!(fs::read(&out).unwrap(), before);
    assert!(!synced_prefix::checkpoint(&out).exists());
}

#[test]
fn duplicate_nullable_primary_keys_refuse_the_whole_capture() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("state.sqlite");
    let out = dir.path().join("state.jsonl");
    Connection::open(&db)
        .unwrap()
        .execute_batch(
            "CREATE TABLE t(id TEXT PRIMARY KEY,n TEXT);
        INSERT INTO t VALUES(NULL,'a'),(NULL,'b')",
        )
        .unwrap();
    assert!(Exporter::default().export(&db, &out, None).is_err());
    assert_eq!(fs::read(&out).unwrap(), b"");
}

#[test]
fn sqlite_values_are_exact_or_the_capture_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("state.sqlite");
    let out = dir.path().join("state.jsonl");
    let source = Connection::open(&db).unwrap();
    source
        .execute_batch(
            "CREATE TABLE t(rowid TEXT, big INTEGER, n, b BLOB, f REAL);
        INSERT INTO t VALUES('shadowed',9223372036854775807,NULL,x'00ff',1.25);
        INSERT INTO t VALUES('shadowed',-9223372036854775808,NULL,x'',-1.25);",
        )
        .unwrap();
    Exporter::default().export(&db, &out, None).unwrap();
    let captured = lines(&out);
    assert_eq!(captured[1]["k"], json!([1]));
    assert_eq!(captured[2]["k"], json!([2]));
    assert_eq!(captured[1]["r"]["big"], i64::MAX);
    assert_eq!(captured[2]["r"]["big"], i64::MIN);
    assert_eq!(captured[1]["r"]["n"], Value::Null);
    assert_eq!(captured[1]["r"]["f"], 1.25);
    assert_eq!(captured[1]["r"]["b"], json!({"base64":"AP8="}));
    let before = fs::read(&out).unwrap();
    for statement in [
        "UPDATE t SET rowid=CAST(x'ff' AS TEXT)",
        "UPDATE t SET rowid='valid',f=1e999",
    ] {
        source.execute_batch(statement).unwrap();
        assert!(Exporter::default().export(&db, &out, None).is_err());
        assert_eq!(fs::read(&out).unwrap(), before);
    }
}

#[test]
fn a_large_row_population_replays_and_compares_without_repeating_unchanged_rows() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("state.sqlite");
    let out = dir.path().join("state.jsonl");
    let source = Connection::open(&db).unwrap();
    source
        .execute_batch(
            "CREATE TABLE t(id INTEGER PRIMARY KEY,n TEXT);
        WITH RECURSIVE ids(n) AS(VALUES(1) UNION ALL SELECT n+1 FROM ids WHERE n<20000)
        INSERT INTO t SELECT n,printf('%0128d',n) FROM ids;",
        )
        .unwrap();
    assert_eq!(
        Exporter::default()
            .export(&db, &out, None)
            .unwrap()
            .unwrap()
            .rows,
        20000
    );
    let length = fs::metadata(&out).unwrap().len();
    // An unchanged source is not exported again after a restart.
    assert_eq!(Exporter::default().export(&db, &out, None).unwrap(), None);
    assert_eq!(fs::metadata(&out).unwrap().len(), length);
    source
        .execute_batch(
            "UPDATE t SET n='changed' WHERE id%1000=0;
        DELETE FROM t WHERE id%1000=1",
        )
        .unwrap();
    assert_eq!(
        Exporter::default()
            .export(&db, &out, None)
            .unwrap()
            .unwrap(),
        Exported { rows: 20, gone: 20 }
    );
    assert_eq!(Exporter::default().export(&db, &out, None).unwrap(), None);
    // A source that changed without changing its rows exports nothing.
    source.execute_batch("PRAGMA user_version=1").unwrap();
    assert_eq!(
        Exporter::default()
            .export(&db, &out, None)
            .unwrap()
            .unwrap(),
        Exported::default()
    );
}

mod source_binding;

#[test]
fn a_restart_reuses_the_kept_catalog_and_skips_an_unchanged_source() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("goals_1.sqlite");
    let out = dir.path().join("goals_1.sqlite.jsonl");
    let c = Connection::open(&db).unwrap();
    c.execute_batch(
        "CREATE TABLE g (id INTEGER PRIMARY KEY, s TEXT); INSERT INTO g VALUES (1, 'x');",
    )
    .unwrap();
    Exporter::default()
        .export(&db, &out, None)
        .unwrap()
        .unwrap();
    assert!(catalog::rows_path(&out).exists());
    // Nothing changed: a new process exports nothing.
    assert_eq!(Exporter::default().export(&db, &out, None).unwrap(), None);
    c.execute_batch("UPDATE g SET s = 'y' WHERE id = 1; INSERT INTO g VALUES (2, 'z');")
        .unwrap();
    assert_eq!(
        Exporter::default()
            .export(&db, &out, None)
            .unwrap()
            .unwrap(),
        Exported { rows: 2, gone: 0 }
    );
    let marker: Value =
        serde_json::from_slice(&fs::read(synced_prefix::checkpoint(&out)).unwrap()).unwrap();
    assert_eq!(marker["segments"].as_array().unwrap().len(), 2);
    // A lost catalog is made again from the capture.
    fs::remove_file(catalog::rows_path(&out)).unwrap();
    c.execute_batch("INSERT INTO g VALUES (3, 'w');").unwrap();
    assert_eq!(
        Exporter::default()
            .export(&db, &out, None)
            .unwrap()
            .unwrap(),
        Exported { rows: 1, gone: 0 }
    );
}

#[test]
fn a_capture_whose_facts_changed_is_checked_by_its_chain() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("state.sqlite");
    let out = dir.path().join("state.jsonl");
    let c = Connection::open(&db).unwrap();
    c.execute_batch("CREATE TABLE t(id); INSERT INTO t VALUES(1)")
        .unwrap();
    Exporter::default().export(&db, &out, None).unwrap();
    // Same bytes, new facts: the chain is computed again and matches.
    let bytes = fs::read(&out).unwrap();
    fs::remove_file(&out).unwrap();
    fs::write(&out, &bytes).unwrap();
    c.execute_batch("INSERT INTO t VALUES(2)").unwrap();
    assert_eq!(
        Exporter::default()
            .export(&db, &out, None)
            .unwrap()
            .unwrap(),
        Exported { rows: 1, gone: 0 }
    );
    let marker: Value =
        serde_json::from_slice(&fs::read(synced_prefix::checkpoint(&out)).unwrap()).unwrap();
    assert_eq!(marker["segments"].as_array().unwrap().len(), 2);
    assert!(marker["facts"].is_array());
}

#[test]
fn invalid_segment_lengths_refuse_without_truncating_the_capture() {
    for segments in [json!([]), json!([0]), json!([u64::MAX, 1])] {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("state.sqlite");
        let out = dir.path().join("state.jsonl");
        Connection::open(&db)
            .unwrap()
            .execute_batch("CREATE TABLE t(id); INSERT INTO t VALUES(1)")
            .unwrap();
        Exporter::default().export(&db, &out, None).unwrap();
        let before = fs::read(&out).unwrap();
        let path = synced_prefix::checkpoint(&out);
        let mut marker: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        marker["segments"] = segments;
        fs::write(&path, serde_json::to_vec(&marker).unwrap()).unwrap();
        assert!(Exporter::default().export(&db, &out, None).is_err());
        assert_eq!(fs::read(&out).unwrap(), before);
    }
}

#[test]
fn unchanged_capture_and_source_need_no_replay_after_restart() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("state.sqlite");
    let out = dir.path().join("state.jsonl");
    Connection::open(&db)
        .unwrap()
        .execute_batch("CREATE TABLE t(id); INSERT INTO t VALUES(1)")
        .unwrap();
    Exporter::default().export(&db, &out, None).unwrap();
    let mut visited = 0;
    let result = Exporter::default()
        .export_with(&db, &out, None, &mut || {
            visited += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(result, None);
    assert_eq!(visited, 0);
}

#[test]
fn comparing_a_changed_source_writes_only_changed_catalog_rows() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("state.sqlite");
    let out = dir.path().join("state.jsonl");
    let source = Connection::open(&db).unwrap();
    source
        .execute_batch(
            "CREATE TABLE t(id INTEGER PRIMARY KEY, value TEXT);
        INSERT INTO t VALUES(1,'unchanged'),(2,'before')",
        )
        .unwrap();
    Exporter::default().export(&db, &out, None).unwrap();
    Connection::open(catalog::rows_path(&out))
        .unwrap()
        .execute_batch(
            "CREATE TRIGGER unchanged_row BEFORE UPDATE ON row_state
         WHEN old.h=new.h BEGIN SELECT RAISE(ABORT,'unchanged row written'); END;",
        )
        .unwrap();
    source
        .execute_batch("UPDATE t SET value='after' WHERE id=2")
        .unwrap();
    assert_eq!(
        Exporter::default()
            .export(&db, &out, None)
            .unwrap()
            .unwrap(),
        Exported { rows: 1, gone: 0 }
    );
}

#[test]
fn readers_get_only_completed_capture_bytes_during_an_append() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("state.sqlite");
    let out = dir.path().join("state.jsonl");
    assert_eq!(completed_facts(&out).unwrap(), None);
    Connection::open(&db)
        .unwrap()
        .execute_batch("CREATE TABLE t(id); INSERT INTO t VALUES(1)")
        .unwrap();
    Exporter::default().export(&db, &out, None).unwrap();
    let completed = completed_facts(&out).unwrap().unwrap();
    fs::OpenOptions::new()
        .append(true)
        .open(&out)
        .unwrap()
        .write_all(b"incomplete row")
        .unwrap();
    assert!(fs::metadata(&out).unwrap().len() > completed.size);
    assert_eq!(completed_facts(&out).unwrap(), Some(completed));
    Exporter::default().export(&db, &out, None).unwrap();
    assert_eq!(fs::metadata(&out).unwrap().len(), completed.size);
}
