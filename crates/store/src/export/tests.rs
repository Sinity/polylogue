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
    let first = ex.export(&db, &out).unwrap().unwrap();
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
    assert_eq!(ex.export(&db, &out).unwrap(), None);
    c.execute_batch(
        "UPDATE threads SET title = 'renamed' WHERE id = 'a'; DELETE FROM threads WHERE id = 'b';",
    )
    .unwrap();
    // Changed: due once quiet, so the first look only notes the change.
    assert_eq!(ex.export(&db, &out).unwrap(), None);
    let second = ex.export(&db, &out).unwrap().unwrap();
    assert_eq!(second, Exported { rows: 1, gone: 1 });
    let l = lines(&out);
    assert_eq!(l.len(), 6);
    assert_eq!(l[4]["r"]["title"], "renamed");
    let kh = format!("{:016x}", row_key("threads", &json!(["b"])));
    assert_eq!(l[5], json!({"t": "threads", "kh": kh, "gone": true}));
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
    Exporter::default().export(&db, &out).unwrap().unwrap();
    c.execute_batch("INSERT INTO g VALUES (2, 'y');").unwrap();
    let mut fresh = Exporter::default();
    let done = fresh.export(&db, &out).unwrap().unwrap();
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
