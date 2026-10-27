use super::*;

fn database(w: &Morrow) -> std::path::PathBuf {
    let db=w.file(".codex/state_99.sqlite");
    fs::create_dir_all(db.parent().unwrap()).unwrap();
    rusqlite::Connection::open(&db).unwrap().execute_batch("CREATE TABLE kept_rows(id INTEGER PRIMARY KEY,body TEXT); WITH RECURSIVE ids(n) AS(VALUES(1) UNION ALL SELECT n+1 FROM ids WHERE n<12) INSERT INTO kept_rows SELECT n,printf('%01000000d',n) FROM ids;").unwrap();
    db
}
fn row_lines(w: &Morrow, db: &std::path::Path) -> Vec<serde_json::Value> {
    String::from_utf8(w.stored_bytes(db).unwrap()).unwrap().lines().map(|line|serde_json::from_str(line).unwrap()).collect()
}

#[test]
fn interrupted_database_read_keeps_committed_batches_and_retries_without_duplicates() {
    for action in [Action::Fail,Action::Panic] {
        fault::clear();
        let w=Morrow::lay_out(Set::Main);
        let db=database(&w);
        fault::arm("tool_export.after_row","kept_rows",action,9,1);
        {
            let (mut engine,_lock)=w.engine();
            let first=engine.run_pass().unwrap();
            assert!(first.errors.iter().any(|e|e.contains("database")),"{:?}",first.errors);
        }
        fault::clear();
        w.run_pass();
        let rows=row_lines(&w,&db);
        let keys:Vec<_>=rows.iter().filter(|r|r["t"]=="kept_rows").map(|r|r["k"].to_string()).collect();
        assert_eq!(keys.len(),12,"{action:?}");
        assert_eq!(keys.iter().collect::<std::collections::BTreeSet<_>>().len(),12);
        assert!(!w.root.join("exports").exists());
    }
}

#[test]
fn stopped_comparison_commit_reads_only_the_unacknowledged_suffix() {
    fault::clear();
    let w=Morrow::lay_out(Set::Main);
    let db=database(&w);
    // Fail both attempts to write this group's index rows; its bytes already
    // committed. The next reader reconstructs only their comparison hashes.
    fault::arm("write.index_commit","",Action::Fail,0,2);
    {
        let (mut engine,_lock)=w.engine();
        let _=engine.run_pass();
    }
    fault::clear();
    w.run_pass();
    let rows=row_lines(&w,&db);
    assert_eq!(rows.iter().filter(|r|r["t"]=="kept_rows").count(),12);
}
