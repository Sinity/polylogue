use super::*;
const PREVIOUS: &str = include_str!("../session_counts_before_transaction.sql");
fn cursor(conn: &Connection) -> (i64, String, i64, Vec<u8>) {
    conn.query_row(
        "SELECT chain,reading_version,parsed_upto,checkpoint FROM parse_state WHERE input_pk=1",
        [],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
    )
    .unwrap()
}
fn generation(c: &Connection) -> Vec<(String, String)> {
    c.prepare("SELECT key,value FROM meta WHERE key IN ('generation','cursor_key','boot_id') ORDER BY key").unwrap()
        .query_map([],|r|Ok((r.get(0)?,r.get(1)?))).unwrap().collect::<rusqlite::Result<_>>().unwrap()
}
fn count_rows(c: &Connection) -> Vec<String> {
    c.prepare("SELECT 'session:'||quote(message_count)||':'||quote(thinking_count)||':'||quote(tool_call_count)||':'||quote(error_count) FROM session
        UNION ALL SELECT 'message:'||quote(pk)||':'||quote(id)||':'||quote(session_pk)||':'||quote(words) FROM message
        UNION ALL SELECT 'block:'||quote(pk)||':'||quote(body) FROM block
        UNION ALL SELECT 'call:'||quote(pk)||':'||quote(run_by_script)||':'||quote(action)||':'||quote(outcome) FROM tool_call ORDER BY 1").unwrap()
        .query_map([],|r|r.get(0)).unwrap().collect::<rusqlite::Result<_>>().unwrap()
}
fn previous(conn: &Connection, primary: bool, partial: bool) {
    create_index_in(conn, primary).unwrap();
    conn.execute_batch(PREVIOUS).unwrap();
    conn.execute_batch("INSERT INTO input_ref(pk,id) VALUES(1,'synthetic-input');
      INSERT INTO parse_state(input_pk,chain,reading_version,status,parsed_upto,records,used,commit_seq,checkpoint)
      VALUES(1,7,'synthetic-reading','parsing',42,3,3,19,x'0001ff');").unwrap();
    conn.execute_batch("INSERT INTO session(pk,id,origin,native_id,kind,cwd_json,changed) VALUES(1,'test:counts','test','counts','main','[]',19);
        INSERT INTO message(pk,id,session_pk,input_pk,session_order,role,written_by,words) VALUES(1,'test:message',1,1,'0001','assistant','model',3);").unwrap();
    for (pk, kind, body) in [
        (
            1,
            "thinking",
            serde_json::json!({"type":"thinking","text":null}),
        ),
        (
            2,
            "tool_call",
            serde_json::json!({"type":"tool_call","call_id":"call","tool":"run","action":"shell","input":{}}),
        ),
    ] {
        conn.execute(
            "INSERT INTO block(pk,message_pk,session_pk,idx,kind,body) VALUES(?1,1,1,?1,?2,?3)",
            rusqlite::params![pk, kind, crate::encoding::pack_json(&body).unwrap()],
        )
        .unwrap();
    }
    conn.execute("INSERT INTO tool_call(pk,session_pk,call_block_pk,call_id,tool,action,outcome,run_by_script) VALUES(1,1,2,'call','run','shell','error',1)",[]).unwrap();
    if partial {
        conn.execute("INSERT INTO meta(key,value)VALUES('building','1') ON CONFLICT(key)DO UPDATE SET value='1'",[]).unwrap();
        for name in crate::writer::deferred_index_names() {
            conn.execute_batch(&format!("DROP INDEX IF EXISTS {name}"))
                .unwrap();
        }
    } else {
        conn.execute("DELETE FROM meta WHERE key='building'", [])
            .unwrap();
    }
}
#[test]
fn count_trigger_transition_keeps_both_current_formats_and_partial_progress() {
    for primary in [false, true] {
        for partial in [false, true] {
            let c = Connection::open_in_memory().unwrap();
            previous(&c, primary, partial);
            let before = cursor(&c);
            let rows_before = count_rows(&c);
            let identity_before = generation(&c);
            let changes = c.total_changes();
            ensure_index_in(&c, primary).unwrap();
            assert_eq!(cursor(&c), before);
            assert_eq!(count_rows(&c), rows_before);
            assert_eq!(generation(&c), identity_before);
            assert_eq!(c.total_changes(), changes);
            assert_eq!(version(&c).unwrap(), index_version(primary));
            let identity = index_identity(&c).unwrap();
            assert_eq!(identity, build_index_identity(primary).unwrap());
            ensure_index_in(&c, primary).unwrap();
            assert_eq!(cursor(&c), before);
            assert_eq!(index_identity(&c).unwrap(), identity);
            assert_eq!(
                c.query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |r| r
                    .get::<_, i64>(
                    0
                ))
                .unwrap(),
                0
            );
        }
    }
}
#[test]
fn unknown_count_trigger_body_is_refused_without_mutation() {
    let c = Connection::open_in_memory().unwrap();
    previous(&c, false, false);
    c.execute_batch("DROP TRIGGER message_counts_added;CREATE TRIGGER message_counts_added AFTER INSERT ON message BEGIN SELECT 1;END;").unwrap();
    let before = shape(&c).unwrap();
    let progress = cursor(&c);
    let changed = c.total_changes();
    assert!(matches!(
        ensure_index(&c),
        Err(StoreError::NoMigration {
            database: "index",
            ..
        })
    ));
    assert_eq!(shape(&c).unwrap(), before);
    assert_eq!(cursor(&c), progress);
    assert_eq!(c.total_changes(), changed);
}
#[cfg(feature = "faults")]
#[test]
fn count_trigger_transition_rolls_back_then_retries() {
    let d = tempfile::tempdir().unwrap();
    let file = d.path().join("count-transition-rollback.db");
    let c = Connection::open(&file).unwrap();
    previous(&c, false, true);
    let before = shape(&c).unwrap();
    let progress = cursor(&c);
    crate::fault::arm(
        "index.session_counts_transition",
        "count-transition-rollback.db",
        crate::fault::Action::Fail,
        0,
        1,
    );
    assert!(ensure_index(&c).is_err());
    assert_eq!(shape(&c).unwrap(), before);
    assert_eq!(cursor(&c), progress);
    ensure_index(&c).unwrap();
    assert_eq!(cursor(&c), progress);
}
#[cfg(unix)]
#[test]
fn ordinary_open_advances_count_ownership_without_setting_aside_the_index() {
    use std::os::unix::fs::MetadataExt;
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("archive");
    let (a, lock) = crate::Archive::open_or_create(&root).unwrap();
    let path = a.index_db();
    let c = Connection::open(&path).unwrap();
    c.execute_batch(PREVIOUS).unwrap();
    c.execute_batch("INSERT INTO input_ref(pk,id)VALUES(1,'synthetic-input');INSERT INTO parse_state(input_pk,chain,reading_version,status,parsed_upto,records,used,checkpoint)VALUES(1,7,'synthetic-reading','parsing',42,3,3,x'0001ff');").unwrap();
    let progress = cursor(&c);
    let identity_before = generation(&c);
    drop(c);
    drop(a);
    drop(lock);
    let inode = std::fs::metadata(&path).unwrap().ino();
    let (a, _lock) = crate::Archive::open_or_create(&root).unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().ino(), inode);
    let c = Connection::open(a.index_db()).unwrap();
    assert_eq!(cursor(&c), progress);
    assert_eq!(generation(&c), identity_before);
    assert_eq!(
        index_identity(&c).unwrap(),
        build_index_identity(false).unwrap()
    );
    assert!(!std::fs::read_dir(&root).unwrap().any(|e| {
        e.unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with("index-set-aside-")
    }));
}
