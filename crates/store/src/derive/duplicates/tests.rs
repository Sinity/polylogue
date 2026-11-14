use super::*;

fn turns(keys: &[&str]) -> Vec<Turn> {
    keys.iter()
        .enumerate()
        .map(|(i, key)| Turn {
            prompt: (i as i64, format!("m{i}")),
            key: (*key).to_owned(),
            replies: Vec::new(),
        })
        .collect()
}

#[test]
fn a_prompt_repeated_in_either_route_is_never_paired() {
    let captured = turns(&["plan it", "continue", "continue", "thanks"]);
    let exported = turns(&["plan it", "continue", "thanks"]);
    assert_eq!(paired(&captured, &exported), [(0, 0), (3, 2)]);
}

#[test]
fn pairs_keep_one_order_in_both_routes() {
    let captured = turns(&["a", "b", "c", "d"]);
    let exported = turns(&["b", "a", "c", "", "d"]);
    assert_eq!(paired(&captured, &exported), [(1, 0), (2, 2), (3, 4)]);
}

#[test]
fn empty_prompts_are_never_paired() {
    assert!(paired(&turns(&[""]), &turns(&[""])).is_empty());
}

/// SQLite page reads, rather than preparation callbacks or elapsed time.
#[allow(unsafe_code)]
fn page_reads(conn: &Connection) -> i32 {
    let (mut current, mut peak) = (0, 0);
    // SAFETY: this connection owns the handle throughout this status read.
    let result = unsafe {
        rusqlite::ffi::sqlite3_db_status(
            conn.handle(),
            rusqlite::ffi::SQLITE_DBSTATUS_CACHE_MISS,
            &mut current,
            &mut peak,
            1,
        )
    };
    assert_eq!(result, rusqlite::ffi::SQLITE_OK);
    current
}

#[test]
fn unmarked_sessions_do_not_read_all_message_rows_and_withdrawn_capture_marks_clear() {
    use crate::fixture::{Fixture, SessionFacts};
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let sessions = ["gemini:withdrawn", "gemini:other"].map(|id| {
        fixture
            .session(SessionFacts {
                id,
                ..Default::default()
            })
            .unwrap()
    });
    let index = crate::archive::open_write(&fixture.archive.index_db(), 64).unwrap();
    let input: i64 = index
        .query_row("SELECT pk FROM input_ref LIMIT 1", [], |r| r.get(0))
        .unwrap();
    let attrs = format!("{{\"retained\":\"{}\"}}", "x".repeat(2048));
    for session in sessions {
        index
            .execute(
                "WITH RECURSIVE n(i) AS (VALUES(1) UNION ALL SELECT i+1 FROM n WHERE i<256)
             INSERT INTO message(id,session_pk,input_pk,session_order,role,written_by,attrs_json)
             SELECT printf('gemini:%d-%d',?1,i),?1,?2,printf('%08d',i),'user','human',?3 FROM n",
                params![session, input, attrs],
            )
            .unwrap();
    }
    drop(index);
    let conn = crate::archive::open_read(&fixture.archive.index_db()).unwrap();
    conn.pragma_update(None, "cache_size", -64).unwrap();
    conn.execute_batch("PRAGMA shrink_memory").unwrap();
    page_reads(&conn);
    // Independent old access path: identical rows, without the sparse index.
    let old: HashMap<i64, String> = conn
        .prepare(
            "SELECT pk,duplicate_of FROM message INDEXED BY message_session
         WHERE session_pk=?1 AND duplicate_of IS NOT NULL",
        )
        .unwrap()
        .query_map([sessions[0]], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    let old_reads = page_reads(&conn);
    assert!(old.is_empty());
    conn.execute_batch("PRAGMA shrink_memory").unwrap();
    page_reads(&conn);
    assert!(compute(&conn, sessions[0]).unwrap().is_empty());
    let current_reads = page_reads(&conn);
    assert!(
        old_reads > 128,
        "fixture must exceed its SQLite cache: {old_reads}"
    );
    assert!(
        current_reads * 8 < old_reads,
        "old={old_reads}, current={current_reads}"
    );

    let index = crate::archive::open_write(&fixture.archive.index_db(), 64).unwrap();
    let marked: Vec<i64> = index
        .prepare("SELECT pk FROM message WHERE session_pk=?1 ORDER BY pk LIMIT 2")
        .unwrap()
        .query_map([sessions[0]], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    index.execute("UPDATE message SET duplicate_of='gemini:original',duplicate_reason='capture' WHERE pk IN (?1,?2)",params![marked[0],marked[1]]).unwrap();
    index
        .execute(
            "UPDATE message SET duplicate_of='gemini:elsewhere' WHERE session_pk=?1",
            [sessions[1]],
        )
        .unwrap();
    // No current capture membership remains, but its old marks must be cleared.
    let expected: Vec<_> = marked.iter().map(|pk| (*pk, None)).collect();
    assert_eq!(compute(&conn, sessions[0]).unwrap(), expected);
    let tx = index.unchecked_transaction().unwrap();
    let changes = crate::change_summary::ChangeSummary::start(&tx).unwrap();
    apply(&tx, &expected, &changes).unwrap();
    tx.commit().unwrap();
    assert!(compute(&conn, sessions[0]).unwrap().is_empty());
    let remaining: i64 = index
        .query_row(
            "SELECT count(*) FROM message WHERE session_pk=?1 AND duplicate_of IS NOT NULL",
            [sessions[1]],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        remaining, 256,
        "clearing one session preserves the other's marks"
    );
}
