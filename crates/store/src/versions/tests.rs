use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

fn database(unrelated: i64) -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    track_unreferenced(&conn).unwrap();
    for id in 1..=unrelated {
        let text = format!("retained synthetic text {id}");
        conn.execute(
            "INSERT INTO text(id,hash,body) VALUES(?1,?2,?3)",
            params![
                id,
                blake3::hash(text.as_bytes()).as_bytes(),
                encoding::pack_text(&text).unwrap()
            ],
        )
        .unwrap();
    }
    conn
}

fn cleanup_steps(conn: &Connection) -> u64 {
    conn.flush_prepared_statement_cache();
    let steps = Arc::new(AtomicU64::new(0));
    let count = Arc::clone(&steps);
    conn.progress_handler(
        1,
        Some(move || {
            count.fetch_add(1, Ordering::Relaxed);
            false
        }),
    )
    .unwrap();
    remove_unreferenced(conn, true).unwrap();
    conn.progress_handler(0, None::<fn() -> bool>).unwrap();
    steps.load(Ordering::Relaxed)
}

#[test]
fn empty_text_cleanup_does_not_read_unrelated_history() {
    let small = cleanup_steps(&database(0));
    let large = cleanup_steps(&database(8192));
    assert!(
        large <= small + 32,
        "SQLite {}: empty cleanup {small} -> {large} VM steps",
        rusqlite::version()
    );
}

#[test]
fn text_cleanup_checks_only_candidates_and_keeps_all_reference_kinds() {
    let conn = database(8192);
    conn.execute_batch(
        "INSERT INTO input_ref(pk,id) VALUES(1,'synthetic');
         INSERT INTO session(pk,id,origin,native_id,kind,changed)
             VALUES(1,'claude-code:synthetic','claude-code','synthetic','main',1);
         INSERT INTO message(pk,id,session_pk,session_order,role,written_by,input_pk)
             VALUES(1,'claude-code:synthetic',1,'0001','user','human',1);
         INSERT INTO block(pk,message_pk,session_pk,idx,kind,body,text_id)
             VALUES(1,1,1,0,'text',x'00',1);
         INSERT INTO document(pk,id,origin,native_id,purpose,authorship,version_hash,input_pk,
             source_chain,source_start,source_end,fields,changed)
             VALUES(1,'synthetic','claude-code','synthetic','context','human',x'01',1,1,0,1,x'00',1);
         INSERT INTO document_block(document_pk,idx,kind,body,text_id) VALUES(1,0,'text',x'00',2);
         INSERT INTO message_version(version_hash,body) VALUES(x'01',x'00');
         INSERT INTO message_version_text(version_hash,idx,text_id) VALUES(x'01',0,3);
         INSERT INTO document_version(version_hash,body) VALUES(x'01',x'00');
         INSERT INTO document_version_text(version_hash,idx,text_id) VALUES(x'01',0,4);
         INSERT INTO temp.unreferenced_text VALUES(1),(2),(3),(4),(5),(9000);",
    ).unwrap();
    let steps = cleanup_steps(&conn);
    let missing: Vec<i64> = conn
        .prepare("SELECT id FROM text WHERE id<=6 ORDER BY id")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(missing, [1, 2, 3, 4, 6]);
    assert!(steps < 2000, "fixed six candidates used {steps} VM steps");
    assert_eq!(
        conn.query_row("SELECT count(*) FROM temp.unreferenced_text", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
}
