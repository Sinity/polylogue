use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

fn insert_version(conn: &Connection, hash: &[u8], body: &[u8]) -> i64 {
    let pk: i64 = conn
        .query_row(
            "INSERT INTO message_version(version_hash) VALUES(?1) RETURNING pk",
            [hash],
            |r| r.get(0),
        )
        .unwrap();
    conn.execute(
        "INSERT INTO message_version_body VALUES(?1,?2)",
        params![pk, body],
    )
    .unwrap();
    pk
}

fn database(unrelated: i64) -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&conn).unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    track_unreferenced(&conn).unwrap();
    for id in 1..=unrelated {
        let text = format!("retained synthetic text {id}");
        conn.execute(
            "INSERT INTO text(id,hash,normalized_hash,body) VALUES(?1,?2,?2,?3)",
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
    let identities = crate::text_rows::TextIds::new(conn, true).unwrap();
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
    remove_unreferenced(conn, Some(&FirstBuild::default()), &identities).unwrap();
    conn.progress_handler(0, None::<fn() -> bool>).unwrap();
    steps.load(Ordering::Relaxed)
}

fn context_reference_cleanup_steps(unrelated: i64, without_index: bool) -> u64 {
    let conn = database(0);
    if without_index {
        conn.execute_batch("DROP INDEX IF EXISTS parser_context_version")
            .unwrap();
    }
    conn.execute_batch(
        "INSERT INTO input_ref(pk,id) VALUES(1,'synthetic');
         INSERT INTO session(pk,id,origin,native_id,kind,changed)
             VALUES(1,'codex:synthetic','codex','synthetic','main',1);
         INSERT INTO message(pk,id,session_pk,session_order,role,written_by,input_pk)
             VALUES(1,'codex:response',1,'0001','assistant','model',1);",
    )
    .unwrap();
    let orphan = [0u8; 32];
    let winner = [255u8; 32];
    for hash in [orphan, winner] {
        insert_version(&conn, &hash, &[0]);
    }
    for id in 1..=unrelated + 1 {
        let mut hash = [0u8; 32];
        hash[..8].copy_from_slice(&id.to_le_bytes());
        insert_version(&conn, &hash, &[0]);
        conn.execute(
            "INSERT INTO parser_context VALUES(1,7,'synthetic',?1,?2,x'00',?3)",
            params![
                id,
                format!("context-{id}"),
                message_key(&conn, &hash).unwrap()
            ],
        )
        .unwrap();
    }
    conn.execute("INSERT INTO message_source(message_pk,input_pk,chain,source_start,source_end,id_kind,session_pk,route,completion,version_pk) VALUES(1,1,7,0,1,'native',1,'live','finished',(SELECT pk FROM message_version WHERE version_hash=?1))",[orphan]).unwrap();
    let mut protected = [0u8; 32];
    protected[..8].copy_from_slice(&1i64.to_le_bytes());
    conn.execute_batch("SAVEPOINT reading").unwrap();
    // Actual source-reference changes queue both versions for collection.
    for hash in [protected, winner] {
        conn.execute(
            "UPDATE message_source SET version_pk=(SELECT pk FROM message_version WHERE version_hash=?1) WHERE message_pk=1",
            [hash],
        )
        .unwrap();
    }
    let steps = cleanup_steps(&conn);
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM message_version WHERE version_hash=?1",
            [orphan],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM message_version WHERE version_hash=?1",
            [protected],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        1
    );
    assert_eq!(
        conn.query_row("SELECT count(*) FROM parser_context", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        unrelated + 1
    );
    assert_eq!(
        conn.query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |r| r
            .get::<_, i64>(
            0
        ))
        .unwrap(),
        0
    );
    conn.execute_batch("ROLLBACK TO reading; RELEASE reading")
        .unwrap();
    assert_eq!(
        conn.query_row(
            "SELECT v.version_hash FROM message_source s JOIN message_version v ON v.pk=s.version_pk WHERE message_pk=1",
            [],
            |r| r.get::<_, Vec<u8>>(0)
        )
        .unwrap(),
        orphan
    );
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM message_version WHERE version_hash=?1",
            [orphan],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        1
    );
    assert_eq!(
        conn.query_row("SELECT count(*) FROM temp.unreferenced_value", [], |r| {
            r.get::<_, i64>(0)
        })
        .unwrap(),
        0
    );
    steps
}

#[test]
fn version_cleanup_checks_context_references_without_scanning_history() {
    let small = context_reference_cleanup_steps(0, false);
    let large = context_reference_cleanup_steps(8192, false);
    assert!(
        large <= small + 64,
        "SQLite {}: candidate version cleanup {small} -> {large} VM steps",
        rusqlite::version()
    );
    let unindexed = context_reference_cleanup_steps(8192, true);
    assert!(
        unindexed >= 8192 * 4,
        "missing-index positive control used only {unindexed} steps"
    );
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
         INSERT INTO message_version(pk,version_hash) VALUES(1,x'01'); INSERT INTO message_version_body VALUES(1,x'00');
         INSERT INTO message_version_text(version_pk,idx,text_id) VALUES(1,0,3);
         INSERT INTO document_version(version_hash,body) VALUES(x'01',x'00');
         INSERT INTO document_version_text(version_hash,idx,text_id) VALUES(x'01',0,4);
         INSERT INTO temp.unreferenced_text VALUES(1),(2),(3),(4),(5),(9000);",
    ).unwrap();
    let steps = cleanup_steps(&conn);
    let retained: Vec<i64> = conn
        .prepare("SELECT id FROM text WHERE id<=6 ORDER BY id")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(retained, [1, 2, 3, 4, 6]);
    assert!(steps < 2000, "fixed six candidates used {steps} VM steps");
    assert_eq!(
        conn.query_row("SELECT count(*) FROM temp.unreferenced_text", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn removing_the_last_text_reference_and_its_cleanup_roll_back_together() {
    let conn = database(1);
    conn.execute_batch(
        "INSERT INTO message_version(pk,version_hash) VALUES(1,x'01'); INSERT INTO message_version_body VALUES(1,x'00');
         INSERT INTO message_version_text(version_pk,idx,text_id) VALUES(1,0,1);
         SAVEPOINT reading;
         DELETE FROM message_version_text;",
    )
    .unwrap();
    assert_eq!(
        conn.query_row("SELECT count(*) FROM temp.unreferenced_text", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    remove_unreferenced(
        &conn,
        Some(&FirstBuild::default()),
        &crate::text_rows::TextIds::new(&conn, true).unwrap(),
    )
    .unwrap();
    assert_eq!(
        conn.query_row("SELECT count(*) FROM text", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    conn.execute_batch("ROLLBACK TO reading; RELEASE reading;")
        .unwrap();
    assert_eq!(
        conn.query_row("SELECT text_id FROM message_version_text", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        conn.query_row("SELECT count(*) FROM text", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        conn.query_row("SELECT count(*) FROM temp.unreferenced_text", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn first_build_cleanup_retains_version_text_and_forgets_deleted_identity() {
    let mut conn = database(0);
    let first = FirstBuild::load(&conn).unwrap();
    let identities = crate::text_rows::TextIds::new(&conn, true).unwrap();
    let hash = [17; 32];
    let version = [18; 32];
    let texts = identities.transaction();
    let tx = conn.transaction().unwrap();
    let id = identities
        .resolve(
            &tx,
            &hash,
            Some(&crate::encoding::PreparedText::new(
                "kept",
                b"kept".to_vec(),
            )),
        )
        .unwrap()
        .id;
    let version_pk = insert_version(&tx, &version, &[0]);
    tx.execute(
        "INSERT INTO message_version_text(version_pk,idx,text_id) VALUES(?1,0,?2)",
        params![version_pk, id],
    )
    .unwrap();
    tx.execute(
        "INSERT INTO temp.unreferenced_text(text_id) VALUES(?1)",
        [id],
    )
    .unwrap();
    remove_unreferenced(&tx, Some(&first), &identities).unwrap();
    assert_eq!(identities.resolve(&tx, &hash, None).unwrap().id, id);
    tx.execute(
        "DELETE FROM message_version WHERE version_hash=?1",
        [version.as_slice()],
    )
    .unwrap();
    remove_unreferenced(&tx, Some(&first), &identities).unwrap();
    assert!(identities.resolve(&tx, &hash, None).is_err());
    let replacement = identities
        .resolve(
            &tx,
            &hash,
            Some(&crate::encoding::PreparedText::new(
                "again",
                b"again".to_vec(),
            )),
        )
        .unwrap()
        .id;
    assert_eq!(
        tx.query_row(
            "SELECT count(*) FROM text WHERE id=?1 AND hash=?2",
            params![replacement, hash.as_slice()],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        1
    );
    tx.commit().unwrap();
    texts.commit();
    assert_eq!(
        identities.resolve(&conn, &hash, None).unwrap().id,
        replacement
    );
}

#[test]
fn exact_version_keys_and_payload_children_survive_failed_removal() {
    let conn = database(1);
    let mut hashes = [[42u8; 32]; 2];
    hashes[1][31] = 43; // Same long prefix must not alias a different canonical hash.
    let bodies = [vec![5u8; 65536], vec![6u8; 65536]];
    let keys = [
        insert_version(&conn, &hashes[0], &bodies[0]),
        insert_version(&conn, &hashes[1], &bodies[1]),
    ];
    assert_ne!(keys[0], keys[1]);
    assert!(
        conn.execute(
            "INSERT INTO message_version(version_hash) VALUES(?1)",
            [hashes[0]]
        )
        .is_err()
    );
    conn.execute_batch("INSERT INTO input_ref(pk,id) VALUES(1,'input');
        INSERT INTO session(pk,id,origin,native_id,kind,changed) VALUES(1,'codex:session','codex','session','main',1);
        INSERT INTO message(pk,id,session_pk,session_order,role,written_by,input_pk) VALUES(1,'codex:message',1,'0','assistant','model',1);").unwrap();
    conn.execute("INSERT INTO message_source(message_pk,input_pk,chain,source_start,source_end,id_kind,session_pk,route,completion,version_pk) VALUES(1,1,1,4,91,'native',1,'live','finished',?1)",[keys[0]]).unwrap();
    conn.execute(
        "INSERT INTO message_projection VALUES(?1,?2,7)",
        params![keys[0], serde_json::to_vec(&(9u64, vec![2u64, 8])).unwrap()],
    )
    .unwrap();
    conn.execute("INSERT INTO message_version_text VALUES(?1,2,1)", [keys[0]])
        .unwrap();
    conn.execute(
        "INSERT INTO message_version_blob VALUES(?1,?2)",
        params![keys[0], [8u8; 32]],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO parser_context VALUES(1,1,'reader',91,'response',x'00',?1)",
        [keys[0]],
    )
    .unwrap();
    assert!(
        conn.execute("INSERT INTO message_version_text VALUES(999999,0,1)", [])
            .is_err()
    );
    let snapshot = || {
        conn.query_row("SELECT v.version_hash,b.body,s.source_start,s.source_end,p.layout,p.admitted FROM message_source s JOIN message_version v ON v.pk=s.version_pk JOIN message_version_body b ON b.version_pk=v.pk JOIN message_projection p ON p.version_pk=v.pk",[],|r|Ok((r.get::<_,Vec<u8>>(0)?,r.get::<_,Vec<u8>>(1)?,r.get::<_,i64>(2)?,r.get::<_,i64>(3)?,r.get::<_,Vec<u8>>(4)?,r.get::<_,i64>(5)?))).unwrap()
    };
    let original = snapshot();
    assert_eq!(original.0, hashes[0]);
    assert_eq!(original.1, bodies[0]);
    assert_eq!((original.2, original.3), (4, 91));
    conn.execute_batch("SAVEPOINT change").unwrap();
    conn.execute("UPDATE message_source SET version_pk=?1", [keys[1]])
        .unwrap();
    let ids = crate::text_rows::TextIds::new(&conn, true).unwrap();
    remove_unreferenced(&conn, Some(&FirstBuild::default()), &ids).unwrap();
    assert_eq!(
        conn.query_row(
            "SELECT count(*) FROM message_version_body WHERE version_pk=?1",
            [keys[0]],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        1,
        "retained parser context owns the old payload"
    );
    conn.execute_batch("DELETE FROM parser_context").unwrap();
    remove_unreferenced(&conn, Some(&FirstBuild::default()), &ids).unwrap();
    for table in [
        "message_version_body",
        "message_projection",
        "message_version_text",
        "message_version_blob",
    ] {
        assert_eq!(
            conn.query_row(
                &format!("SELECT count(*) FROM {table} WHERE version_pk=?1"),
                [keys[0]],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
            0
        );
    }
    conn.execute_batch("ROLLBACK TO change; RELEASE change")
        .unwrap();
    assert_eq!(snapshot(), original);
    assert_eq!(
        conn.query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |r| r
            .get::<_, i64>(
            0
        ))
        .unwrap(),
        0
    );
}

#[test]
fn a_missing_version_payload_is_an_error_rather_than_an_absent_message() {
    let conn = database(0);
    let hash = [19u8; 32];
    conn.execute(
        "INSERT INTO message_version(version_hash) VALUES(?1)",
        [hash],
    )
    .unwrap();
    assert!(
        matches!(read_message(&conn,&hash),Err(crate::StoreError::Invalid(reason)) if reason=="stored message version has no payload")
    );
}
