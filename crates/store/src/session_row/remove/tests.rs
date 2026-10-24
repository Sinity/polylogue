use super::*;

fn database() -> Connection {
    let db = Connection::open_in_memory().unwrap();
    crate::schema::create_index(&db).unwrap();
    db.pragma_update(None, "foreign_keys", true).unwrap();
    db.execute_batch(
        "INSERT INTO input_ref(pk,id) VALUES(1,'first'),(2,'second');
         INSERT INTO session(pk,id,origin,native_id,kind,changed) VALUES
             (1,'claude-code:removed','claude-code','removed','main',1),
             (2,'claude-code:survivor','claude-code','survivor','main',1);",
    )
    .unwrap();
    db
}

fn message(tx: &Transaction<'_>, pk: i64, session: i64) {
    tx.execute("INSERT INTO message(pk,id,session_pk,order_input_pk,order_input_id,seq,role,written_by,input_pk)
        VALUES(?1,?2,?3,1,'first',?1,'user','human',1)",params![pk,format!("claude-code:m-{pk}"),session]).unwrap();
}

fn text_block(
    tx: &Transaction<'_>,
    message: i64,
    session: i64,
    idx: i64,
    text: &str,
    hash: &[u8],
) -> i64 {
    let tid = crate::text_rows::intern(tx, text, 0, None, false).unwrap();
    tx.execute(
        "INSERT INTO block(message_pk,session_pk,idx,kind,text_id,body,blob_hash)
        VALUES(?1,?2,?3,'text',?4,x'00',?5)",
        params![message, session, idx, tid, hash],
    )
    .unwrap();
    tid
}

fn staged_rows(db: &Connection) -> i64 {
    let mut count = 0;
    for table in ["session_remove_text", "session_remove_blob"] {
        let exists: bool = db
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_temp_master WHERE name=?1)",
                [table],
                |r| r.get(0),
            )
            .unwrap();
        if exists {
            count += db
                .query_row(&format!("SELECT count(*) FROM temp.{table}"), [], |r| {
                    r.get::<_, i64>(0)
                })
                .unwrap();
        }
    }
    count
}

#[test]
fn removal_callback_failure_rolls_back_before_a_successful_retry() {
    let mut db = database();
    let tx = db.transaction().unwrap();
    message(&tx, 1, 1);
    let tid = text_block(&tx, 1, 1, 0, "preserved until commit", &[7; 32]);
    tx.commit().unwrap();
    let tx = db.transaction().unwrap();
    let error = remove_with_references(&tx, 1, "claude-code:removed", |refs| {
        let mut texts = Vec::new();
        refs.visit_texts(|id| {
            texts.push(id);
            Ok(())
        })?;
        assert_eq!(texts, [tid]);
        assert_eq!(
            tx.query_row("SELECT count(*) FROM session WHERE pk=1", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
        Err(crate::StoreError::Invalid("consumer failed".into()))
    })
    .unwrap_err();
    assert!(matches!(error,crate::StoreError::Invalid(s) if s=="consumer failed"));
    tx.rollback().unwrap();
    assert_eq!(staged_rows(&db), 0);
    assert_eq!(
        db.query_row("SELECT count(*) FROM block", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    let tx = db.transaction().unwrap();
    remove_with_references(&tx, 1, "claude-code:removed", |refs| {
        refs.visit_blobs(|hash| {
            assert_eq!(hash, &[7; 32]);
            Ok(())
        })
    })
    .unwrap();
    assert_eq!(staged_rows(&tx), 0);
    tx.commit().unwrap();
}

#[test]
fn many_tool_calls_remove_exact_search_entries_and_keep_another_session() {
    let mut db = database();
    let tx = db.transaction().unwrap();
    message(&tx, 1, 1);
    message(&tx, 2, 2);
    for pk in 1..=4097 {
        let session = if pk == 4097 { 2 } else { 1 };
        tx.execute("INSERT INTO block(pk,message_pk,session_pk,idx,kind,body) VALUES(?1,?2,?2,?1,'tool_call',x'00')",params![pk,session]).unwrap();
        let command = format!("echo syntheticcommand{pk}");
        let path = format!("/synthetic/path/{pk}");
        tx.execute("INSERT INTO tool_call(pk,session_pk,call_block_pk,tool,command,path,outcome) VALUES(?1,?2,?1,'shell',?3,?4,'unknown')",params![pk,session,command,path]).unwrap();
        tx.execute(
            "INSERT INTO tool_call_trigram(rowid,command,path) VALUES(?1,?2,?3)",
            params![pk, command, path],
        )
        .unwrap();
    }
    remove_with_references(&tx, 1, "claude-code:removed", |_| Ok(())).unwrap();
    assert_eq!(
        tx.query_row("SELECT count(*) FROM tool_call", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        tx.query_row("SELECT count(*) FROM tool_call_trigram", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        1
    );
    assert_eq!(tx.query_row("SELECT rowid FROM tool_call_trigram WHERE tool_call_trigram MATCH 'syntheticcommand4097'",[],|r|r.get::<_,i64>(0)).unwrap(),4097);
    assert_eq!(staged_rows(&tx), 0);
}

#[test]
fn removal_reuses_empty_staging_while_an_outer_temp_cursor_remains_open() {
    let mut db = database();
    let tx = db.transaction().unwrap();
    tx.execute_batch(
        "CREATE TEMP TABLE outer_selection(pk INTEGER); INSERT INTO outer_selection VALUES(1),(2)",
    )
    .unwrap();
    let mut statement = tx
        .prepare("SELECT pk FROM outer_selection ORDER BY pk")
        .unwrap();
    let mut rows = statement.query([]).unwrap();
    while let Some(row) = rows.next().unwrap() {
        let pk: i64 = row.get(0).unwrap();
        let id = if pk == 1 {
            "claude-code:removed"
        } else {
            "claude-code:survivor"
        };
        remove_with_references(&tx, pk, id, |_| Ok(())).unwrap();
        assert_eq!(staged_rows(&tx), 0);
    }
    drop(rows);
    drop(statement);
    tx.commit().unwrap();
}
