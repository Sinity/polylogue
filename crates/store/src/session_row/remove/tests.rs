use super::*;
use crate::encoding::PreparedMessage;
use polylogue_model::{Block, Message};
use std::collections::HashMap;

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
    tx.execute("INSERT INTO message(pk,id,session_pk,order_input_pk,order_input_id,seq,role,written_by,input_pk,parser_version)
        VALUES(?1,?2,?3,1,'first',?1,'user','human',1,1)",params![pk,format!("claude-code:m-{pk}"),session]).unwrap();
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
    for table in [
        "session_remove_text",
        "session_remove_blob",
        "session_remove_copy",
    ] {
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
fn removal_references_freeze_current_and_historical_unions_with_exact_order() {
    let mut db = database();
    let tx = db.transaction().unwrap();
    message(&tx, 1, 1);
    message(&tx, 2, 2);
    let current = text_block(&tx, 1, 1, 0, "current", &[2; 32]);
    let historical = text_block(&tx, 1, 1, 1, "historical", &[1; 32]);
    let from_source = text_block(&tx, 2, 2, 0, "source placement", &[3; 32]);
    for (pk, tid) in [(1, historical), (2, from_source)] {
        let hash = vec![pk as u8; 32];
        tx.execute(
            "INSERT INTO message_version(pk,hash,attrs) VALUES(?1,?2,x'00')",
            params![pk, hash],
        )
        .unwrap();
        tx.execute(
            "INSERT INTO version_block VALUES(?1,0,x'00',?2,?3)",
            params![pk, tid, hash],
        )
        .unwrap();
        tx.execute("INSERT INTO message_source(message_pk,input_pk,chain,session_pk,parser_version,source_start,source_end,preferred,version_hash)
            VALUES(?1,1,1,1,1,0,1,0,?2)",params![pk,hash]).unwrap();
    }
    let refs = References::capture(&tx, 1).unwrap();
    tx.execute_batch("DELETE FROM message_source; DELETE FROM version_block; DELETE FROM message_version; DELETE FROM block").unwrap();
    let mut texts = Vec::new();
    refs.visit_texts(|id| {
        texts.push(id);
        Ok(())
    })
    .unwrap();
    assert_eq!(texts, [current, historical, from_source]);
    let mut hashes = Vec::new();
    refs.visit_blobs(|hash| {
        hashes.push(hash.to_vec());
        Ok(())
    })
    .unwrap();
    // Current and historical references share hashes; the union stays unique.
    assert_eq!(hashes, [vec![1; 32], vec![2; 32]]);
    refs.finish().unwrap();
    assert_eq!(staged_rows(&tx), 0);
}

#[test]
fn ordinary_removal_skips_unused_reference_tables_and_cleans_copy_state() {
    let mut db = database();
    let tx = db.transaction().unwrap();
    // Existing sentinel tables would make reference capture fail.
    tx.execute_batch(
        "CREATE TEMP TABLE session_remove_text(sentinel TEXT);
        CREATE TEMP TABLE session_remove_blob(sentinel TEXT);
        INSERT INTO session_remove_text VALUES('unchanged');",
    )
    .unwrap();
    remove(&tx, 1, "claude-code:removed", 2).unwrap();
    assert_eq!(
        tx.query_row("SELECT sentinel FROM temp.session_remove_text", [], |r| {
            r.get::<_, String>(0)
        })
        .unwrap(),
        "unchanged"
    );
    assert_eq!(staged_rows(&tx), 1);
    tx.execute_batch("DROP TABLE temp.session_remove_text; DROP TABLE temp.session_remove_blob")
        .unwrap();
    remove_with_references(&tx, 2, "claude-code:survivor", 2, |refs| {
        refs.visit_texts(|_| panic!("empty session"))?;
        refs.visit_blobs(|_| panic!("empty session"))
    })
    .unwrap();
    assert_eq!(staged_rows(&tx), 0);
    tx.commit().unwrap();
}

#[test]
fn removal_callback_failure_rolls_back_before_a_successful_retry() {
    let mut db = database();
    let tx = db.transaction().unwrap();
    message(&tx, 1, 1);
    let tid = text_block(&tx, 1, 1, 0, "preserved until commit", &[7; 32]);
    tx.commit().unwrap();
    let tx = db.transaction().unwrap();
    let error = remove_with_references(&tx, 1, "claude-code:removed", 2, |refs| {
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
    remove_with_references(&tx, 1, "claude-code:removed", 2, |refs| {
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
fn removal_consumer_sees_surviving_copy_promotion_before_orphan_checks() {
    let mut db = database();
    let tx = db.transaction().unwrap();
    message(&tx, 2, 2);
    let mut versions = Vec::new();
    for text in ["old retained body", "surviving promoted body"] {
        let mut msg: Message = serde_json::from_value(serde_json::json!({
            "id":"claude-code:m-2","session":"claude-code:survivor","role":"user",
            "prompt_kind":"prose","blocks":[],"usage":{}
        }))
        .unwrap();
        msg.blocks.push(Block::Text { text: text.into() });
        let packed = PreparedMessage::new(&msg).unwrap();
        crate::versions::retain(&tx, &packed, &msg, &HashMap::new(), false).unwrap();
        versions.push(packed);
    }
    let old = text_block(&tx, 2, 2, 0, "old retained body", &[4; 32]);
    for (input, session, preferred, version) in [(1, 1, 1, &versions[0]), (2, 2, 0, &versions[1])] {
        tx.execute("INSERT INTO message_source(message_pk,input_pk,chain,session_pk,parser_version,source_start,source_end,preferred,version_hash)
            VALUES(2,?1,1,?2,1,0,1,?3,?4)",params![input,session,preferred,&version.hash[..]]).unwrap();
    }
    remove_with_references(&tx,1,"claude-code:removed",2,|refs|{
        let mut captured=Vec::new();refs.visit_texts(|id|{captured.push(id);Ok(())})?;
        assert_eq!(captured,[old]);
        let body:Vec<u8>=tx.query_row("SELECT t.body FROM block b JOIN text t ON t.id=b.text_id WHERE b.message_pk=2",[],|r|r.get(0))?;
        assert_eq!(crate::encoding::unpack_text(&body)?,"surviving promoted body");
        let owns_old:bool=tx.query_row("SELECT EXISTS(SELECT 1 FROM block WHERE text_id=?1) OR EXISTS(SELECT 1 FROM version_block WHERE text_id=?1)",[old],|r|r.get(0))?;
        assert!(!owns_old);
        Ok(())
    }).unwrap();
    assert_eq!(staged_rows(&tx), 0);
    tx.commit().unwrap();
    crate::text_rows::index_pending(&db).unwrap();
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM text_fts WHERE text_fts MATCH 'promoted'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        1
    );
    assert_eq!(staged_rows(&db), 0);
}

#[test]
fn removing_a_copys_only_source_marks_it_absent() {
    let mut db = database();
    let tx = db.transaction().unwrap();
    message(&tx, 2, 2);
    text_block(&tx, 2, 2, 0, "no surviving source", &[3; 32]);
    tx.execute("INSERT INTO message_source(message_pk,input_pk,chain,session_pk,parser_version,source_start,source_end,preferred)
        VALUES(2,1,1,1,1,0,1,1)",[]).unwrap();
    remove(&tx, 1, "claude-code:removed", 2).unwrap();
    assert_eq!(
        tx.query_row("SELECT present FROM message WHERE pk=2", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        tx.query_row("SELECT count(*) FROM block WHERE message_pk=2", [], |r| r
            .get::<_, i64>(
            0
        ))
        .unwrap(),
        0
    );
    assert_eq!(staged_rows(&tx), 0);
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
    remove(&tx, 1, "claude-code:removed", 2).unwrap();
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
        remove_with_references(&tx, pk, id, 2, |_| Ok(())).unwrap();
        assert_eq!(staged_rows(&tx), 0);
    }
    drop(rows);
    drop(statement);
    tx.commit().unwrap();
}
