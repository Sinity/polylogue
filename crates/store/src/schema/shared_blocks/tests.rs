use super::*;
use polylogue_model::{Message, bundle::MessageProjection};
use serde_json::json;

fn old_index(
    primary: bool,
    lineage: bool,
    building: bool,
    sparse: bool,
) -> (Connection, Message, MessageProjection) {
    let conn = predecessor(primary, lineage).unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    conn.pragma_update(None, "application_id", FORMAT_ID)
        .unwrap();
    set_version(&conn, 1).unwrap();
    conn.execute_batch(
        "INSERT INTO meta VALUES('generation','retained-generation'),('cursor_key','retained-key');
         INSERT INTO input_ref(pk,id,format) VALUES(1,'synthetic-input','claude-code');
         INSERT INTO session(pk,id,origin,native_id,kind,changed) VALUES(7,'claude-code:session','claude-code','session','main',19);
         INSERT INTO message(pk,id,session_pk,session_order,role,written_by,input_pk,source_chain,source_start,source_end)
         VALUES(17,'claude-code:message',7,'order','assistant','model',1,3,4,555);",
    ).unwrap();
    let message: Message = serde_json::from_value(json!({
        "id":"claude-code:message","session":"claude-code:session","role":"assistant",
        "blocks":[
            {"type":"thinking","text":"retained reasoning","signature":"opaque-signature"},
            {"type":"tool_call","tool":"shell","call_id":"call:1","input":{"command":"true","extra":[null,42]}},
            {"type":"tool_result","call_id":"call:1","output":"retained output","outcome":{"kind":"ok"},"data":{"structured":[true,{"value":"retained"}]}}
        ],"attrs":{"unknown":{"retained":true}}
    })).unwrap();
    let hash = message.content_hash().unwrap();
    let projection = MessageProjection {
        source_version: hash,
        block_count: if sparse { 30 } else { 3 },
        block_ordinals: if sparse {
            vec![2, 11, 27]
        } else {
            vec![0, 1, 2]
        },
    };
    let mut packer = RowPacker::new().unwrap();
    let body = packer.pack_version(&message, &message.blocks).unwrap();
    conn.execute(
        "INSERT INTO message_version(version_hash,body) VALUES(?1,?2)",
        params![hash.0.as_slice(), body],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO message_projection VALUES(?1,?2,44)",
        params![
            hash.0.as_slice(),
            crate::versions::message_layout(&projection).unwrap()
        ],
    )
    .unwrap();
    conn.execute("INSERT INTO message_source(message_pk,input_pk,chain,source_start,source_end,id_kind,session_pk,version_hash,route,completion) VALUES(17,1,3,4,555,'native',7,?1,'live','finished')",[hash.0.as_slice()]).unwrap();
    for (idx, block) in message.blocks.iter().enumerate() {
        let (body, data) = packer.pack_block_payload(block).unwrap();
        let text_id = encoding::block_text(block).map(|text| {
            let id = idx as i64 + 33;
            conn.execute(
                "INSERT INTO text(id,hash,body) VALUES(?1,?2,?3)",
                params![
                    id,
                    blake3::hash(text.as_bytes()).as_bytes(),
                    encoding::pack_text(text).unwrap()
                ],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO message_version_text VALUES(?1,?2,?3)",
                params![hash.0.as_slice(), idx as i64, id],
            )
            .unwrap();
            id
        });
        let signature = match block {
            Block::Thinking { signature, .. } => signature.as_deref(),
            _ => None,
        };
        conn.execute("INSERT INTO block(pk,message_pk,session_pk,idx,kind,text_id,body,data,signature,ordinal) VALUES(?1,17,7,?2,?3,?4,?5,?6,?7,?8)",params![idx as i64+21,idx as i64,if idx==0 {"thinking"} else if idx==1 {"tool_call"} else {"tool_result"},text_id,body,data,signature,projection.block_ordinals[idx].to_string()]).unwrap();
    }
    conn.execute_batch(
        "INSERT INTO tool_call(pk,session_pk,call_block_pk,result_block_pk,tool,outcome) VALUES(41,7,22,23,'shell','ok');
         INSERT INTO commit_link VALUES(7,'synthetic-commit',22,'commit-output');
         INSERT INTO parser_context_state VALUES(1,3,'retained-reading',555);
         INSERT INTO parse_state(input_pk,chain,status,parsed_upto,reading_version,checkpoint) VALUES(1,3,'done',555,'retained-reading',x'010203');",
    ).unwrap();
    conn.execute("INSERT INTO parser_context VALUES(1,3,'retained-reading',555,'retained-response',x'040506',?1)",[hash.0.as_slice()]).unwrap();
    if building {
        conn.execute("INSERT INTO meta VALUES('building','1')", [])
            .unwrap();
        for name in crate::writer::deferred_index_names() {
            conn.execute_batch(&format!("DROP INDEX IF EXISTS {name}"))
                .unwrap();
        }
    }
    (conn, message, projection)
}

fn retained(conn: &Connection) -> Vec<(String, String)> {
    let mut rows = conn
        .prepare("SELECT key,value FROM meta ORDER BY key")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    for (name, sql) in [
        (
            "checkpoint",
            "SELECT input_pk||':'||chain||':'||parsed_upto||':'||reading_version||':'||hex(checkpoint) FROM parse_state",
        ),
        (
            "context",
            "SELECT input_pk||':'||chain||':'||reading_version||':'||cursor||':'||key||':'||hex(value)||':'||hex(version_hash) FROM parser_context",
        ),
        (
            "call",
            "SELECT pk||':'||call_block_pk||':'||result_block_pk FROM tool_call",
        ),
        (
            "commit",
            "SELECT session_pk||':'||commit_hash||':'||evidence_block_pk FROM commit_link",
        ),
        (
            "blocks",
            "SELECT group_concat(pk||':'||message_pk||':'||session_pk||':'||idx||':'||ordinal,';') FROM block ORDER BY pk",
        ),
    ] {
        rows.push((name.into(), conn.query_row(sql, [], |r| r.get(0)).unwrap()));
    }
    rows
}

#[test]
fn shared_block_conversion_preserves_readings_references_and_progress() {
    for primary in [false, true] {
        for lineage in [false, true] {
            for building in [false, true] {
                for sparse in [false, true] {
                    let (conn, message, projection) = old_index(primary, lineage, building, sparse);
                    let before = retained(&conn);
                    let ensure = if primary {
                        ensure_primary_index
                    } else {
                        ensure_index
                    };
                    for _ in 0..2 {
                        ensure(&conn).unwrap();
                        assert_eq!(version(&conn).unwrap(), 2);
                        assert_eq!(retained(&conn), before);
                        assert_eq!(building, super::super::building(&conn).unwrap());
                        let (read, layout) = crate::versions::read_message(
                            &conn,
                            projection.source_version.0.as_slice(),
                        )
                        .unwrap()
                        .unwrap();
                        assert_eq!(read, message);
                        assert_eq!(layout, projection);
                        assert_eq!(
                            read.content_hash().unwrap(),
                            message.content_hash().unwrap()
                        );
                        assert_eq!(
                            conn.query_row("SELECT count(*) FROM block_value", [], |r| r
                                .get::<_, i64>(0))
                                .unwrap(),
                            3
                        );
                        assert_eq!(conn.query_row("SELECT count(*) FROM block b JOIN message_version_block v ON v.value_pk=b.value_pk AND v.idx=b.idx",[],|r|r.get::<_,i64>(0)).unwrap(),3);
                        assert_eq!(
                            conn.query_row(
                                "SELECT count(*) FROM pragma_foreign_key_check",
                                [],
                                |r| r.get::<_, i64>(0)
                            )
                            .unwrap(),
                            0
                        );
                        assert_eq!(
                            conn.query_row("PRAGMA foreign_keys", [], |r| r.get::<_, i64>(0))
                                .unwrap(),
                            1
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn shared_block_conversion_keeps_inline_selected_text() {
    let (conn, _, _) = old_index(false, false, false, false);
    let block: Block = serde_json::from_value(json!({"type":"tool_result","call_id":"call:1","output":"retained inline output","outcome":{"kind":"ok"}})).unwrap();
    let packed = encoding::pack_json(&block).unwrap();
    conn.execute(
        "UPDATE block SET text_id=NULL,body=?1,data=NULL WHERE pk=23",
        [packed],
    )
    .unwrap();
    ensure_index(&conn).unwrap();
    let packed: Vec<u8> = conn
        .query_row(
            "SELECT v.body FROM block b JOIN block_value v ON v.pk=b.value_pk WHERE b.pk=23",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(encoding::unpack_block(&packed).unwrap(), block);
    assert!(
        conn.query_row("SELECT text_id IS NULL FROM block WHERE pk=23", [], |row| {
            row.get::<_, bool>(0)
        })
        .unwrap()
    );
    assert_eq!(
        conn.query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |row| {
            row.get::<_, i64>(0)
        })
        .unwrap(),
        0
    );
}

#[test]
#[cfg(feature = "faults")]
fn shared_block_conversion_rolls_back_before_format_commit() {
    let (conn, message, projection) = old_index(false, false, true, false);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shared-block-transition.db");
    conn.execute("VACUUM INTO ?1", [path.to_str().unwrap()])
        .unwrap();
    drop(conn);
    let conn = Connection::open(&path).unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    let before = retained(&conn);
    let shape_before = shape(&conn).unwrap();
    let body: Vec<u8> = conn
        .query_row("SELECT body FROM message_version", [], |r| r.get(0))
        .unwrap();
    crate::fault::arm(
        "index.shared_blocks.converted",
        conn.path().unwrap(),
        crate::fault::Action::Fail,
        0,
        1,
    );
    assert!(ensure_index(&conn).is_err());
    assert_eq!(version(&conn).unwrap(), 1);
    assert_eq!(shape(&conn).unwrap(), shape_before);
    assert_eq!(retained(&conn), before);
    assert_eq!(
        conn.query_row("SELECT body FROM message_version", [], |r| r
            .get::<_, Vec<u8>>(0))
            .unwrap(),
        body
    );
    ensure_index(&conn).unwrap();
    assert_eq!(
        crate::versions::read_message(&conn, projection.source_version.0.as_slice())
            .unwrap()
            .unwrap()
            .0,
        message
    );
}
