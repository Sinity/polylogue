use super::*;

#[test]
fn repeated_source_upserts_keep_context_bodies_and_their_cleanup_markers() {
    for primary in [false, true] {
        let conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "foreign_keys", true).unwrap();
        let ensure = if primary {
            ensure_primary_index
        } else {
            ensure_index
        };
        ensure(&conn).unwrap();
        conn.execute_batch(
                "INSERT INTO input_ref(pk,id,format) VALUES(1,'context-input','claude-code');
                 INSERT INTO session(pk,id,origin,native_id,kind,changed) VALUES(1,'claude-code:context','claude-code','context','main',1);
                 INSERT INTO message(pk,id,session_pk,order_input_pk,order_input_id,seq,role,written_by,input_pk)
                    VALUES(1,'claude-code:response',1,1,'context-input',0,'assistant','model',1);
                 INSERT INTO message_version VALUES(x'01',x'616263'),(x'02',x'646566');
                 INSERT INTO message_source(message_pk,input_pk,chain,source_start,source_end,id_kind,session_pk,version_hash,route,completion)
                    VALUES(1,1,17,0,4,'native',1,x'01','live','finished');
                 INSERT INTO changed_version VALUES('message',x'01');
                 INSERT INTO parser_context_state VALUES(1,17,'reading=1',4);
                 INSERT INTO parser_context VALUES(1,17,'reading=1',4,'retained-key',x'636f6e74657874',x'01');
                 INSERT INTO parse_state(input_pk,chain,status,parsed_upto,reading_version,checkpoint)
                    VALUES(1,17,'partial',4,'reading=1',x'636865636b706f696e74');",
            ).unwrap();
        let retained = snapshot(&conn);
        ensure(&conn).unwrap();
        assert_eq!(version(&conn).unwrap(), INDEX_VERSION);
        assert_eq!(snapshot(&conn), retained);
        // This is the same source-key UPSERT used for an assembled response.
        // Revisiting an old version must reuse its pending cleanup marker.
        for hash in [vec![2u8], vec![1], vec![2]] {
            conn.execute(
                    "INSERT INTO message_source(message_pk,input_pk,chain,source_start,source_end,id_kind,session_pk,version_hash,route,completion)
                     VALUES(1,1,17,0,4,'native',1,?1,'live','finished')
                     ON CONFLICT(message_pk,input_pk,chain) DO UPDATE SET version_hash=excluded.version_hash",
                    [hash],
                ).unwrap();
        }
        assert_eq!(snapshot(&conn), retained);
        assert_eq!(
            conn.query_row("SELECT count(*) FROM changed_version", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            2
        );
        crate::versions::remove_readings(&conn).unwrap();
        assert_eq!(snapshot(&conn), retained);
        assert_eq!(
            conn.query_row("SELECT count(*) FROM changed_version", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            1
        );
        conn.execute_batch("DELETE FROM parser_context; DELETE FROM parser_context_state;")
            .unwrap();
        crate::versions::remove_readings(&conn).unwrap();
        assert_eq!(
            conn.query_row("SELECT hex(version_hash) FROM message_version", [], |r| {
                r.get::<_, String>(0)
            })
            .unwrap(),
            "02"
        );
        assert_eq!(
            conn.query_row("SELECT count(*) FROM changed_version", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            conn.query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |r| {
                r.get::<_, i64>(0)
            })
            .unwrap(),
            0
        );
        ensure(&conn).unwrap();
    }
}

#[derive(Debug, PartialEq, Eq)]
struct ContextSnapshot {
    body: Vec<u8>,
    value: Vec<u8>,
    chain: i64,
    cursor: i64,
    reading_version: String,
    checkpoint: Vec<u8>,
    version_hash: Vec<u8>,
}

fn snapshot(conn: &Connection) -> ContextSnapshot {
    conn.query_row(
        "SELECT v.body,c.value,c.chain,c.cursor,p.reading_version,p.checkpoint,c.version_hash
         FROM parser_context c JOIN message_version v ON v.version_hash=c.version_hash
         JOIN parse_state p ON p.input_pk=c.input_pk AND p.chain=c.chain",
        [],
        |r| {
            Ok(ContextSnapshot {
                body: r.get(0)?,
                value: r.get(1)?,
                chain: r.get(2)?,
                cursor: r.get(3)?,
                reading_version: r.get(4)?,
                checkpoint: r.get(5)?,
                version_hash: r.get(6)?,
            })
        },
    )
    .unwrap()
}
