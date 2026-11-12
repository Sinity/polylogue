use super::*;

#[test]
fn existing_indexes_gain_writer_reverse_lookups_without_reparse() {
    assert!(!crate::writer::deferred_index_names().any(|name| name == "parser_context_version"));
    for primary in [false, true] {
        for building in [false, true] {
            let conn = Connection::open_in_memory().unwrap();
            conn.pragma_update(None, "foreign_keys", true).unwrap();
            let ensure = if primary {
                ensure_primary_index
            } else {
                ensure_index
            };
            ensure(&conn).unwrap();
            conn.execute_batch(
                "INSERT INTO input_ref(pk,id,format) VALUES(1,'context-input','codex');
                 INSERT INTO message_version VALUES(x'01',x'626f6479');
                 INSERT INTO parser_context_state VALUES(1,17,'synthetic',4);
                 INSERT INTO parser_context VALUES(1,17,'synthetic',4,'retained',x'636f6e74657874',x'01');
                 INSERT INTO parser_context VALUES(1,17,'synthetic',4,'independent',x'6f74686572',NULL);
                 INSERT INTO parse_state(input_pk,chain,status,parsed_upto,reading_version,checkpoint)
                     VALUES(1,17,'partial',4,'synthetic',x'636865636b706f696e74');
                 INSERT INTO session(pk,id,origin,native_id,kind,changed)
                     VALUES(1,'claude-code:context','claude-code','context','main',1);
                 INSERT INTO message(pk,id,session_pk,session_order,role,written_by,input_pk)
                     VALUES(1,'claude-code:anchor',1,'0','user','human',1);
                 INSERT INTO event(pk,id,session_pk,seq,after_message_pk,kind,input_pk,fields,data)
                     VALUES(1,'synthetic-event',1,0,1,'synthetic',1,x'6669656c6473',x'64617461');
                 INSERT INTO link(pk,parent_id,child_id,child_pk,kind,evidence,input_pk)
                     VALUES(1,'claude-code:parent','claude-code:context',1,'fork','synthetic',1);
                 DROP INDEX parser_context_version;
                 DROP INDEX event_after;
                 CREATE INDEX event_after   ON event(after_message_pk) WHERE after_message_pk IS NOT NULL;
                 DROP INDEX link_child;
                 CREATE INDEX link_child  ON link(child_pk);"
            ).unwrap();
            if building {
                conn.execute_batch("DROP INDEX event_after; DROP INDEX link_child")
                    .unwrap();
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key,value) VALUES('building','1')",
                    [],
                )
                .unwrap();
                for name in crate::writer::deferred_index_names() {
                    conn.execute_batch(&format!("DROP INDEX IF EXISTS {name}"))
                        .unwrap();
                }
            } else {
                conn.execute("DELETE FROM meta WHERE key='building'", [])
                    .unwrap();
            }
            let retained = snapshot(&conn);
            let identity = index_identity(&conn).unwrap();
            let roots = || {
                conn.prepare(
                    "SELECT name,rootpage FROM sqlite_schema WHERE type='table' ORDER BY name",
                )
                .unwrap()
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap()
            };
            let table_roots = roots();
            for _ in 0..2 {
                ensure(&conn).unwrap();
                assert_eq!(snapshot(&conn), retained);
                assert_eq!(index_identity(&conn).unwrap(), identity);
                assert_eq!(roots(), table_roots, "table storage must stay in place");
                assert_eq!(conn.query_row("SELECT hex(fields)||':'||hex(data)||':'||after_message_pk FROM event WHERE pk=1",[],|r|r.get::<_,String>(0)).unwrap(),"6669656C6473:64617461:1");
                assert_eq!(
                    conn.query_row(
                        "SELECT parent_id||':'||child_id||':'||child_pk FROM link WHERE pk=1",
                        [],
                        |r| r.get::<_, String>(0)
                    )
                    .unwrap(),
                    "claude-code:parent:claude-code:context:1"
                );
                assert_eq!(conn.query_row("SELECT count(*) FROM sqlite_schema WHERE name IN('event_after','link_child')",[],|r|r.get::<_,i64>(0)).unwrap(),2);
                assert_eq!(version(&conn).unwrap(), INDEX_VERSION);
                assert_eq!(crate::schema::building(&conn).unwrap(), building);
                assert_eq!(conn.query_row("SELECT count(*) FROM sqlite_master WHERE type='index' AND name='parser_context_version'",[],|r|r.get::<_,i64>(0)).unwrap(),1);
                assert_eq!(
                    conn.query_row("SELECT count(*) FROM parser_context", [], |r| r
                        .get::<_, i64>(0))
                        .unwrap(),
                    2
                );
                assert_eq!(
                    conn.query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |r| r
                        .get::<_, i64>(
                        0
                    ))
                    .unwrap(),
                    0
                );
            }
        }
    }
}

#[test]
fn repeated_source_upserts_keep_context_bodies_until_the_context_goes() {
    for primary in [false, true] {
        let conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "foreign_keys", true).unwrap();
        let ensure = if primary {
            ensure_primary_index
        } else {
            ensure_index
        };
        ensure(&conn).unwrap();
        crate::versions::track_unreferenced(&conn).unwrap();
        conn.execute_batch(
                "INSERT INTO input_ref(pk,id,format) VALUES(1,'context-input','claude-code');
                 INSERT INTO session(pk,id,origin,native_id,kind,changed) VALUES(1,'claude-code:context','claude-code','context','main',1);
                 INSERT INTO message(pk,id,session_pk,session_order,role,written_by,input_pk)
                    VALUES(1,'claude-code:response',1,'0','assistant','model',1);
                 INSERT INTO message_version VALUES(x'01',x'616263'),(x'02',x'646566');
                 INSERT INTO message_source(message_pk,input_pk,chain,source_start,source_end,id_kind,session_pk,version_hash,route,completion)
                    VALUES(1,1,17,0,4,'native',1,x'01','live','finished');
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
        // Moving off a version the context keeps leaves its body.
        for hash in [vec![2u8], vec![1], vec![2]] {
            conn.execute(
                    "INSERT INTO message_source(message_pk,input_pk,chain,source_start,source_end,id_kind,session_pk,version_hash,route,completion)
                     VALUES(1,1,17,0,4,'native',1,?1,'live','finished')
                     ON CONFLICT(message_pk,input_pk,chain) DO UPDATE SET version_hash=excluded.version_hash",
                    [hash],
                ).unwrap();
        }
        crate::versions::remove_unreferenced(&conn, false).unwrap();
        assert_eq!(snapshot(&conn), retained);
        assert_eq!(versions(&conn), ["01", "02"]);
        // Releasing the context leaves the earlier body without references.
        conn.execute_batch("DELETE FROM parser_context; DELETE FROM parser_context_state;")
            .unwrap();
        crate::versions::remove_unreferenced(&conn, false).unwrap();
        assert_eq!(versions(&conn), ["02"]);
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

fn versions(conn: &Connection) -> Vec<String> {
    conn.prepare("SELECT hex(version_hash) FROM message_version ORDER BY version_hash")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
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
