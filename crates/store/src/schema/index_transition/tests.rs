use super::*;
use crate::encoding;
use polylogue_model::Block;
use polylogue_model::{Message, bundle::MessageProjection};
use rusqlite::params;
use serde_json::json;

fn old_index(
    primary: bool,
    lineage: bool,
    current_authored: bool,
    building: bool,
    sparse: bool,
) -> (Connection, Message, MessageProjection) {
    let conn = predecessor(primary, lineage, current_authored).unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    conn.pragma_update(None, "application_id", FORMAT_ID)
        .unwrap();
    set_version(&conn, 1).unwrap();
    conn.execute_batch(
        "INSERT INTO meta VALUES('generation','retained-generation'),('cursor_key','1111111111111111111111111111111111111111111111111111111111111111');
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
    if primary {
        conn.execute_batch("INSERT INTO source_assertion(id,root,kind,author,via,machine,created_at,commit_seq)
             VALUES('retained-note','retained-note','note','human','cli','synthetic',123456,9);
             INSERT INTO source_link(record,role,ref) VALUES('retained-note','target','session:claude-code:session');
             INSERT INTO source_revision_state VALUES('retained-note','retained-note',1,0);
             INSERT INTO source_assertion_slot VALUES(zeroblob(32),'retained-note');").unwrap();
        if current_authored {
            conn.execute_batch("INSERT INTO source_assertion(id,root,kind,author,via,machine,created_at,commit_seq)
                VALUES('unknown-time','unknown-time','note','human','import','synthetic',NULL,10);
                INSERT INTO source_assertion_import VALUES('retained-note','physical-input',3,'synthetic','human','2026-01-01T00:00:00Z',9);").unwrap();
        }
    }
    let prepared = encoding::PreparedMessage::projected(&message, &projection).unwrap();
    let body = &prepared.body;
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
        let body = &prepared.blocks[idx];
        let data = &prepared.data[idx];
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
    conn.execute_batch("INSERT INTO session(pk,id,origin,native_id,kind,changed) VALUES(8,'claude-code:other','claude-code','other','main',3); DELETE FROM derive_due; INSERT INTO derive_due(session_pk,rows_changed,repo_lookup,rewritten,marked) VALUES(7,19,1,1,23);").unwrap();
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

type Rows = Vec<Vec<rusqlite::types::Value>>;

/// Include rowids where they exist, and opaque packed bytes rather than
/// decoding and rewriting them through the current model.
fn retained(conn: &Connection) -> std::collections::BTreeMap<String, Rows> {
    let tables = conn
        .prepare("SELECT name,wr FROM pragma_table_list WHERE schema='main' AND type<>'view' AND name NOT LIKE 'sqlite_%' AND name<>'derive_due' ORDER BY name")
        .unwrap()
        .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, bool>(1)?)))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    tables
        .into_iter()
        .map(|(name, without_rowid)| {
            let quoted = name.replace('"', "\"\"");
            let columns = if without_rowid { "*" } else { "rowid,*" };
            let mut statement = conn
                .prepare(&format!("SELECT {columns} FROM \"{quoted}\""))
                .unwrap();
            let count = statement.column_count();
            let mut rows = statement
                .query_map([], |row| {
                    (0..count)
                        .map(|column| row.get(column))
                        .collect::<rusqlite::Result<Vec<rusqlite::types::Value>>>()
                })
                .unwrap()
                .collect::<rusqlite::Result<Rows>>()
                .unwrap();
            // The fixture has no NaNs. Sort typed rows without depending on a
            // query planner's row order, retaining every opaque byte.
            rows.sort_by_key(|row| format!("{row:?}"));
            (name, rows)
        })
        .collect()
}

fn unchanged(conn: &Connection, before: &std::collections::BTreeMap<String, Rows>) {
    let after = retained(conn);
    for (name, rows) in before {
        assert_eq!(after.get(name), Some(rows), "retained table {name}");
    }
}

fn queue(conn: &Connection) -> Vec<(i64, i64, i64, i64, i64)> {
    conn.prepare("SELECT session_pk,rows_changed,repo_lookup,rewritten,marked FROM derive_due ORDER BY session_pk").unwrap()
        .query_map([], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?))).unwrap()
        .collect::<rusqlite::Result<_>>().unwrap()
}

#[test]
fn known_index1_shapes_gain_only_the_declared_additions_and_keep_every_old_row() {
    for primary in [false, true] {
        for lineage in [false, true] {
            for authored in [false, true]
                .into_iter()
                .filter(|current| primary || *current)
            {
                for building in [false, true] {
                    for sparse in [false, true] {
                        let (conn, message, projection) =
                            old_index(primary, lineage, authored, building, sparse);
                        let before = retained(&conn);
                        let original_queue = queue(&conn);
                        let ensure = if primary {
                            ensure_primary_index
                        } else {
                            ensure_index
                        };
                        ensure(&conn).unwrap();
                        assert_eq!(version(&conn).unwrap(), 1);
                        assert_eq!(super::super::building(&conn).unwrap(), building);
                        unchanged(&conn, &before);
                        assert_eq!(
                            conn.pragma_query_value::<u8, _>(None, "foreign_keys", |row| row
                                .get(0))
                                .unwrap(),
                            1
                        );
                        assert!(
                            conn.prepare("PRAGMA foreign_key_check")
                                .unwrap()
                                .query([])
                                .unwrap()
                                .next()
                                .unwrap()
                                .is_none()
                        );
                        if lineage {
                            assert_eq!(queue(&conn), original_queue);
                        } else {
                            assert_eq!(queue(&conn), [(7, 19, 1, 1, 24), (8, 0, 0, 0, 0)]);
                            assert_eq!(
                                conn.query_row("SELECT sum(lineage) FROM derive_due", [], |row| {
                                    row.get::<_, i64>(0)
                                })
                                .unwrap(),
                                2
                            );
                        }
                        let (read, layout) =
                            crate::versions::read_message(&conn, &projection.source_version.0)
                                .unwrap()
                                .unwrap();
                        assert_eq!(read, message);
                        assert_eq!(layout, projection);
                        let after = retained(&conn);
                        let after_queue = queue(&conn);
                        ensure(&conn).unwrap();
                        assert_eq!(retained(&conn), after);
                        assert_eq!(queue(&conn), after_queue);
                    }
                }
            }
        }
    }
}

#[test]
fn an_unknown_index1_shape_is_refused_without_partial_additions() {
    for primary in [false, true] {
        for lineage in [false, true] {
            let (conn, _, _) = old_index(primary, lineage, false, true, false);
            conn.execute_batch(
                "CREATE TABLE unexpected(value); INSERT INTO unexpected VALUES('kept')",
            )
            .unwrap();
            let before = retained(&conn);
            let before_shape = shape(&conn).unwrap();
            let ensure = if primary {
                ensure_primary_index
            } else {
                ensure_index
            };
            assert!(matches!(
                ensure(&conn),
                Err(StoreError::NoMigration {
                    database: "index",
                    found: 1
                })
            ));
            assert_eq!(shape(&conn).unwrap(), before_shape);
            assert_eq!(retained(&conn), before);
        }
    }
}

#[test]
#[cfg(feature = "faults")]
fn index_additions_roll_back_all_metadata_and_retry_in_place() {
    for primary in [false, true] {
        let (conn, _, _) = old_index(primary, false, false, true, true);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.db");
        conn.execute("VACUUM INTO ?1", [path.to_str().unwrap()])
            .unwrap();
        drop(conn);
        let conn = Connection::open(&path).unwrap();
        conn.pragma_update(None, "foreign_keys", true).unwrap();
        let before = retained(&conn);
        let before_queue = queue(&conn);
        let before_shape = shape(&conn).unwrap();
        crate::fault::arm(
            "index.additions_transition",
            conn.path().unwrap(),
            crate::fault::Action::Fail,
            0,
            1,
        );
        let ensure = if primary {
            ensure_primary_index
        } else {
            ensure_index
        };
        assert!(ensure(&conn).is_err());
        assert_eq!(retained(&conn), before);
        assert_eq!(queue(&conn), before_queue);
        assert_eq!(shape(&conn).unwrap(), before_shape);
        ensure(&conn).unwrap();
        unchanged(&conn, &before);
    }
}

#[test]
fn ordinary_open_keeps_originals_vectors_and_progress_across_separate_atomic_transitions() {
    use crate::archive::{Archive, cache, open_read, open_write};
    use std::io::Read;
    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;

    for fail in std::iter::once(false).chain(cfg!(feature = "faults").then_some(true)) {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let (archive, lock) = Archive::open_or_create(root).unwrap();
        let bytes = vec![b'o'; 555];
        let (blob, prepared) = archive.prepare_blob(&bytes).unwrap();
        let writer = crate::Writer::start(&archive).unwrap();
        let mut batch = crate::fixture::writer::step(
            "synthetic-input",
            "claude-code",
            "/home/op/session",
            Default::default(),
            0,
            0,
        );
        batch.parse.clear();
        batch.length = bytes.len() as u64;
        writer.store_bytes(prepared.unwrap(), batch).unwrap();
        drop(writer);

        let source = open_write(&archive.archive_db(), cache::SETUP).unwrap();
        source
            .execute_batch(
                "DROP TABLE assertion_import;
            ALTER TABLE assertion ALTER COLUMN created_at SET NOT NULL;
            PRAGMA user_version=1;
            INSERT INTO assertion(id,root,kind,author,via,machine,created_at,commit_seq)
            VALUES('old-note','old-note','note','human','cli','synthetic',123456,9);",
            )
            .unwrap();
        let source_before = retained(&source);
        drop(source);
        let vectors = open_write(&archive.embeddings_db(), cache::SETUP).unwrap();
        vectors.execute("INSERT INTO embedded(text_hash,model,dims,vector,embedded_at) VALUES(x'0102','paid/synthetic',1,x'0000803f',1)",[]).unwrap();
        drop(vectors);

        let (old, _, _) = old_index(false, false, true, true, true);
        old.execute(
            "INSERT INTO meta(key,value) VALUES('boot_id',?1)",
            [std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
                .unwrap()
                .trim()],
        )
        .unwrap();
        let index_before = retained(&old);
        let old_queue = queue(&old);
        // Test setup installs the exact predecessor in this private archive;
        // the actual open below must retain this file, not replace it.
        for suffix in ["", "-wal", "-shm"] {
            let path = root.join(format!("index.db{suffix}"));
            if path.exists() {
                std::fs::remove_file(path).unwrap();
            }
        }
        old.execute("VACUUM INTO ?1", [archive.index_db().to_str().unwrap()])
            .unwrap();
        drop(old);
        #[cfg(unix)]
        let inode = std::fs::metadata(archive.index_db()).unwrap().ino();
        drop(lock);

        #[cfg(feature = "faults")]
        if fail {
            // SQLite resolves symlinked temporary roots in its filename. Bind
            // the fault to that filename, as the production setup connection does.
            let index = open_read(&archive.index_db()).unwrap();
            let filename = index.path().unwrap().to_owned();
            drop(index);
            crate::fault::arm(
                "index.additions_transition",
                &filename,
                crate::fault::Action::Fail,
                0,
                1,
            );
        }
        let first = Archive::open_or_create(root);
        let (archive, lock) = if fail {
            assert!(first.is_err());
            let source = open_read(&archive.archive_db()).unwrap();
            assert_eq!(version(&source).unwrap(), 2);
            unchanged(&source, &source_before);
            drop(source);
            let index = open_read(&archive.index_db()).unwrap();
            assert_eq!(retained(&index), index_before);
            assert_eq!(queue(&index), old_queue);
            assert!(
                !index
                    .query_row(
                        "SELECT EXISTS(SELECT 1 FROM sqlite_schema WHERE name='session_lineage')",
                        [],
                        |r| r.get::<_, bool>(0)
                    )
                    .unwrap()
            );
            drop(index);
            Archive::open_or_create(root).unwrap()
        } else {
            first.unwrap()
        };
        #[cfg(unix)]
        assert_eq!(std::fs::metadata(archive.index_db()).unwrap().ino(), inode);
        assert!(!std::fs::read_dir(root).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("index-set-aside-")
        }));
        let index = open_read(&archive.index_db()).unwrap();
        unchanged(&index, &index_before);
        assert_eq!(queue(&index), [(7, 19, 1, 1, 24), (8, 0, 0, 0, 0)]);
        assert_eq!(
            index_identity(&index).unwrap(),
            build_index_identity(false).unwrap()
        );
        drop(index);
        let source = open_read(&archive.archive_db()).unwrap();
        unchanged(&source, &source_before);
        assert_eq!(version(&source).unwrap(), 2);
        drop(source);
        let mut restored = Vec::new();
        archive
            .blobs()
            .unwrap()
            .read(&hashed_blobs::Hash(blob.hash.0))
            .unwrap()
            .read_to_end(&mut restored)
            .unwrap();
        assert_eq!(restored, bytes);
        let vectors = open_read(&archive.embeddings_db()).unwrap();
        assert_eq!(
            vectors
                .query_row(
                    "SELECT vector FROM embedded WHERE model='paid/synthetic'",
                    [],
                    |r| r.get::<_, Vec<u8>>(0)
                )
                .unwrap(),
            [0, 0, 128, 63]
        );
        drop(vectors);
        drop(lock);
        let (archive, _lock) = Archive::open_or_create(root).unwrap();
        unchanged(&open_read(&archive.index_db()).unwrap(), &index_before);
    }
}
