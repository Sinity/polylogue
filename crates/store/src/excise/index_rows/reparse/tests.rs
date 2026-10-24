use super::*;
use crate::archive::{cache, open_write};

#[test]
fn text_reparse_freezes_all_input_routes_in_pk_order_before_deletion() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = crate::Archive::open_or_create(dir.path()).unwrap();
    let store = hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600).unwrap();
    let mut a = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    let mut i = open_write(&archive.index_db(), cache::SETUP).unwrap();
    let tx = i.transaction().unwrap();
    tx.execute("INSERT INTO session(pk,id,origin,native_id,kind,changed) VALUES (1,'claude-code:removed','claude-code','removed','main',1)", []).unwrap();
    for pk in 1..=4104 {
        tx.execute(
            "INSERT INTO input_ref(pk,id) VALUES (?1,?2)",
            params![pk, format!("source:{:04}", 5000 - pk)],
        )
        .unwrap();
        if pk == 6 || pk >= 8 {
            tx.execute("INSERT INTO session_record VALUES (1,?1,1,1,'{}')", [pk])
                .unwrap();
        }
    }
    tx.execute_batch(
        "INSERT INTO parsed VALUES(1,1,1,1);
         INSERT INTO process(pk,boot_id,pid,start_ticks,role,input_pk,parser_version)
             VALUES(1,'synthetic-boot',42,1,'harness',2,1);
         INSERT INTO process_session(process_pk,session_pk) VALUES(1,1);
         INSERT INTO run(pk,id,kind,state,evidence,input_pk,parser_version)
             VALUES(1,'synthetic-run','task','completed','synthetic',3,1);
         INSERT INTO run_session VALUES(1,1);
         INSERT INTO link(parent_id,child_id,kind,how,evidence,input_pk,parser_version)
             VALUES('claude-code:removed','claude-code:pending','reference','mention','synthetic',4,1);
         INSERT INTO link(parent_id,child_id,child_pk,kind,how,evidence,input_pk,parser_version)
             VALUES('claude-code:other','claude-code:removed',1,'reference','mention','synthetic',5,1);",
    ).unwrap();
    tx.commit().unwrap();
    let holding = BTreeSet::from(["source:4999".to_owned()]);
    let expected: Vec<_> = (1..=4104)
        .filter(|&pk| pk != 7)
        .map(|pk| format!("source:{:04}", 5000 - pk))
        .collect();
    for dry_run in [true, false] {
        let at = a.transaction().unwrap();
        let it = i.transaction().unwrap();
        let t = Transactions {
            archive: &at,
            index: &it,
            embeddings: None,
            store: &store,
            seq: 1,
            counter: 1,
        };
        let hashes = RemovalHashes::new(&it).unwrap();
        let inputs = clear_for_reparse(&t, "not present", &holding, dry_run, &hashes).unwrap();
        assert_eq!(inputs, expected);
        let sessions: i64 = it
            .query_row("SELECT count(*) FROM session", [], |r| r.get(0))
            .unwrap();
        assert_eq!(sessions, i64::from(dry_run));
        let scratch: i64 = it
            .query_row(
                "SELECT count(*) FROM sqlite_temp_master WHERE name LIKE 'excision_reparse_%'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(scratch, 0);
        if !dry_run {
            for table in [
                "parsed",
                "process",
                "process_session",
                "run",
                "run_session",
                "link",
                "session_record",
            ] {
                let n: i64 = it
                    .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
                    .unwrap();
                assert_eq!(n, 0, "{table}");
            }
            assert_eq!(
                it.query_row("SELECT id FROM input_ref WHERE pk=7", [], |r| r
                    .get::<_, String>(0))
                    .unwrap(),
                "source:4993"
            );
        }
        hashes.finish().unwrap();
        it.commit().unwrap();
        at.rollback().unwrap();
    }
}

#[test]
fn text_reparse_preserves_text_owned_only_by_a_retained_version() {
    use crate::encoding::PreparedMessage;
    use crate::fixture::{Fixture, MessageFacts, SessionFacts, writer::message};
    use polylogue_model::{Block, Role};

    for retained in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let mut fixture = Fixture::new(dir.path()).unwrap();
        let session = fixture
            .session(SessionFacts {
                id: "claude-code:retained",
                ..Default::default()
            })
            .unwrap();
        let message_pk = fixture
            .message(
                session,
                MessageFacts {
                    id: "claude-code:retained-m",
                    role: "assistant",
                    written_by: "model",
                    text: Some("Unrelated current text"),
                    ..Default::default()
                },
            )
            .unwrap();
        let archive = &fixture.archive;
        let store = hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600).unwrap();
        let mut a = open_write(&archive.archive_db(), cache::SETUP).unwrap();
        let mut i = open_write(&archive.index_db(), cache::SETUP).unwrap();
        let historical = message(
            "m",
            "retained",
            Role::Assistant,
            1,
            vec![Block::Text {
                text: "historical synthetic marker".into(),
            }],
        );
        let prepared = PreparedMessage::new(&historical).unwrap();
        let at = a.transaction().unwrap();
        let it = i.transaction().unwrap();
        let version = crate::versions::retain(
            &it,
            &prepared,
            &historical,
            &std::collections::HashMap::new(),
            false,
        )
        .unwrap();
        let tid: i64 = it
            .query_row(
                "SELECT text_id FROM version_block WHERE version_pk=?1",
                [version],
                |r| r.get(0),
            )
            .unwrap();
        // A source association owns retained history; a bare version may be pruned.
        if retained {
            it.execute(
                "INSERT INTO message_source(message_pk,input_pk,chain,session_pk,
                parser_version,source_start,source_end,preferred,current,version_hash)
                SELECT pk,input_pk,1,session_pk,1,0,1,0,0,?2 FROM message WHERE pk=?1",
                params![message_pk, &prepared.hash[..]],
            )
            .unwrap();
        }
        let t = Transactions {
            archive: &at,
            index: &it,
            embeddings: None,
            store: &store,
            seq: 1,
            counter: 1,
        };
        let hashes = RemovalHashes::new(&it).unwrap();
        let inputs =
            clear_for_reparse(&t, "synthetic marker", &BTreeSet::new(), false, &hashes).unwrap();
        assert!(inputs.is_empty());
        assert_eq!(hashes.text_count().unwrap(), 1);
        let texts: i64 = it
            .query_row("SELECT count(*) FROM text WHERE id=?1", [tid], |r| r.get(0))
            .unwrap();
        let blocks: i64 = it
            .query_row(
                "SELECT count(*) FROM version_block WHERE text_id=?1",
                [tid],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!((texts, blocks), (i64::from(retained), i64::from(retained)));
        assert_eq!(
            it.query_row(
                "SELECT count(*) FROM block WHERE message_pk=?1",
                [message_pk],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
            1
        );
        if retained {
            assert_eq!(
                crate::versions::load(&it, &prepared.hash).unwrap(),
                (version, historical)
            );
        }
    }
}
