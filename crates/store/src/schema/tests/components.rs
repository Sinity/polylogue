use super::super::*;
use rusqlite::params;

fn source_rows(conn: &Connection, prefix: crate::source::Source) {
    conn.execute(
        &prefix.sql("INSERT INTO @source.input(id,format) VALUES('retained','synthetic')"),
        [],
    )
    .unwrap();
    conn.execute(&prefix.sql("INSERT INTO @source.client_request(id,content_hash,created_at,request,outcome,effect,commit_seq) VALUES('request',x'0102',1,x'0304',x'0506',x'0708',3)"), []).unwrap();
    conn.execute(&prefix.sql("INSERT INTO @source.note(id,first_note,kind,value,author,machine,created_at,commit_seq) VALUES('request','request','note','retained text','operator','machine',1,3)"), []).unwrap();
    conn.execute(&prefix.sql("INSERT INTO @source.delivery(id,recipient_kind,sender,event,at,content_hash,bytes,body,refs_json,commit_seq) VALUES('delivery','operator','signal','signal',1,x'01',2,x'0203','[]',3)"), []).unwrap();
    for (id, millis) in [("negative", -1_i64), ("epoch", 0), ("positive", 1234)] {
        conn.execute(&prefix.sql("INSERT INTO @source.variant(id,root,source,source_hash,kind,author,via,machine,created_at,commit_seq) VALUES(?1,?1,'message:source',zeroblob(32),'translation','operator','cli','machine',?2,3)"), params![id,millis]).unwrap();
        conn.execute(&prefix.sql("INSERT INTO @source.variant_part(variant,n,of,of_hash,relation,body) VALUES(?1,0,'message:source',zeroblob(32),'translation',x'0011ff')"), [id]).unwrap();
    }
}

fn retained(
    conn: &Connection,
    prefix: crate::source::Source,
) -> (String, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Option<Vec<u8>>) {
    conn.query_row(&prefix.sql("SELECT n.value,r.request,r.outcome,r.effect,d.body,d.signal FROM @source.note n JOIN @source.client_request r ON r.id=n.id JOIN @source.delivery d ON d.id='delivery'"), [], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?,row.get(5)?))).unwrap()
}

fn retained_variants(conn: &Connection, source: crate::source::Source) {
    let actual: Vec<(String,String,Vec<u8>)> = conn.prepare(&source.sql("SELECT v.id,v.created_at,p.body FROM @source.variant v JOIN @source.variant_part p ON p.variant=v.id ORDER BY v.id")).unwrap().query_map([], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?))).unwrap().collect::<rusqlite::Result<_>>().unwrap();
    assert_eq!(
        actual,
        vec![
            (
                "epoch".into(),
                "1970-01-01T00:00:00Z".into(),
                vec![0, 17, 255]
            ),
            (
                "negative".into(),
                "1969-12-31T23:59:59.999Z".into(),
                vec![0, 17, 255]
            ),
            (
                "positive".into(),
                "1970-01-01T00:00:01.234Z".into(),
                vec![0, 17, 255]
            ),
        ]
    );
    let violations: i64 = conn
        .query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(violations, 0);
    let imported: i64 = conn
        .query_row(
            &source.sql("SELECT count(*) FROM @source.variant_import"),
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(imported, 0);
}

#[test]
fn source_two_upgrades_query_and_signal_storage_without_rewriting_authored_bytes() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(include_str!("../fixtures/archive_2.sql"))
        .unwrap();
    conn.execute_batch("PRAGMA user_version=2;").unwrap();
    source_rows(&conn, crate::source::Source::Ledger);
    ensure_archive(&conn).unwrap();
    ensure_archive(&conn).unwrap();
    assert_eq!(version(&conn).unwrap(), 4);
    retained_variants(&conn, crate::source::Source::Ledger);
    let values = retained(&conn, crate::source::Source::Ledger);
    assert_eq!(
        values,
        (
            "retained text".into(),
            vec![3, 4],
            vec![5, 6],
            vec![7, 8],
            vec![2, 3],
            None
        )
    );
    for table in ["query", "query_run", "signal_state", "signal_event"] {
        let n: i64 = conn
            .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(n, 0);
    }
}

#[test]
fn index_six_upgrades_real_components_in_both_modes_and_preserves_source_and_parsed_rows() {
    for primary in [false, true] {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(include_str!("../fixtures/index_5.sql"))
            .unwrap();
        crate::archive::generation::create(&conn).unwrap();
        conn.execute_batch("PRAGMA user_version=6; INSERT INTO input_ref(pk,id,format) VALUES(1,'retained','synthetic');").unwrap();
        let text = crate::encoding::pack_text("unchanged indexed bytes").unwrap();
        conn.execute(
            "INSERT INTO text(id,hash,body) VALUES(1,?1,?2)",
            params![
                &blake3::hash(b"unchanged indexed bytes").as_bytes()[..],
                text
            ],
        )
        .unwrap();
        if primary {
            conn.execute_batch(
                &crate::source::from_archive(include_str!("../fixtures/archive_2.sql"))
                    .unwrap()
                    .ddl,
            )
            .unwrap();
            conn.execute_batch(crate::primary::projection::SCHEMA)
                .unwrap();
            conn.execute(
                "INSERT INTO meta(key,value) VALUES('primary_cache','synthetic-cache')",
                [],
            )
            .unwrap();
            source_rows(&conn, crate::source::Source::Cache);
            ensure_primary_index(&conn).unwrap();
            ensure_primary_index(&conn).unwrap();
            assert_eq!(
                retained(&conn, crate::source::Source::Cache).0,
                "retained text"
            );
            retained_variants(&conn, crate::source::Source::Cache);
        } else {
            ensure_index(&conn).unwrap();
            ensure_index(&conn).unwrap();
        }
        assert_eq!(version(&conn).unwrap(), 12);
        let body: Vec<u8> = conn
            .query_row("SELECT body FROM text WHERE id=1", [], |row| row.get(0))
            .unwrap();
        assert_eq!(body, text);
        for table in [
            "message_injection_field",
            "attachment_binding",
            "text_fragment",
            "text_fragment_source",
        ] {
            let n: i64 = conn
                .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                    row.get(0)
                })
                .unwrap();
            assert_eq!(n, 0);
        }
    }
}

#[test]
fn unpublished_assertion_version_is_refused_without_running_later_migrations() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(include_str!("../fixtures/index_5.sql"))
        .unwrap();
    conn.execute_batch("PRAGMA user_version=7; INSERT INTO input_ref(pk,id,format) VALUES(1,'retained','synthetic');").unwrap();
    assert!(matches!(
        ensure_index(&conn),
        Err(StoreError::NoMigration {
            database: "index",
            found: 7
        })
    ));
    assert_eq!(version(&conn).unwrap(), 7);
    assert_eq!(
        conn.query_row("SELECT id FROM input_ref", [], |row| row
            .get::<_, String>(0))
            .unwrap(),
        "retained"
    );
}

#[test]
fn invalid_legacy_variant_time_rolls_back_the_entire_source_upgrade() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(include_str!("../fixtures/archive_2.sql"))
        .unwrap();
    conn.execute_batch("PRAGMA user_version=2;").unwrap();
    source_rows(&conn, crate::source::Source::Ledger);
    conn.execute(
        "UPDATE variant SET created_at=?1 WHERE id='negative'",
        [i64::MAX],
    )
    .unwrap();
    assert!(ensure_archive(&conn).is_err());
    assert_eq!(version(&conn).unwrap(), 2);
    let actual: i64 = conn
        .query_row(
            "SELECT created_at FROM variant WHERE id='negative'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(actual, i64::MAX);
    let parts: i64 = conn
        .query_row("SELECT count(*) FROM variant_part", [], |row| row.get(0))
        .unwrap();
    assert_eq!(parts, 3);
}
