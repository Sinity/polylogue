use super::*;

#[test]
fn historical_archive4_upgrades_without_changing_authored_data_or_identity() {
    let dir = tempfile::tempdir().unwrap();
    let archive = create(dir.path()).unwrap();
    let db = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    assert_eq!(crate::schema::version(&db).unwrap(), 4);
    for (table, column) in [
        ("observation", "commit_seq"),
        ("delivery_event", "commit_seq"),
        ("machine", "first_commit_seq"),
    ] {
        let count: i64 = db
            .query_row(
                "SELECT count(*) FROM pragma_table_info(?1) WHERE name=?2",
                [table, column],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 0, "{table}.{column}");
    }
    for table in [
        "vector_deletion_policy",
        "excision_document",
        "extracted_blob",
    ] {
        let count: i64 = db
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE name=?1",
                [table],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 0, "{table}");
    }
    let closure = delivery(&db).unwrap();
    assert_eq!(closure.iter().filter(|&&b| b == b'\n').count(), 2);
    db.execute_batch(
        "INSERT INTO note(id,root,kind,value,author,machine,created_at,commit_seq)
        VALUES ('note','note','note','authored note','operator','test',3,3);
        INSERT INTO setting(id,machine,key,value,author,set_at,commit_seq)
        VALUES ('setting','*','day.zone','UTC','operator',4,4);",
    )
    .unwrap();
    let identity: String = db
        .query_row("SELECT value FROM meta WHERE key='archive_id'", [], |r| {
            r.get(0)
        })
        .unwrap();
    let packed: Vec<u8> = db
        .query_row("SELECT body FROM delivery", [], |r| r.get(0))
        .unwrap();
    let hash: Vec<u8> = db
        .query_row("SELECT content_hash FROM delivery", [], |r| r.get(0))
        .unwrap();
    drop(db);
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let db = open_write(&archive.archive_db(), cache::SETUP).unwrap();
    assert_eq!(crate::schema::version(&db).unwrap(), 9);
    assert_eq!(
        db.query_row("SELECT value FROM meta WHERE key='archive_id'", [], |r| {
            r.get::<_, String>(0)
        })
        .unwrap(),
        identity
    );
    assert_eq!(
        db.query_row("SELECT body FROM delivery", [], |r| r.get::<_, Vec<u8>>(0))
            .unwrap(),
        packed
    );
    assert_eq!(
        db.query_row("SELECT content_hash FROM delivery", [], |r| r
            .get::<_, Vec<u8>>(0))
            .unwrap(),
        hash
    );
    assert_eq!(
        crate::encoding::unpack_bytes(&packed).unwrap(),
        b"Synthetic legacy body"
    );
    for (sql, expected) in [
        (
            "SELECT detail FROM delivery_event",
            "legacy private evidence",
        ),
        ("SELECT value FROM note", "authored note"),
        ("SELECT value FROM setting", "UTC"),
    ] {
        assert_eq!(
            db.query_row(sql, [], |r| r.get::<_, String>(0)).unwrap(),
            expected
        );
    }
    let backfill: i64 = db
        .query_row("SELECT commit_seq FROM delivery_event", [], |r| r.get(0))
        .unwrap();
    assert!(backfill > 4, "backfill follows authored rows");
}
