use crate::{Archive, ReadPool, fixture::Fixture};

#[test]
fn exact_revisions_are_distinguishable_and_missing_selection_stays_empty() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let conn = crate::archive::open_write(&fixture.archive.archive_db(), 1).unwrap();
    conn.execute_batch(
        "INSERT INTO input VALUES ('input:with:colons','claude-code');
        INSERT INTO revision VALUES ('input:with:colons',1,'first',1,10,1);
        INSERT INTO revision VALUES ('input:with:colons',2,'appended',1,20,2);
        INSERT INTO revision VALUES ('input:with:colons',3,'rewritten',3,5,3);",
    )
    .unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let first = reader
        .schema_input("input:with:colons", Some(1))
        .unwrap()
        .unwrap();
    let latest = reader
        .schema_input("input:with:colons", None)
        .unwrap()
        .unwrap();
    assert_eq!((first.revision, first.chain, first.length), (1, 1, 10));
    assert_eq!((latest.revision, latest.chain, latest.length), (3, 3, 5));
    assert!(reader.schema_input("missing", None).unwrap().is_none());
    assert!(
        reader
            .schema_input("input:with:colons", Some(4))
            .unwrap()
            .is_none()
    );
}

#[test]
fn primary_selection_uses_the_typed_store_without_a_standalone_archive() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("primary");
    let fixture = Fixture::new(&root).unwrap();
    let conn = crate::archive::open_write(&fixture.archive.archive_db(), 1).unwrap();
    conn.execute_batch(
        "INSERT INTO input VALUES ('primary-input','claude-code');
        INSERT INTO revision VALUES ('primary-input',1,'first',1,0,1);",
    )
    .unwrap();
    drop(conn);
    drop(fixture);
    std::fs::rename(root.join("archive.db"), root.join("sinex.db")).unwrap();
    let archive = Archive::on_sinex(&root, &dir.path().join("sinex"));
    let pool = ReadPool::open(&archive).unwrap();
    let reader = pool.reader().unwrap();
    assert_eq!(
        reader
            .schema_input("primary-input", None)
            .unwrap()
            .unwrap()
            .revision,
        1
    );
    assert!(!root.join("archive.db").exists());
}
