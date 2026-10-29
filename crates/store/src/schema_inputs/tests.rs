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
    let (archive, _lock) = Archive::open_primary(&root, &dir.path().join("sinex")).unwrap();
    let conn = crate::archive::open_write(&archive.index_db(), 1).unwrap();
    conn.execute_batch(
        "INSERT INTO source_input VALUES ('primary-input','claude-code');
        INSERT INTO source_revision VALUES ('primary-input',1,'first',1,0,1);",
    )
    .unwrap();
    drop(conn);
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
    let page = reader
        .schema_inputs_page(None, None, None, None, 1)
        .unwrap();
    assert_eq!(page.inputs[0].input, "primary-input");
    assert_eq!(page.unknown_time_inputs, 1);
    assert!(page.next.is_none());
}

#[test]
fn schema_selection_pages_all_origins_and_preserves_missing_revisions_and_unknown_times() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let conn = crate::archive::open_write(&fixture.archive.archive_db(), 1).unwrap();
    conn.execute_batch("INSERT INTO input VALUES('a','claude-code'),('b','codex'),('c','claude-code');
      INSERT INTO revision VALUES('a',1,'first',1,10,1),('b',1,'first',1,20,2),('b',2,'appended',1,30,3),('c',1,'first',1,5,4);
      INSERT INTO observation(id,input,machine,path,class,revision,length,polylogue_version,seen_at,commit_seq) VALUES
      ('a-o','a','m','a.json','first',1,10,'test',100,1),('b-o','b','m','b.jsonl','appended',2,30,'test',200,3);").unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let first = reader
        .schema_inputs_page(None, None, None, None, 1)
        .unwrap();
    assert_eq!(first.inputs[0].input, "a");
    assert_eq!(first.next.as_deref(), Some("a"));
    assert_eq!(first.unknown_time_inputs, 1);
    let second = reader
        .schema_inputs_page(None, None, None, first.next.as_deref(), 1)
        .unwrap();
    assert_eq!(second.inputs[0].input, "b");
    assert_eq!(second.inputs[0].revision, 2);
    let third = reader
        .schema_inputs_page(None, None, None, second.next.as_deref(), 1)
        .unwrap();
    assert_eq!(third.inputs[0].input, "c");
    assert!(third.next.is_none());
    let since = reader
        .schema_inputs_page(None, None, Some(150), None, 2)
        .unwrap();
    assert_eq!(
        since
            .inputs
            .iter()
            .map(|input| input.input.as_str())
            .collect::<Vec<_>>(),
        ["b"]
    );
    assert_eq!(since.unknown_time_inputs, 1);
    let missing = reader
        .schema_inputs_page(Some("claude-code"), Some(2), None, None, 1)
        .unwrap();
    assert!(missing.inputs.is_empty());
    assert_eq!(missing.missing_revisions, ["a"]);
    assert_eq!(missing.next.as_deref(), Some("a"));
    let tail = reader
        .schema_inputs_page(
            Some("claude-code"),
            Some(2),
            None,
            missing.next.as_deref(),
            1,
        )
        .unwrap();
    assert_eq!(tail.missing_revisions, ["c"]);
    assert!(tail.next.is_none());
}
