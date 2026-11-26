use crate::fixture::Fixture;

#[test]
fn exact_revisions_are_distinguishable_and_missing_selection_stays_empty() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let conn = crate::archive::open_write(&fixture.archive.archive_db(), 1).unwrap();
    conn.execute_batch(
        "INSERT INTO input VALUES ('input:with:colons','claude-code');
        INSERT INTO revision(input,number,produced,chain,length,commit_seq) VALUES ('input:with:colons',1,'first',1,10,1);
        INSERT INTO revision(input,number,produced,chain,length,commit_seq) VALUES ('input:with:colons',2,'appended',1,20,2);
        INSERT INTO revision(input,number,produced,chain,length,commit_seq) VALUES ('input:with:colons',3,'rewritten',3,5,3);",
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
fn schema_selection_pages_all_origins_and_preserves_missing_revisions_and_unknown_times() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let conn = crate::archive::open_write(&fixture.archive.archive_db(), 1).unwrap();
    conn.execute_batch("INSERT INTO input VALUES('a','claude-code'),('b','codex'),('c','claude-code');
      INSERT INTO revision(input,number,produced,chain,length,commit_seq) VALUES('a',1,'first',1,10,1),('b',1,'first',1,20,2),('b',2,'appended',1,30,3),('c',1,'first',1,5,4);
      INSERT INTO observation(id,input,machine,path_bytes,class,revision,length,polylogue_version,seen_at,commit_seq) VALUES
      ('a-o','a','m', CAST('a.json' AS BLOB),'first',1,10,'test',100,1),('b-o','b','m', CAST('b.jsonl' AS BLOB),'appended',2,30,'test',200,3);").unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let first = reader
        .schema_inputs_page(None, None, None, None, 1)
        .unwrap();
    let fields = |page: &crate::schema_inputs::SchemaInputsPage| {
        page.inputs
            .iter()
            .map(|i| {
                (
                    i.input.clone(),
                    i.format.clone(),
                    i.revision,
                    i.chain,
                    i.length,
                    i.path.clone(),
                )
            })
            .collect::<Vec<_>>()
    };
    let row = |input: &str, format: &str, revision: i64, chain: i64, length: u64, path: &str| {
        (
            input.to_owned(),
            format.to_owned(),
            revision,
            chain,
            length,
            path.to_owned(),
        )
    };
    assert_eq!(
        fields(&first),
        [row("a", "claude-code", 1, 1, 10, "a.json")]
    );
    assert_eq!(first.next.as_deref(), Some("a"));
    assert_eq!(first.unknown_time_inputs, Some(1));
    let second = reader
        .schema_inputs_page(None, None, None, first.next.as_deref(), 1)
        .unwrap();
    assert_eq!(fields(&second), [row("b", "codex", 2, 1, 30, "b.jsonl")]);
    assert_eq!(second.unknown_time_inputs, None);
    let third = reader
        .schema_inputs_page(None, None, None, second.next.as_deref(), 1)
        .unwrap();
    assert_eq!(fields(&third), [row("c", "claude-code", 1, 1, 5, "")]);
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
    assert_eq!(since.unknown_time_inputs, Some(1));
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

#[test]
fn exact_revision_time_selection_reports_missing_separately_from_unobserved_time() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let conn = crate::archive::open_write(&fixture.archive.archive_db(), 1).unwrap();
    conn.execute_batch("INSERT INTO input VALUES('missing','fixture'),('recent','fixture'),('unknown','fixture');
      INSERT INTO revision(input,number,produced,chain,length,commit_seq) VALUES('missing',1,'first',1,1,1),('recent',2,'first',2,1,2),('unknown',2,'first',2,1,3);
      INSERT INTO observation(id,input,machine,path_bytes,class,revision,length,polylogue_version,seen_at,commit_seq)
      VALUES('o','recent','m', CAST('recent.json' AS BLOB),'first',2,1,'test',200,2);").unwrap();
    let pool = fixture.reads().unwrap();
    let page = pool
        .reader()
        .unwrap()
        .schema_inputs_page(Some("fixture"), Some(2), Some(150), None, 10)
        .unwrap();
    assert_eq!(
        page.inputs
            .iter()
            .map(|input| input.input.as_str())
            .collect::<Vec<_>>(),
        ["recent"]
    );
    assert_eq!(page.missing_revisions, ["missing"]);
    assert_eq!(page.unknown_time_inputs, Some(1));
    assert!(page.next.is_none());
}
