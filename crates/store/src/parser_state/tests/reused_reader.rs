use super::*;

#[test]
fn reuse_releases_the_old_view_and_checks_a_fresh_predecessor() {
    let (_dir, root, mut writer, input) = fixture();
    let first = writer.begin_parser(&basis(input, 10)).unwrap();
    let snapshot = writer
        .parser_reader()
        .unwrap()
        .snapshot(&first.token)
        .unwrap();
    assert_eq!(snapshot.cell("test", b"key", 100).unwrap(), None);
    let next = write_step(
        &mut writer._index,
        &root.join("index.db"),
        &first,
        progress(10, 10, Phase::Finalizing, true),
        &[change("key", "new")],
        budget(),
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(snapshot.cell("test", b"key", 100).unwrap(), None);
    let reader = snapshot.into_reader().unwrap();
    assert!(
        reader.db.is_autocommit(),
        "idle means no pinned read transaction"
    );
    let checkpoint: (i64, i64, i64) = writer
        ._index
        .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .unwrap();
    assert_eq!(
        checkpoint.0, 0,
        "released view must not obstruct checkpoint"
    );
    let snapshot = reader.snapshot(&next.token).unwrap();
    assert_eq!(
        snapshot.cell("test", b"key", 100).unwrap(),
        Some(b"new".to_vec())
    );
    let reader = snapshot.into_reader().unwrap();
    assert!(matches!(
        reader.snapshot(&first.token),
        Err(StoreError::StaleParserState)
    ));
}

#[test]
fn idle_parser_reader_keeps_archive_authority_until_released() {
    let (_dir, root, mut writer, input) = fixture();
    let first = writer.begin_parser(&basis(input, 10)).unwrap();
    let reader = writer
        .parser_reader()
        .unwrap()
        .snapshot(&first.token)
        .unwrap()
        .into_reader()
        .unwrap();
    drop(writer);
    assert!(matches!(
        Writer::open(&root),
        Err(StoreError::WriterHeld { .. })
    ));
    drop(reader);
    assert!(Writer::open(&root).is_ok());
}
