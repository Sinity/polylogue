use super::*;
#[test]
fn same_range_with_different_hash_is_checked_again() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("synthetic.jsonl");
    let raw = br#"{"value":"stored","timestamp":"volatile"}"#;
    std::fs::write(&path, raw).unwrap();
    let mut writer = Writer::open(&dir.path().join("archive")).unwrap();
    let source = writer
        .acquire_claude_main(
            &path,
            "synthetic-machine",
            "synthetic-host",
            "synthetic-input",
            1,
            &std::sync::atomic::AtomicBool::new(false),
        )
        .unwrap();
    let canonical =
        polylogue_model::identity::canonical_record(raw, &["timestamp", "ordinal"]).unwrap();
    let good = Citation {
        input: source.input.id,
        chain: source.input.chain,
        start: 0,
        end: raw.len() as u64,
        origin: ToolOrigin::Rollout {
            tag: "call".into(),
            start: 0,
            end: raw.len() as u64,
            raw_hash: *blake3::hash(&canonical).as_bytes(),
        },
        version_hash: "synthetic-version".into(),
        constructor: CODEX_CONSTRUCTOR.into(),
        adapter_revision: CODEX_REVISION.into(),
    };
    let mut different = good.clone();
    different.origin = ToolOrigin::Rollout {
        tag: "status".into(),
        start: 0,
        end: raw.len() as u64,
        raw_hash: [0; 32],
    };
    let reader = writer.read_handle().unwrap();
    assert!(
        verify_sources(
            &reader,
            &[good.clone()],
            &mut Verification::default(),
            &mut || Ok(())
        )
        .is_ok()
    );
    // The caller's deadline check runs inside the source read.
    let mut checks = 0;
    assert!(matches!(
        verify_sources(&reader, &[good.clone()], &mut Verification::default(), &mut || {
            checks += 1;
            if checks >= 3 { Err(bad("synthetic operation deadline")) } else { Ok(()) }
        }),
        Err(Error::Invariant(message)) if message == "synthetic operation deadline"
    ));
    assert_eq!(checks, 3);
    let mut verified = Verification::default();
    assert!(matches!(
        verify_sources(&reader, &[good, different], &mut verified, &mut || Ok(())),
        Err(Error::Invariant(message)) if message == "operation rollout record hash differs"
    ));
}
#[test]
fn duplicate_values_add_references_and_text_rows_are_shared() {
    let dir = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(&dir.path().join("archive")).unwrap();
    let p = writer.root.join("index.db");
    let tx = writer._index.transaction().unwrap();
    let first = store_value(&tx, &p, "json", br#"{"synthetic":true}"#).unwrap();
    assert_eq!(
        store_value(&tx, &p, "json", br#"{"synthetic":true}"#).unwrap(),
        first
    );
    assert_eq!(
        tx.query_row(
            "SELECT incoming FROM body_value WHERE pk=?1",
            [first],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        2
    );
    // A field already stores the same UTF-8 bytes. The new text row adds one
    // reference to that value; a second output reuses the text row.
    let field = store_value(&tx, &p, "utf8", b"synthetic-output").unwrap();
    let text = output_text(&tx, &p, "synthetic-output").unwrap();
    assert_eq!(output_text(&tx, &p, "synthetic-output").unwrap(), text);
    assert_eq!(
        tx.query_row("SELECT value_pk FROM text WHERE id=?1", [text], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        field
    );
    assert_eq!(
        tx.query_row(
            "SELECT incoming FROM body_value WHERE pk=?1",
            [field],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        2
    );
    // Dropping one reference leaves the value readable; rollback undoes
    // every reference change.
    tx.execute(
        "UPDATE body_value SET incoming=incoming-1 WHERE pk IN(?1,?2)",
        params![first, field],
    )
    .unwrap();
    assert_eq!(
        read_value(&tx, &p, first, 4096).unwrap(),
        br#"{"synthetic":true}"#
    );
    assert_eq!(
        read_value(&tx, &p, field, 4096).unwrap(),
        b"synthetic-output"
    );
    tx.rollback().unwrap();
    assert_eq!(
        writer
            ._index
            .query_row("SELECT count(*) FROM body_value", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
}
