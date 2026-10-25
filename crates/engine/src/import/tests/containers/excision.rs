use super::*;

#[test]
fn removing_a_member_removes_its_zip_and_keeps_the_other_member_across_rebuild() {
    let w = ClaudeHome::new();
    let (path, original) = archive(&w);
    let original_id = blake3::hash(&original).to_hex().to_string();
    let selected = blake3::hash(b"first member").to_hex().to_string();
    let sibling = blake3::hash(b"second member").to_hex().to_string();
    {
        let (mut e, _lock) = engine(&w, 1024);
        assert!(
            e.import(std::slice::from_ref(&path))
                .unwrap()
                .pass
                .errors
                .is_empty()
        );
        let removed = e
            .excise(ExciseRequest {
                id: "selected-member".into(),
                scope: ExciseScope::Input(selected.clone()),
                reason: None,
                dry_run: false,
            })
            .unwrap();
        assert_eq!(
            removed
                .inputs_removed
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>(),
            [selected.clone(), original_id.clone()]
                .into_iter()
                .collect()
        );
        assert!(
            !e.blobs
                .contains(&hashed_blobs::Hash(*blake3::hash(&original).as_bytes()))
        );
        assert!(!e.blobs.contains(&hashed_blobs::Hash(
            *blake3::hash(b"first member").as_bytes()
        )));
        assert!(
            e.import(std::slice::from_ref(&path))
                .unwrap()
                .imported
                .is_empty()
        );
    }
    std::fs::remove_file(path).unwrap();
    polylogue_testkit::delete_index(&w.root);
    let (mut e, _lock) = engine(&w, 1024);
    assert!(e.converge().unwrap().errors.is_empty());
    let reader = e.reads.reader().unwrap();
    assert!(reader.input_progress(&original_id).unwrap().is_none());
    assert!(reader.input_progress(&selected).unwrap().is_none());
    let retained = reader.input_progress(&sibling).unwrap().unwrap();
    assert_eq!(
        reader
            .stored_range(&sibling, retained.chain, 0, retained.length, &e.blobs)
            .unwrap(),
        b"second member"
    );
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.observation_container"),
        0
    );
}

#[test]
fn nested_zips_are_all_removed_and_members_at_both_levels_stay() {
    use std::io::Write;
    let w = ClaudeHome::new();
    let (inner, inner_bytes) = archive(&w);
    let outer = w.dir.path().join("outer.zip");
    let mut zip = zip::ZipWriter::new(std::fs::File::create(&outer).unwrap());
    for (name, body) in [
        ("inner.zip", inner_bytes.as_slice()),
        ("outer-sibling.dat", b"outer sibling".as_slice()),
    ] {
        zip.start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.write_all(body).unwrap();
    }
    zip.finish().unwrap();
    let outer_bytes = std::fs::read(&outer).unwrap();
    let (mut e, _lock) = engine(&w, 1024);
    assert!(e.import(&[outer, inner]).unwrap().pass.errors.is_empty());
    let removed = e
        .excise(ExciseRequest {
            id: "nested-real".into(),
            scope: ExciseScope::Input(blake3::hash(b"first member").to_hex().to_string()),
            reason: None,
            dry_run: false,
        })
        .unwrap();
    assert_eq!(
        removed
            .zips_removed
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>(),
        [
            blake3::hash(&inner_bytes).to_hex().to_string(),
            blake3::hash(&outer_bytes).to_hex().to_string()
        ]
        .into_iter()
        .collect()
    );
    let reader = e.reads.reader().unwrap();
    for body in [b"second member".as_slice(), b"outer sibling".as_slice()] {
        let id = blake3::hash(body).to_hex().to_string();
        let retained = reader.input_progress(&id).unwrap().unwrap();
        assert_eq!(
            reader
                .stored_range(&id, retained.chain, 0, retained.length, &e.blobs)
                .unwrap(),
            body
        );
    }
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.observation_container"),
        0
    );
}

#[test]
fn actual_watched_copy_of_original_refuses_member_removal_before_any_mutation() {
    let w = ClaudeHome::new();
    let (path, mut original) = archive(&w);
    // A complete watched line must include the entire original byte stream.
    let end = original
        .windows(4)
        .rposition(|b| b == b"PK\x05\x06")
        .unwrap();
    let comment = u16::from_le_bytes(original[end + 20..end + 22].try_into().unwrap());
    original.extend_from_slice(b"\n");
    original[end + 20..end + 22].copy_from_slice(&(comment + 1).to_le_bytes());
    std::fs::write(&path, &original).unwrap();
    let hash = blake3::hash(&original);
    let (mut e, _lock) = engine(&w, 64 << 20);
    std::fs::write(
        w.file("unlinked.jsonl"),
        polylogue_testkit::claude_user("watched", "unlinked", "unrelated initial content"),
    )
    .unwrap();
    assert!(e.converge().unwrap().errors.is_empty());
    assert!(e.import(&[path]).unwrap().pass.errors.is_empty());
    // The watched source records these bytes even though they are not Claude JSONL.
    std::fs::write(w.file("unlinked.jsonl"), &original).unwrap();
    let _pass = e.converge().unwrap();
    let before = w.count("SELECT count(*) AS n FROM a.extent");
    let references = e
        .reads
        .reader()
        .unwrap()
        .query(
            "SELECT count(DISTINCT input) AS n FROM a.extent WHERE lower(hex(blob))=?1",
            &[polylogue_store::SqlValue::Text(hash.to_hex().to_string())],
        )
        .unwrap();
    assert_eq!(
        references[0]["n"], 2,
        "imported original and unlinked watched source must both own the same bytes; pass={_pass:?}; extents={:?}",
        e.reads.reader().unwrap().query("SELECT i.format,e.input,e.offset,e.length,lower(hex(e.blob)) AS blob FROM a.extent e JOIN a.input i ON i.id=e.input",&[]).unwrap()
    );
    let result = e.excise(ExciseRequest {
        id: "watched-refusal".into(),
        scope: ExciseScope::Input(blake3::hash(b"first member").to_hex().to_string()),
        reason: None,
        dry_run: false,
    });
    assert!(matches!(
        result,
        Err(EngineError::Store(
            polylogue_store::StoreError::ExcisionFormat { .. }
        ))
    ));
    assert_eq!(w.count("SELECT count(*) AS n FROM a.extent"), before);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.excision"), 0);
    let reader = e.reads.reader().unwrap();
    let id = hash.to_hex().to_string();
    let retained = reader.input_progress(&id).unwrap().unwrap();
    assert_eq!(
        reader
            .stored_range(&id, retained.chain, 0, retained.length, &e.blobs)
            .unwrap(),
        original
    );
}
