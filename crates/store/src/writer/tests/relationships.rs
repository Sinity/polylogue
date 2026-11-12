use super::*;

fn relationship_version(w: &TestArchive) -> polylogue_model::BlobHash {
    let pool = ReadPool::open(&w.archive).unwrap();
    pool.reader().unwrap().lineage_version().unwrap()
}

#[test]
fn exact_chain_replacement_keeps_the_relationship_revision_but_retraction_changes_it() {
    let w = test_archive();
    let mut original = batch(
        "relations",
        Route::Live,
        0,
        vec![row(
            0,
            text_message("m1", "child", Role::User, "hello", None),
        )],
        &["parent", "child"],
    );
    original.parse[0].output.links.push(polylogue_model::Link {
        parent: sid("parent"),
        child: sid("child"),
        kind: LinkKind::Continuation,
        copies_prefix: false,
        how: None,
        at: None,
        call_id: None,
        evidence: "synthetic explicit marker".into(),
    });
    let empty = relationship_version(&w);
    let stored = w
        .store_reading(original.clone(), Some(&SOURCE_RECORD))
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let before = relationship_version(&w);
    assert_ne!(before, empty);
    let mut replacement = original.clone();
    let mut reading = ChainReading::new().unwrap();
    reading.push(&mut replacement.parse[0]).unwrap();
    w.writer
        .sender()
        .unwrap()
        .replace_chain(replacement.input, stored.chain, replacement.length, reading)
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    assert_eq!(
        relationship_version(&w),
        before,
        "exact reparse preserves links cursors"
    );
    let mut removed = original;
    removed.parse[0].output.links.clear();
    let mut reading = ChainReading::new().unwrap();
    reading.push(&mut removed.parse[0]).unwrap();
    w.writer
        .sender()
        .unwrap()
        .replace_chain(removed.input, stored.chain, removed.length, reading)
        .unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    assert_ne!(relationship_version(&w), before);
}
