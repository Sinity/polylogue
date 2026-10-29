use super::*;
#[test]
fn an_unpublished_old_generation_is_rebuilt_before_the_next_activation() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let fixture =
        polylogue_store::fixture::Fixture::new(&directory.path().join("archive")).unwrap();
    let reads = fixture.reads().unwrap();
    let reader = reads.reader().unwrap();
    let current = reader.change_position().unwrap();
    let pointer = completion::CatalogPointer::new(path.clone());
    let old = Metadata {
        archive: "synthetic".into(),
        position: ChangePosition {
            generation: "retired-generation".into(),
            index: 0,
            archive: 0,
        },
        written_at: 1,
    };
    let writer = CatalogWriter::build(
        &pointer.generation_path(&old.position.generation),
        &old,
        |writer| {
            writer.put_catalog(&super::super::base_catalog())?;
            writer.put_values(
                "old",
                &BTreeMap::from([(
                    "repo".into(),
                    BTreeMap::from([("old-scope-value".into(), 1)]),
                )]),
            )
        },
    )
    .unwrap();
    let mut state = State::new(false);
    // A build may finish before its activation is rejected by the scope fence.
    state.writer = Some((old.position.generation, writer));
    assert!(state.position.is_none());
    state.refresh(&reader, &path, "synthetic").unwrap();
    let cache = completion::CatalogReader::open(&path, "synthetic")
        .unwrap()
        .unwrap();
    assert_eq!(cache.metadata.position, current);
    assert!(cache.values("repo", "").unwrap().is_empty());
}
