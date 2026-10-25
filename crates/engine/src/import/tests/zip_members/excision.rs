use super::*;

#[test]
fn a_file_read_from_a_zip_and_the_zip_itself_refuse_excision() {
    let w = ClaudeHome::new();
    let (path, original) = archive(&w);
    let original_id = blake3::hash(&original).to_hex().to_string();
    let (mut e, _lock) = engine(&w, 1024);
    assert!(
        e.import(std::slice::from_ref(&path))
            .unwrap()
            .pass
            .errors
            .is_empty()
    );
    let stored = |e: &Engine| {
        let reader = e.reads.reader().unwrap();
        [
            original.as_slice(),
            b"first member".as_slice(),
            b"second member".as_slice(),
        ]
        .map(|body| {
            let id = blake3::hash(body).to_hex().to_string();
            let progress = reader.input_progress(&id).unwrap().unwrap();
            reader
                .stored_range(&id, progress.chain, 0, progress.length, &e.blobs)
                .unwrap()
        })
    };
    let before = (
        stored(&e),
        w.count("SELECT count(*) AS n FROM a.observation_zip"),
    );
    assert_eq!(before.0[0], original);
    for target in [b"first member".as_slice(), original.as_slice()] {
        for dry_run in [true, false] {
            let refused = e
                .excise(ExciseRequest {
                    id: format!("refused-{dry_run}"),
                    target: ExciseTarget::Input(blake3::hash(target).to_hex().to_string()),
                    reason: None,
                    dry_run,
                })
                .unwrap_err();
            assert!(
                matches!(
                    &refused,
                    EngineError::Store(polylogue_store::StoreError::ExcisionInsideZip { input, file })
                        if *input == original_id && file.ends_with("export.zip")
                ),
                "{refused}"
            );
            assert_eq!(
                (
                    stored(&e),
                    w.count("SELECT count(*) AS n FROM a.observation_zip")
                ),
                before
            );
            assert_eq!(w.count("SELECT count(*) AS n FROM a.excision"), 0);
        }
    }
}
