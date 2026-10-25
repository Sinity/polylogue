use super::*;
use crate::archive::open_read;
use crate::excise::{ExciseRequest, Scope};
use polylogue_model::BlobHash;

fn id(body: &[u8]) -> String {
    BlobHash::of(body).to_string()
}

fn write_group(writer: &Writer, store: &Arc<hashed_blobs::Store>, group: Vec<PreparedBatch>) {
    let results = writer
        .sender()
        .unwrap()
        .store_group(group, Arc::clone(store))
        .unwrap();
    assert!(results.iter().all(Result::is_ok), "{results:?}");
}

/// Every archive row an excision could touch, and the write counter.
fn rows(archive: &Archive) -> Vec<String> {
    let db = open_read(&archive.archive_db()).unwrap();
    let mut out = vec![
        db.query_row(
            "SELECT value FROM meta WHERE key='write_counter'",
            [],
            |r| r.get(0),
        )
        .unwrap(),
    ];
    for table in [
        "input",
        "revision",
        "extent",
        "blob",
        "observation",
        "observation_container",
        "excision",
    ] {
        let mut stmt = db
            .prepare(&format!("SELECT * FROM {table} ORDER BY 1"))
            .unwrap();
        let columns = stmt.column_count();
        let mut found = stmt.query([]).unwrap();
        while let Some(row) = found.next().unwrap() {
            let values: Vec<String> = (0..columns)
                .map(|i| match row.get_ref(i).unwrap() {
                    rusqlite::types::ValueRef::Blob(bytes) => crate::encoding::hex(bytes),
                    value => format!("{value:?}"),
                })
                .collect();
            out.push(format!("{table}: {}", values.join(",")));
        }
    }
    out
}

#[test]
fn excision_inside_a_zip_refuses_and_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let store = Arc::new(hashed_blobs::Store::open(&archive.root, vec![], 0o600).unwrap());
    write_group(&writer, &store, linked(&archive, "refused", true));
    let before = rows(&archive);
    for target in [MEMBER, ORIGINAL] {
        for dry_run in [true, false] {
            let refused = writer.excise(
                ExciseRequest {
                    id: format!("refused-{dry_run}"),
                    scope: Scope::Input(id(target)),
                    reason: None,
                    dry_run,
                },
                Arc::clone(&store),
            );
            assert!(
                matches!(
                    &refused,
                    Err(StoreError::ExcisionInsideZip { input, file })
                        if *input == id(ORIGINAL) && file == "/synthetic/original.zip"
                ),
                "{refused:?}"
            );
            assert_eq!(rows(&archive), before);
            for body in [ORIGINAL, MEMBER] {
                assert!(store.contains(&hashed_blobs::Hash(BlobHash::of(body).0)));
            }
        }
    }
}

#[test]
fn an_input_beside_a_zip_is_excised_without_touching_the_zip() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let store = Arc::new(hashed_blobs::Store::open(&archive.root, vec![], 0o600).unwrap());
    let unrelated = b"synthetic unrelated input";
    write_group(&writer, &store, linked(&archive, "beside", false));
    write_group(
        &writer,
        &store,
        vec![stage(&archive, unrelated, "other", "other.dat", None)],
    );
    let zip_rows = |archive: &Archive| {
        rows(archive)
            .into_iter()
            .filter(|row| {
                [id(ORIGINAL), id(MEMBER)]
                    .iter()
                    .any(|hash| row.contains(hash.as_str()))
                    || row.starts_with("observation_container")
            })
            .collect::<Vec<_>>()
    };
    let before = zip_rows(&archive);
    let removed = writer
        .excise(
            ExciseRequest {
                id: "beside".into(),
                scope: Scope::Input(id(unrelated)),
                reason: None,
                dry_run: false,
            },
            Arc::clone(&store),
        )
        .unwrap();
    assert_eq!(removed.inputs_removed, [id(unrelated)]);
    assert_eq!(zip_rows(&archive), before);
    assert!(!store.contains(&hashed_blobs::Hash(BlobHash::of(unrelated).0)));
    for body in [ORIGINAL, MEMBER] {
        assert!(store.contains(&hashed_blobs::Hash(BlobHash::of(body).0)));
    }
}
