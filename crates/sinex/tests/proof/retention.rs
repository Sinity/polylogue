use super::*;
use polylogue_store::{
    StoreError,
    fault::{self, Action},
    retention::UnfinishedState,
};

/// Exercise unfinished local custody against the same real Sinex as the
/// ordinary restore proof. Only an exact retry or verified installed bytes
/// can turn the reservation into a publishable source.
pub(super) async fn verify(sinex: &mut Sinex, root: &Path, home: &Path, scratch: &Path) {
    let id = polylogue_store::sinex_rows::Published::open(root)
        .unwrap()
        .archive_id()
        .unwrap();
    for point in ["import.reserved", "import.installed"] {
        let text = format!("synthetic native custody {point}");
        let bytes = polylogue_testkit::claude_user(point, point, &text);
        let path = scratch.join("pending.jsonl");
        fs::write(&path, &bytes).unwrap();
        let input = blake3::hash(bytes.as_bytes()).to_hex().to_string();
        let (archive, lock) = Archive::open_or_create(root).unwrap();
        let mut engine = Engine::new(
            Config {
                home: home.into(),
                tool_dirs: Default::default(),
                workers: 2,
                reader_memory_bytes: 128 << 20,
                bytes_in_flight: 16 << 20,
                machine: "proof-machine".into(),
                host: "proof".into(),
                chunk: 64 << 20,
                blob_roots: vec![],
            },
            archive.clone(),
        )
        .unwrap();
        fault::arm(point, &input, Action::Fail, 0, 1);
        assert!(
            !engine
                .import(std::slice::from_ref(&path))
                .unwrap()
                .pass
                .errors
                .is_empty()
        );
        drop(engine);
        fs::remove_file(&path).unwrap();
        assert!(
            sinex.publish(root).await.is_err(),
            "{point}: unfinished unit published"
        );
        assert!(
            sinex.restore(&archive, &id).await.is_err(),
            "{point}: confirmed prefix discarded local custody"
        );
        let pool = ReadPool::open(&archive).unwrap();
        let reader = pool.reader().unwrap();
        let reserved = reader.retained_input(&input).unwrap().unwrap();
        assert!(matches!(
            reader.stored_chain(&input, reserved.chain, reserved.length),
            Err(StoreError::InputUnfinished { .. })
        ));
        drop(reader);
        drop(pool);
        drop(lock);
        if point == "import.reserved" {
            converge(root, home, vec![], |engine| {
                let pass = engine.converge().unwrap();
                assert_eq!(pass.unfinished_inputs.len(), 1);
                assert_eq!(pass.unfinished_inputs[0].state, UnfinishedState::Missing);
                fs::write(&path, &bytes).unwrap();
                assert!(
                    engine
                        .import(std::slice::from_ref(&path))
                        .unwrap()
                        .pass
                        .errors
                        .is_empty()
                );
            });
            fs::remove_file(&path).unwrap();
        } else {
            converge(root, home, vec![], |_| {});
        }
        assert!(answers(root).contains(&text));
        let published = sinex.publish(root).await.unwrap();
        assert!(
            published.units > 0,
            "{point}: recovered source never published"
        );
    }
}
