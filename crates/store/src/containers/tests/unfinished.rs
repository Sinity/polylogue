use super::*;
use crate::fault::{self, Action};
use crate::retention::UnfinishedState;
use crate::row_batches::Snapshot;

#[test]
fn linked_reservations_remain_owned_across_reserved_and_installed_failures() {
    for point in ["import.reserved", "import.installed"] {
        fault::clear();
        let dir = tempfile::tempdir().unwrap();
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        let writer = Writer::start(&archive).unwrap();
        let store = Arc::new(hashed_blobs::Store::open(dir.path(), vec![], 0o600).unwrap());
        fault::arm(
            point,
            &BlobHash::of(ORIGINAL).to_string(),
            Action::Fail,
            0,
            1,
        );
        let outcomes = writer
            .sender()
            .unwrap()
            .retain_group(linked(&archive, "unfinished", false), Arc::clone(&store))
            .unwrap();
        assert!(outcomes.iter().any(Result::is_err), "{point}: {outcomes:?}");
        fault::clear();
        let source = Snapshot::open(&archive.root).unwrap();
        let mut refused = 0;
        for batch in source.batches(None, 10_000).unwrap() {
            if matches!(
                source.rows(&batch, |_| Ok(())),
                Err(StoreError::InputUnfinished { .. })
            ) {
                refused += 1;
            }
        }
        assert!(
            refused >= 2,
            "unfinished original and its member relationship must not publish"
        );
        assert!(matches!(
            remove(&writer, Arc::clone(&store), "unfinished-removal"),
            Err(StoreError::InputUnfinished { .. })
        ));
        drop(source);
        drop(writer);
        let writer = Writer::start(&archive).unwrap();
        let reconciled = writer.reconcile_retained_bytes(Arc::clone(&store)).unwrap();
        if point == "import.reserved" {
            assert_eq!(reconciled.unfinished.len(), 1);
            assert_eq!(reconciled.unfinished[0].blob, BlobHash::of(ORIGINAL));
            assert_eq!(reconciled.unfinished[0].state, UnfinishedState::Missing);
            let outcomes = writer
                .sender()
                .unwrap()
                .retain_group(linked(&archive, "exact-retry", false), Arc::clone(&store))
                .unwrap();
            assert!(outcomes.iter().all(Result::is_ok), "{outcomes:?}");
        } else {
            assert!(reconciled.unfinished.is_empty());
        }
        let source = Snapshot::open(&archive.root).unwrap();
        for batch in source.batches(None, 10_000).unwrap() {
            source.rows(&batch, |_| Ok(())).unwrap();
        }
    }
}
