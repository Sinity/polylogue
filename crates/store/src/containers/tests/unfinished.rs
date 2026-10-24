use super::*;
use crate::fault::{self, Action};
use crate::retention::UnfinishedState;
use crate::sinex_rows::{Catalog, Published, Restoring};

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
        let source = Published::open(&archive.root).unwrap();
        let id = source.archive_id().unwrap();
        let inventory = source.inventory().unwrap();
        let mut refused = 0;
        for position in 1..=inventory.len() {
            let unit = inventory.unit(position).unwrap();
            if matches!(source.read(&unit), Err(StoreError::InputUnfinished { .. })) {
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
        drop(inventory);
        drop(source);
        drop(writer);
        let selected = Catalog::new().unwrap().finish().unwrap();
        assert!(
            Restoring::open(&archive)
                .unwrap()
                .install::<std::io::Empty>(&id, &selected, |_| Ok(None))
                .is_err(),
            "older confirmed state must not discard linked reservations"
        );
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
        let source = Published::open(&archive.root).unwrap();
        let inventory = source.inventory().unwrap();
        for position in 1..=inventory.len() {
            source.read(&inventory.unit(position).unwrap()).unwrap();
        }
    }
}
