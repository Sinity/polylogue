use super::*;
use crate::fault::{self, Action};
use crate::staging::PendingState;
use polylogue_model::BlobHash;

#[test]
fn a_zip_group_stopped_before_or_after_installing_finishes_or_reports() {
    for point in ["import.install", "import.installed"] {
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
            .store_group(linked(&archive, "pending", false), Arc::clone(&store))
            .unwrap();
        assert!(outcomes.iter().any(Result::is_err), "{point}: {outcomes:?}");
        fault::clear();
        drop(writer);
        let writer = Writer::start(&archive).unwrap();
        let finished = writer.finish_writing(Arc::clone(&store)).unwrap();
        if point == "import.install" {
            assert_eq!(finished.pending.len(), 1);
            assert_eq!(finished.pending[0].blob, BlobHash::of(ORIGINAL));
            assert_eq!(finished.pending[0].state, PendingState::Missing);
            let outcomes = writer
                .sender()
                .unwrap()
                .store_group(linked(&archive, "exact-retry", false), Arc::clone(&store))
                .unwrap();
            assert!(outcomes.iter().all(Result::is_ok), "{outcomes:?}");
        } else {
            assert!(finished.pending.is_empty());
        }
    }
}
