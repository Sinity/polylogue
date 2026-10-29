use super::*;
use crate::fault::{self, Action};

#[test]
fn a_zip_group_commits_all_member_links_or_retries_after_an_uncommitted_tail() {
    for point in ["segment.after_sync", "segment.after_commit"] {
        fault::clear();
        let dir = tempfile::tempdir().unwrap();
        let (archive, _lock) = Archive::open_or_create(dir.path()).unwrap();
        let writer = Writer::start(&archive).unwrap();
        let store = Arc::new(archive.blobs().unwrap());
        fault::arm(point, "", Action::Fail, 0, 1);
        let outcomes = writer
            .sender()
            .unwrap()
            .store_group(linked(&archive, "interrupted", false), Arc::clone(&store))
            .unwrap();
        assert!(outcomes.iter().all(Result::is_err), "{point}: {outcomes:?}");
        fault::clear();
        let committed = point == "segment.after_commit";
        let db = crate::archive::open_read(&archive.archive_db()).unwrap();
        for table in ["input", "revision", "extent", "blob", "observation"] {
            let count: i64 = db
                .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r.get(0))
                .unwrap();
            assert_eq!(count, if committed { 2 } else { 0 }, "{point} {table}");
        }
        assert_eq!(
            db.query_row("SELECT count(*) FROM observation_zip", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            i64::from(committed)
        );
        drop(writer);
        let writer = Writer::start(&archive).unwrap();
        let results = writer
            .sender()
            .unwrap()
            .store_group(linked(&archive, "exact-retry", false), Arc::clone(&store))
            .unwrap();
        assert!(results.iter().all(Result::is_ok), "{results:?}");
        assert!(
            results
                .iter()
                .all(|result| result.as_ref().unwrap().already_stored == committed)
        );
        assert_eq!(
            db.query_row("SELECT count(*) FROM blob", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            2
        );
        let extent = store
            .extent(&hashed_blobs::Hash(BlobHash::of(ORIGINAL).0))
            .unwrap();
        let length = std::fs::metadata(archive.root.join(&extent.segment))
            .unwrap()
            .len();
        let results = writer
            .sender()
            .unwrap()
            .store_group(linked(&archive, "duplicate", false), store)
            .unwrap();
        assert!(results.iter().all(Result::is_ok));
        assert_eq!(
            std::fs::metadata(archive.root.join(&extent.segment))
                .unwrap()
                .len(),
            length
        );
    }
    fault::clear();
}
