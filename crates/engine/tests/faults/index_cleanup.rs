use super::*;
use polylogue_store::Archive;

#[test]
fn interrupted_index_disposal_retries_without_losing_retained_content() {
    for point in ["index_cleanup.after_sidecars", "index_cleanup.after_unlink"] {
        fault::clear();
        let w = Morrow::build(Set::Main);
        let before = w.view();
        let old = w.root.join("index.db.v26");
        let index = rusqlite::Connection::open(w.root.join("index.db")).unwrap();
        index
            .execute("VACUUM INTO ?1", [old.to_str().unwrap()])
            .unwrap();
        drop(index);
        fault::arm(point, "index.db.v26", Action::Fail, 0, 1);
        assert!(Archive::open_or_create(&w.root).is_err(), "{point}");
        fault::clear();
        w.converge();
        assert_eq!(w.view(), before, "{point}");
        assert!(!old.exists());
        assert!(!w.root.join("index.db.v26-wal").exists());
        assert!(!w.root.join("index.db.v26-shm").exists());
        w.converge();
        assert_eq!(w.view(), before);
    }
}
