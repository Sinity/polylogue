use super::*;
use crate::{
    Archive, ReadPool,
    archive::{cache, open_write},
};

struct World {
    dir: tempfile::TempDir,
    archive: Archive,
    _lock: crate::ArchiveLock,
}
impl World {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let (archive, lock) = Archive::open_or_create(&dir.path().join("a")).unwrap();
        Self {
            dir,
            archive,
            _lock: lock,
        }
    }
    fn canonical(&self) -> Connection {
        open_write(&self.archive.archive_db(), cache::SETUP).unwrap()
    }
    fn vectors(&self) -> Connection {
        open_write(&self.archive.embeddings_db(), cache::SETUP).unwrap()
    }
    fn buy(&self, hash: u8) {
        self.vectors().execute("INSERT INTO embedded(text_hash,model,chunk,dims,vector,embedded_at) VALUES (?1,'test',0,1,x'0000803f',1) ON CONFLICT(text_hash,model,chunk) DO NOTHING",[vec![hash;32]]).unwrap();
    }
    fn record(&self, id: &str, hashes: &[u8]) {
        let c = self.canonical();
        c.execute(
            "INSERT INTO excision(id,scope,created_at,commit_seq) VALUES (?1,'text:synthetic',1,1)",
            [id],
        )
        .unwrap();
        let json = serde_json::to_string(
            &hashes
                .iter()
                .map(|h| crate::sinex_rows::hex(&[*h; 32]))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        c.execute(
            "INSERT INTO vector_deletion_policy VALUES (?1,1,?2)",
            params![id, json],
        )
        .unwrap();
        crate::excise::recovery::acknowledge(
            &c,
            &open_write(&self.archive.index_db(), cache::SETUP).unwrap(),
        )
        .unwrap();
    }
    fn recover(&self) -> Result<()> {
        self.archive.reconcile_vectors(&mut self.canonical())
    }
    fn count(&self, hash: u8) -> i64 {
        self.vectors()
            .query_row(
                "SELECT count(*) FROM embedded WHERE text_hash=?1",
                [vec![hash; 32]],
                |r| r.get(0),
            )
            .unwrap()
    }
    fn backup(&self, name: &str) -> std::path::PathBuf {
        let path = self.dir.path().join(name);
        self.vectors()
            .execute("VACUUM INTO ?1", [path.to_str().unwrap()])
            .unwrap();
        path
    }
    fn restore(&self, path: &std::path::Path) {
        self.vectors()
            .execute_batch("PRAGMA wal_checkpoint(TRUNCATE)")
            .unwrap();
        std::fs::copy(path, self.archive.embeddings_db()).unwrap();
    }
}

#[test]
fn a_pre_policy_backup_replays_exact_deletions_after_pending_was_cleared() {
    let w = World::new();
    w.buy(1);
    w.buy(2);
    w.vectors()
        .execute(
            "INSERT INTO embed_failure VALUES (?1,'test','synthetic',1)",
            [vec![1; 32]],
        )
        .unwrap();
    let before = w.backup("before.db");
    w.record("remove", &[1]);
    w.recover().unwrap();
    assert_eq!(w.count(1), 0);
    assert_eq!(w.count(2), 1);
    assert!(!crate::excise::vector_cleanup::pending(&w.canonical()).unwrap());
    w.restore(&before);
    let pool = ReadPool::open(&w.archive).unwrap();
    let r = pool.reader().unwrap();
    assert!(matches!(r.vectors(), Err(StoreError::ExcisionPending)));
    drop(r);
    drop(pool);
    w.recover().unwrap();
    assert_eq!(w.count(1), 0);
    assert_eq!(w.count(2), 1);
    assert_eq!(
        w.vectors()
            .query_row("SELECT count(*) FROM embed_failure", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn matching_receipts_preserve_same_text_repurchases_and_ignore_publication_changes() {
    let w = World::new();
    w.buy(1);
    w.record("remove", &[1]);
    w.recover().unwrap();
    let pool = ReadPool::open(&w.archive).unwrap();
    let r = pool.reader().unwrap();
    r.vectors_writer()
        .unwrap()
        .put(
            "new",
            &[crate::vectors::Vector {
                text_hash: vec![1; 32],
                chunk: 0,
                values: vec![1.0],
            }],
        )
        .unwrap();
    drop(r);
    drop(pool);
    let after = w.backup("after.db");
    w.restore(&after);
    w.canonical()
        .execute(
            "UPDATE excision SET reason='changed publication metadata' WHERE id='remove'",
            [],
        )
        .unwrap();
    crate::excise::recovery::acknowledge(
        &w.canonical(),
        &open_write(&w.archive.index_db(), cache::SETUP).unwrap(),
    )
    .unwrap();
    w.recover().unwrap();
    assert_eq!(w.count(1), 1);
}

#[test]
fn missing_and_conflicting_receipts_refuse_direct_reads_and_conflicts_roll_back_deletes() {
    for mutation in [
        "DELETE FROM vector_policy_receipt",
        "UPDATE vector_policy_receipt SET policy_hash=zeroblob(32)",
        "UPDATE vector_policy_receipt SET archive_id='different'",
    ] {
        let w = World::new();
        w.record("remove", &[1]);
        w.recover().unwrap();
        w.buy(1);
        w.vectors().execute(mutation, []).unwrap();
        let pool = ReadPool::open(&w.archive).unwrap();
        let r = pool.reader().unwrap();
        assert!(r.vectors().is_err());
        drop(r);
        drop(pool);
        if mutation.starts_with("DELETE") {
            w.recover().unwrap();
            assert_eq!(w.count(1), 0);
        } else {
            assert!(w.recover().is_err());
            assert_eq!(w.count(1), 1);
        }
    }
}

#[test]
fn known_empty_and_legacy_unknown_coverage_are_distinct() {
    let w = World::new();
    w.buy(1);
    w.record("known-empty", &[]);
    w.canonical().execute("INSERT INTO excision(id,scope,created_at,commit_seq) VALUES ('legacy','text:unknown',1,2)",[]).unwrap();
    crate::excise::recovery::acknowledge(
        &w.canonical(),
        &open_write(&w.archive.index_db(), cache::SETUP).unwrap(),
    )
    .unwrap();
    w.recover().unwrap();
    assert_eq!(w.count(1), 1);
    assert_eq!(
        w.vectors()
            .query_row("SELECT count(*) FROM vector_policy_receipt", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        1
    );
    let pool = ReadPool::open(&w.archive).unwrap();
    let r = pool.reader().unwrap();
    assert_eq!(r.vectors().unwrap().status().unwrap().legacy_excisions, 1);
}

#[test]
fn noncanonical_or_changed_policies_refuse_without_deleting_paid_vectors() {
    for json in [
        "[\"00\"]",
        "[\"FF00000000000000000000000000000000000000000000000000000000000000\"]",
        "[ ]",
    ] {
        assert!(digest(1, json).is_err());
    }
    assert!(digest(2, "[]").is_err());
    let w = World::new();
    w.record("remove", &[1]);
    w.recover().unwrap();
    w.buy(2);
    w.canonical()
        .execute(
            "UPDATE vector_deletion_policy SET hashes_json=?1 WHERE excision_id='remove'",
            [serde_json::to_string(&vec![crate::sinex_rows::hex(&[2; 32])]).unwrap()],
        )
        .unwrap();
    assert!(w.recover().is_err());
    assert_eq!(w.count(2), 1);
}

#[test]
fn deletion_and_receipt_roll_back_together() {
    let w = World::new();
    w.buy(1);
    w.record("remove", &[1]);
    let mut v = w.vectors();
    let tx = v.transaction().unwrap();
    reconcile(&tx, &w.canonical()).unwrap();
    tx.rollback().unwrap();
    assert_eq!(w.count(1), 1);
    assert_eq!(
        w.vectors()
            .query_row("SELECT count(*) FROM vector_policy_receipt", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    w.recover().unwrap();
    assert_eq!(w.count(1), 0);
}

#[test]
fn a_receipt_edit_invalidates_an_empty_policy_watermark_too() {
    let w = World::new();
    w.buy(1);
    w.vectors()
        .execute(
            "INSERT INTO vector_policy_receipt VALUES ('alien','alien',zeroblob(32))",
            [],
        )
        .unwrap();
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    assert!(matches!(reader.vectors(), Err(StoreError::ExcisionPending)));
    assert!(w.recover().is_err());
    assert_eq!(w.count(1), 1);
}

#[test]
fn read_snapshots_and_held_writers_keep_the_existing_excision_fence() {
    let w = World::new();
    w.buy(1);
    let pool = ReadPool::open(&w.archive).unwrap();
    let old = pool.reader().unwrap();
    let vectors = old.vectors().unwrap();
    assert_eq!(vectors.status().unwrap().keys[0].vectors, 1);
    let writer_reader = pool.reader().unwrap();
    let mut writer = writer_reader.vectors_writer().unwrap();
    w.record("remove", &[1]);
    w.recover().unwrap();
    assert_eq!(
        vectors.status().unwrap().keys[0].vectors,
        1,
        "admitted snapshot stays pinned"
    );
    assert!(matches!(
        writer.put(
            "test",
            &[crate::vectors::Vector {
                text_hash: vec![1; 32],
                chunk: 0,
                values: vec![1.0]
            }]
        ),
        Err(StoreError::ExcisionPending)
    ));
    assert!(matches!(
        writer.fail("test", &[1; 32], "late"),
        Err(StoreError::ExcisionPending)
    ));
}

#[test]
fn startup_reconciles_a_historical_embeddings_file_without_pending_rows() {
    let w = World::new();
    w.buy(1);
    let backup = w.backup("old.db");
    w.record("remove", &[1]);
    w.recover().unwrap();
    w.restore(&backup);
    let World {
        dir,
        archive,
        _lock,
    } = w;
    drop(_lock);
    let (opened, _lock) = Archive::open_or_create(&archive.root).unwrap();
    let count = open_write(&opened.embeddings_db(), cache::SETUP)
        .unwrap()
        .query_row("SELECT count(*) FROM embedded", [], |r| r.get::<_, i64>(0))
        .unwrap();
    assert_eq!(count, 0);
    drop(dir);
}

#[test]
fn a_foreign_embeddings_backup_refuses_before_any_deletion() {
    let foreign = World::new();
    foreign.record("remove", &[1]);
    foreign.recover().unwrap();
    foreign.buy(2);
    let backup = foreign.backup("foreign.db");
    let local = World::new();
    local.record("remove", &[1]);
    local.recover().unwrap();
    local.restore(&backup);
    assert!(local.recover().is_err());
    assert_eq!(local.count(2), 1);
    let pool = ReadPool::open(&local.archive).unwrap();
    assert!(pool.reader().unwrap().vectors().is_err());
}

#[test]
fn an_old_admitted_reader_cannot_open_a_different_canonical_vector_snapshot() {
    let w = World::new();
    let pool = ReadPool::open(&w.archive).unwrap();
    let old = pool.reader().unwrap();
    old.status().unwrap();
    w.record("remove", &[]);
    w.recover().unwrap();
    assert!(matches!(old.vectors(), Err(StoreError::ExcisionPending)));
    assert!(pool.reader().unwrap().vectors().is_ok());
}

#[test]
fn a_later_excision_cannot_stamp_over_missing_older_receipts() {
    let w = World::new();
    w.buy(1);
    w.buy(2);
    let backup = w.backup("before.db");
    w.record("remove", &[1]);
    w.recover().unwrap();
    w.restore(&backup);
    let writer = crate::Writer::start(&w.archive).unwrap();
    let store = std::sync::Arc::new(
        hashed_blobs::Store::open(w.archive.root.clone(), vec![], 0o600).unwrap(),
    );
    writer
        .excise(
            crate::excise::ExciseRequest {
                id: "later".into(),
                scope: crate::excise::Scope::Text("synthetic absent text".into()),
                reason: None,
                dry_run: false,
            },
            store,
        )
        .unwrap();
    drop(writer);
    assert_eq!(w.count(1), 0);
    assert_eq!(w.count(2), 1);
    assert_eq!(
        w.vectors()
            .query_row("SELECT count(*) FROM vector_policy_receipt", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        2
    );
}

#[test]
fn writable_reads_recheck_changed_receipts_and_recover_after_errors() {
    let world = World::new();
    world.record("empty", &[]);
    world.recover().unwrap();
    world.buy(1);
    let pool = ReadPool::open(&world.archive).unwrap();
    let reader = pool.reader().unwrap();
    let mut vectors = reader.vectors_writer().unwrap();
    for _ in 0..2 {
        assert_eq!(vectors.done("test").unwrap().len(), 1);
        assert_eq!(vectors.nearest("test", &[1.0], 1).unwrap().len(), 1);
        assert_eq!(vectors.status().unwrap().keys[0].texts, 1);
    }
    world
        .vectors()
        .execute("DELETE FROM vector_policy_receipt", [])
        .unwrap();
    assert!(matches!(
        vectors.done("test"),
        Err(StoreError::ExcisionPending)
    ));
    assert!(matches!(
        vectors.nearest("test", &[1.0], 1),
        Err(StoreError::ExcisionPending)
    ));
    assert!(matches!(vectors.status(), Err(StoreError::ExcisionPending)));
    world.recover().unwrap();
    assert_eq!(vectors.done("test").unwrap().len(), 1);
    vectors
        .put(
            "test",
            &[crate::vectors::Vector {
                text_hash: vec![2; 32],
                chunk: 0,
                values: vec![1.0],
            }],
        )
        .unwrap();
    vectors.fail("test", &[3; 32], "synthetic").unwrap();
    assert_eq!(vectors.done("test").unwrap().len(), 3);
}
