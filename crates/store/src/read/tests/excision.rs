use super::*;
use crate::archive::{cache, open_write};

#[test]
fn mismatched_excision_snapshots_refuse_and_release_their_reader_slots() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let pool = ReadPool::with_budget(&fixture.archive, 2, 8 << 20).unwrap();
    let admitted = pool.reader().unwrap();
    admitted
        .query("SELECT count(*) FROM a.excision", &[])
        .unwrap();
    let reserved = pool.reader().unwrap();
    let archive = open_write(&fixture.archive.archive_db(), cache::SETUP).unwrap();
    archive.execute("INSERT INTO excision(id,scope,created_at,commit_seq) VALUES ('remove','session:claude-code:gone',1,1)",[]).unwrap();
    assert!(matches!(
        reserved.status(),
        Err(StoreError::ExcisionPending)
    ));
    drop(reserved);
    for _ in 0..4 {
        let refused = pool.reader().unwrap();
        assert!(matches!(refused.status(), Err(StoreError::ExcisionPending)));
        assert!(matches!(
            refused.query("SELECT 42", &[]),
            Err(StoreError::ExcisionPending)
        ));
        drop(refused);
        let state = pool.state.lock().unwrap();
        assert_eq!(state.live, 1);
        assert_eq!(state.idle.len(), 0);
    }
    assert_eq!(
        admitted
            .query("SELECT count(*) AS n FROM a.excision", &[])
            .unwrap()[0]["n"],
        0,
        "admitted reads keep the same archive and index snapshots"
    );
    drop(admitted);
    assert!(pool.state.lock().unwrap().owners.is_empty());
    // This fixture has no derived rows; acknowledging its index is sufficient.
    let index = open_write(&fixture.archive.index_db(), cache::SETUP).unwrap();
    crate::excise::recovery::acknowledge(&archive, &index).unwrap();
    let recovered = pool.reader().unwrap();
    assert_eq!(
        recovered
            .query("SELECT count(*) AS n FROM a.excision", &[])
            .unwrap()[0]["n"],
        1
    );
}
