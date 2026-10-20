use super::*;
use crate::fixture::Fixture;

#[test]
fn closed_pools_refuse_new_readers_and_let_admitted_reads_finish() {
    let dir = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    reader.status().unwrap();
    pool.close();
    assert!(matches!(pool.reader(), Err(StoreError::ReadPoolClosed)));
    assert_eq!(
        reader.query("SELECT 42 AS answer", &[]).unwrap()[0]["answer"],
        42
    );
    drop(reader);
    pool.close();
    assert!(matches!(pool.reader(), Err(StoreError::ReadPoolClosed)));
}
