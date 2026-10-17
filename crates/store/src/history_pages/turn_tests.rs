use super::*;

fn observation(id: &str) -> ImportObservation<'_> {
    ImportObservation {
        id,
        machine: "synthetic-machine",
        host: "synthetic-host",
        path: "copy:history",
        seen_at: 1,
    }
}
fn budget() -> turns::Budget {
    turns::Budget {
        bytes: 65536,
        checkpoints: 2,
    }
}
#[test]
fn reusable_turns_keep_one_wal_view_and_retry_no_prefix_after_cancel() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("provider.sqlite");
    let provider = Connection::open(&path).unwrap();
    provider.execute_batch("PRAGMA journal_mode=WAL;PRAGMA wal_autocheckpoint=0;CREATE TABLE facts(value TEXT);INSERT INTO facts VALUES('first');CREATE TABLE left_field(value INTEGER);INSERT INTO left_field VALUES(10);CREATE TABLE padding(value BLOB);INSERT INTO padding VALUES(zeroblob(131072));CREATE TABLE right_field(value INTEGER);INSERT INTO right_field VALUES(10);").unwrap();
    let mut writer = Writer::open(&temp.path().join("archive")).unwrap();
    let capture = writer.blob_capture().unwrap();
    let cancel = AtomicBool::new(false);
    let mut turns = Vec::new();
    let mut changed = false;
    let prepared = capture.capture_history_pages_in_turns(&path, None, &cancel, budget(), |progress| {
        assert!(progress.charged_bytes <= 65536 && progress.checkpoints <= 2);
        if !changed && progress.provider_bytes_read >= 16384 {
            // Separate data pages straddle this quantum. Reopening the view
            // could combine left10 with right20, a never-committed population.
            provider.execute_batch("BEGIN;INSERT INTO facts VALUES('later');UPDATE left_field SET value=20;UPDATE right_field SET value=20;COMMIT;BEGIN IMMEDIATE;INSERT INTO facts VALUES('uncommitted');UPDATE left_field SET value=30;UPDATE right_field SET value=30;").unwrap();
            changed = true;
        }
        turns.push(progress);
        Ok(())
    }).unwrap();
    assert!(turns.len() > 8 && changed);
    let hash = prepared.hash();
    let retained = writer
        .retain_history_pages(prepared, &observation("turns"), 131072)
        .unwrap();
    let raw = writer
        .read_handle()
        .unwrap()
        .read_source_range(&retained.id, 1, hash, 0, retained.length)
        .unwrap();
    let selected = temp.path().join("selected.sqlite");
    std::fs::write(&selected, raw).unwrap();
    let selected = Connection::open(&selected).unwrap();
    assert_eq!(
        selected
            .query_row("SELECT group_concat(value) FROM facts", [], |r| r
                .get::<_, String>(0))
            .unwrap(),
        "first"
    );
    assert_eq!(
        selected
            .query_row(
                "SELECT (SELECT value FROM left_field),(SELECT value FROM right_field)",
                [],
                |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))
            )
            .unwrap(),
        (10, 10)
    );
    provider.execute_batch("ROLLBACK").unwrap();
    let old_reader = writer.read_handle().unwrap();
    let before = writer.status().unwrap().write_counter;
    let refused =
        capture.capture_history_pages_in_turns(&path, None, &cancel, budget(), |progress| {
            let old = old_reader.read_source_range(&retained.id, 1, hash, 0, retained.length)?;
            assert_eq!(blake3::hash(&old).as_bytes(), hash.bytes());
            if progress.provider_bytes_read >= 65536 {
                cancel.store(true, Ordering::Release);
                Err(Error::Cancelled)
            } else {
                Ok(())
            }
        });
    assert!(matches!(refused, Err(Error::Cancelled)));
    assert_eq!(writer.status().unwrap().write_counter, before);
    cancel.store(false, Ordering::Release);
    let retried = capture
        .capture_history_pages_in_turns(&path, None, &cancel, budget(), |_| Ok(()))
        .unwrap();
    let retry_hash = retried.hash();
    let retry = writer
        .retain_history_pages(retried, &observation("retry-turns"), 131072)
        .unwrap();
    let bytes = writer
        .read_handle()
        .unwrap()
        .read_source_range(&retry.id, 1, retry_hash, 0, retry.length)
        .unwrap();
    let recovered = temp.path().join("recovered.sqlite");
    std::fs::write(&recovered, bytes).unwrap();
    assert_eq!(
        Connection::open(recovered)
            .unwrap()
            .query_row("SELECT group_concat(value) FROM facts", [], |r| r
                .get::<_, String>(0))
            .unwrap(),
        "first,later"
    );
    assert_eq!(
        Connection::open(temp.path().join("recovered.sqlite"))
            .unwrap()
            .query_row(
                "SELECT (SELECT value FROM left_field),(SELECT value FROM right_field)",
                [],
                |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))
            )
            .unwrap(),
        (20, 20)
    );
}

#[test]
fn many_turns_preserve_one_base_verification_and_current_removal_fences() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("provider.sqlite");
    let provider = Connection::open(&path).unwrap();
    provider
        .execute_batch(
            "CREATE TABLE padding(value BLOB);INSERT INTO padding VALUES(zeroblob(262144));",
        )
        .unwrap();
    let mut writer = Writer::open(&temp.path().join("archive")).unwrap();
    let capture = writer.blob_capture().unwrap();
    let cancel = AtomicBool::new(false);
    let base = capture.capture_history_pages(&path, None, &cancel).unwrap();
    let base_len = base.length();
    let base_hash = base.hash();
    writer
        .retain_history_pages(base, &observation("base-turns"), 131072)
        .unwrap();
    let basis = writer.history_basis(&observation("next-turns")).unwrap();
    let mut phases = Vec::new();
    let next = capture
        .capture_history_pages_in_turns(&path, basis, &cancel, budget(), |p| {
            phases.push(p.phase);
            Ok(())
        })
        .unwrap();
    assert_eq!(next.base_verification_bytes, base_len);
    assert_eq!(next.hash(), base_hash);
    assert!(phases.iter().filter(|p| **p == "base_verification").count() > 2);
    assert!(phases.contains(&"base_ranges"));
    let basis = writer
        .history_basis(&observation("removed-between-turns"))
        .unwrap();
    let before = writer.status().unwrap().write_counter;
    let stopped = capture.capture_history_pages_in_turns(&path, basis, &cancel, budget(), |p| {
        if p.phase == "base_ranges" {
            writer
                .archive
                .execute(
                    "UPDATE blob SET state='deleting' WHERE hash=?1",
                    [base_hash.bytes().as_slice()],
                )
                .unwrap();
        }
        Ok(())
    });
    assert!(matches!(stopped, Err(Error::UnknownBlob)));
    assert_eq!(writer.status().unwrap().write_counter, before);
}
