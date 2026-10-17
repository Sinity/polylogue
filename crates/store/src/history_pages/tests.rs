use super::*;
#[test]
fn page_view_and_backup_share_frozen_wal_facts_but_only_data_pages_are_identical() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("provider.sqlite");
    let writer = Connection::open(&path).unwrap();
    writer.execute_batch("PRAGMA journal_mode=WAL;PRAGMA wal_autocheckpoint=0;CREATE TABLE evidence(value TEXT);INSERT INTO evidence VALUES('committed-wal');").unwrap();
    let image = Image::open(&path).unwrap();
    writer.execute_batch("INSERT INTO evidence VALUES('later-generation');BEGIN IMMEDIATE;INSERT INTO evidence VALUES('uncommitted');").unwrap();
    let mut backup = Connection::open(temp.path().join("backup.sqlite")).unwrap();
    {
        let b = rusqlite::backup::Backup::new(&image.connection, &mut backup).unwrap();
        assert!(matches!(
            b.step(-1).unwrap(),
            rusqlite::backup::StepResult::Done
        ));
    }
    let mut raw = Vec::new();
    for n in 1..=image.pages {
        raw.extend(image.page(n, &AtomicBool::new(false)).unwrap());
    }
    let selected = temp.path().join("selected.sqlite");
    std::fs::write(&selected, &raw).unwrap();
    let copy = Connection::open(&selected).unwrap();
    for c in [&copy, &backup] {
        assert_eq!(
            c.query_row("SELECT group_concat(value) FROM evidence", [], |r| r
                .get::<_, String>(0))
                .unwrap(),
            "committed-wal"
        );
    }
    let copied = std::fs::read(temp.path().join("backup.sqlite")).unwrap();
    assert_eq!(raw.len(), copied.len());
    let differences = raw
        .iter()
        .zip(&copied)
        .enumerate()
        .filter_map(|(i, (a, b))| (a != b).then_some(i))
        .collect::<Vec<_>>();
    println!("exact backup/header differences: {differences:?}");
    assert_eq!(differences, vec![27, 95]);
    println!(
        "source file-change/valid-for counters={:?}/{:?}; backup counters={:?}/{:?}",
        &raw[24..28],
        &raw[92..96],
        &copied[24..28],
        &copied[92..96]
    );
    assert_eq!(&raw[24..28], &raw[92..96]);
    assert_eq!(&copied[24..28], &copied[92..96]);
    assert!(
        differences.iter().all(|i| [27, 95].contains(i)),
        "unexpected page1 payload differences {differences:?}"
    );
    assert_eq!(
        copy.query_row("PRAGMA quick_check", [], |r| r.get::<_, String>(0))
            .unwrap(),
        "ok"
    );
    println!(
        "same frozen snapshot: selected/read={} temporary backup bytes={} data pages equal={}",
        raw.len(),
        copied.len(),
        image.pages - 1
    );
}
fn observation<'a>(id: &'a str) -> ImportObservation<'a> {
    ImportObservation {
        id,
        machine: "synthetic-machine",
        host: "synthetic-host",
        path: "copy:history",
        seen_at: 1,
    }
}
#[test]
fn flat_page_ranges_reconstruct_new_and_old_snapshots_without_a_dependency_chain() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("provider.sqlite");
    let c = Connection::open(&path).unwrap();
    c.execute_batch("PRAGMA journal_mode=WAL;PRAGMA wal_autocheckpoint=0;CREATE TABLE t(value BLOB);INSERT INTO t VALUES(zeroblob(524288));").unwrap();
    let mut w = Writer::open(&temp.path().join("archive")).unwrap();
    let capture = w.blob_capture().unwrap();
    let first = capture
        .capture_history_pages(&path, None, &AtomicBool::new(false))
        .unwrap();
    let first_hash = first.hash;
    let first_length = first.length;
    let a = w
        .retain_history_pages(first, &observation("a"), 131072)
        .unwrap();
    c.execute_batch("INSERT INTO t VALUES('later');").unwrap();
    let basis = w.history_basis(&observation("b")).unwrap();
    let second = capture
        .capture_history_pages(&path, basis, &AtomicBool::new(false))
        .unwrap();
    assert!(second.changed_bytes < second.read_bytes);
    println!(
        "snapshot read={} changed-page bytes={}",
        second.read_bytes, second.changed_bytes
    );
    let second_hash = second.hash;
    let length = second.length;
    let b = w
        .retain_history_pages(second, &observation("b"), 131072)
        .unwrap();
    let reader = w.read_handle().unwrap();
    let raw = reader
        .read_source_range(&b.id, 1, second_hash, 0, length)
        .unwrap();
    assert_eq!(*blake3::hash(&raw).as_bytes(), *second_hash.bytes());
    let old = reader
        .read_source_range(&a.id, 1, first_hash, 0, first_length)
        .unwrap();
    assert_eq!(*blake3::hash(&old).as_bytes(), *first_hash.bytes());
    let recovered = temp.path().join("recovered.sqlite");
    std::fs::write(&recovered, raw).unwrap();
    let r = Connection::open(&recovered).unwrap();
    assert_eq!(
        r.query_row("SELECT count(*) FROM t", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        2
    );
    let basis = w.history_basis(&observation("c")).unwrap();
    let unchanged = capture
        .capture_history_pages(&path, basis, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(unchanged.changed_bytes, 0);
    assert_eq!(unchanged.hash, second_hash);
}

#[test]
fn raw_snapshot_acquisition_exceeds_interpretation_limit_and_fragmentation_rebases() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("provider.sqlite");
    let c = Connection::open(&path).unwrap();
    c.execute_batch("PRAGMA journal_mode=WAL;PRAGMA wal_autocheckpoint=0;CREATE TABLE t(value BLOB);INSERT INTO t VALUES(zeroblob(17825792));CREATE TABLE small(value BLOB);").unwrap();
    for _ in 0..12 {
        c.execute("INSERT INTO small VALUES(zeroblob(3000))", [])
            .unwrap();
    }
    let mut w = Writer::open(&temp.path().join("archive")).unwrap();
    let capture = w.blob_capture().unwrap();
    let p = capture
        .capture_history_pages(&path, None, &AtomicBool::new(false))
        .unwrap();
    assert!(p.length > 16 * 1024 * 1024);
    let source = w
        .retain_history_pages(p, &observation("large"), 131072)
        .unwrap();
    let retained = w.retained_source(&source.id).unwrap().unwrap();
    assert!(matches!(
        w.read_handle()
            .unwrap()
            .codex_history_rows(&retained, &AtomicBool::new(false)),
        Err(Error::ParserAdmission {
            resource: "physical history snapshot bytes",
            ..
        })
    ));
    c.execute_batch("UPDATE small SET value=randomblob(3000) WHERE rowid%2=0")
        .unwrap();
    let basis = w.history_basis(&observation("fragmented")).unwrap();
    let p = capture
        .capture_history_with_limit(&path, basis, &AtomicBool::new(false), 2)
        .unwrap();
    assert!(p.rebased);
    assert_eq!(p.changed_bytes, p.length);
    assert!(p.read_bytes > p.length);
    assert_eq!(p.parts.len(), 1);
    println!(
        "large raw={}, rebase total source-read={}, retained changed={}",
        p.length, p.read_bytes, p.changed_bytes
    );
    let hash = p.hash;
    let n = p.length;
    let r = w
        .retain_history_pages(p, &observation("fragmented"), 131072)
        .unwrap();
    let bytes = w
        .read_handle()
        .unwrap()
        .read_source_range(&r.id, 1, hash, 0, n)
        .unwrap();
    assert_eq!(blake3::hash(&bytes).as_bytes(), hash.bytes());
}
#[test]
fn prepared_or_failed_parts_never_select_an_incomplete_input() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("provider.sqlite");
    let c = Connection::open(&path).unwrap();
    c.execute_batch("CREATE TABLE t(x);INSERT INTO t VALUES(7)")
        .unwrap();
    let mut w = Writer::open(&temp.path().join("archive")).unwrap();
    let mut p = w
        .blob_capture()
        .unwrap()
        .capture_history_pages(&path, None, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(
        w.archive
            .query_row("SELECT count(*) FROM input", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        0
    );
    p.parts.pop();
    assert!(
        w.retain_history_pages(p, &observation("failed-parts"), 131072)
            .is_err()
    );
    for table in [
        "input",
        "revision",
        "extent",
        "observation",
        "client_request",
    ] {
        assert_eq!(
            w.archive
                .query_row(&format!("SELECT count(*) FROM {table}"), [], |r| r
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }
}

#[test]
fn alternating_flat_base_patch_layout_reuses_two_checked_objects_during_reconstruction() {
    use std::sync::atomic::Ordering;
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("provider.sqlite");
    let c = Connection::open(&path).unwrap();
    c.execute_batch(
        "PRAGMA journal_mode=WAL;PRAGMA wal_autocheckpoint=0;CREATE TABLE t(value BLOB)",
    )
    .unwrap();
    for _ in 0..80 {
        c.execute("INSERT INTO t VALUES(zeroblob(3000))", [])
            .unwrap();
    }
    c.execute_batch(
        "CREATE TABLE padding(value BLOB);INSERT INTO padding VALUES(zeroblob(8388608))",
    )
    .unwrap();
    let mut w = Writer::open(&temp.path().join("archive")).unwrap();
    let capture = w.blob_capture().unwrap();
    let p = capture
        .capture_history_pages(&path, None, &AtomicBool::new(false))
        .unwrap();
    let base_length = p.length;
    w.retain_history_pages(p, &observation("base"), 131072)
        .unwrap();
    c.execute_batch("UPDATE t SET value=randomblob(3000) WHERE rowid%2=0")
        .unwrap();
    let basis = w.history_basis(&observation("fragmented")).unwrap();
    let p = capture
        .capture_history_pages(&path, basis, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(p.base_explicit_verification_bytes, base_length);
    let patch_length = p.changed_bytes;
    assert!(p.parts.len() > 40);
    let hash = p.hash;
    let n = p.length;
    let input = w
        .retain_history_pages(p, &observation("fragmented"), 131072)
        .unwrap();
    let source = w.retained_source(&input.id).unwrap().unwrap();
    let reader = w.read_handle().unwrap();
    let mut good_polls = 0u64;
    reader
        .verify_source(&source, || {
            good_polls += 1;
            Ok(())
        })
        .unwrap();
    let mut bytes = Vec::new();
    let mut at = 0;
    while at < n {
        let amount = (n - at).min(65536);
        bytes.extend(
            reader
                .read_source_range_checked(&input.id, 1, hash, at, amount, || {
                    good_polls += 1;
                    Ok(())
                })
                .unwrap(),
        );
        at += amount;
    }
    assert_eq!(blake3::hash(&bytes).as_bytes(), hash.bytes());
    assert_eq!(
        reader
            .source_readers
            .explicit_verifications
            .load(Ordering::Relaxed),
        2
    );
    assert_eq!(
        reader
            .source_readers
            .explicit_verification_bytes
            .load(Ordering::Relaxed),
        base_length + patch_length
    );
    let bad = w.read_handle().unwrap();
    bad.source_readers
        .force_shared_slot
        .store(true, Ordering::Relaxed);
    let mut bad_polls = 0u64;
    bad.verify_source(&source, || {
        bad_polls += 1;
        Ok(())
    })
    .unwrap();
    let mut at = 0;
    let mut actual = Vec::new();
    while at < n {
        let take = (n - at).min(65536);
        actual.extend(
            bad.read_source_range_checked(&input.id, 1, hash, at, take, || {
                bad_polls += 1;
                Ok(())
            })
            .unwrap(),
        );
        at += take;
    }
    assert_eq!(actual, bytes);
    // Both paths retain two bookkeeping hashes. Only the actual library's
    // callback work distinguishes a shared expected-object slot from two.
    assert!(
        bad_polls > 4 * good_polls,
        "actual check polls: normal={good_polls}, shared-slot mutant={bad_polls}"
    );
    println!(
        "actual library check polls normal={good_polls}, shared-slot mutant={bad_polls}; explicit warming bytes are counted separately"
    );
    let basis = w.history_basis(&observation("same")).unwrap();
    let p = capture
        .capture_history_pages(&path, basis, &AtomicBool::new(false))
        .unwrap();
    assert_eq!(p.base_explicit_verification_bytes, base_length);
    assert_eq!(p.changed_bytes, patch_length);
    assert_eq!(p.hash, hash);
    println!(
        "alternating layout: explicit warming calls2, bytes={}, current capture base verification={}, differing page bytes={}",
        base_length + patch_length,
        p.base_explicit_verification_bytes,
        p.changed_bytes
    );
}

#[test]
fn earlier_bytes_observed_again_keep_identity_and_canonical_extent_layout() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("provider.sqlite");
    let c = Connection::open(&path).unwrap();
    c.execute_batch("PRAGMA journal_mode=WAL;PRAGMA wal_autocheckpoint=0;CREATE TABLE t(x);INSERT INTO t VALUES(1)").unwrap();
    let mut w = Writer::open(&temp.path().join("archive")).unwrap();
    let capture = w.blob_capture().unwrap();
    let p = capture
        .capture_history_pages(&path, None, &AtomicBool::new(false))
        .unwrap();
    let first_hash = p.hash;
    let first = w
        .retain_history_pages(p, &observation("first"), 131072)
        .unwrap();
    // Another path selecting equal physical bytes must reuse the checked layout,
    // even when this candidate arrived as an independently encoded whole image.
    let p = capture
        .capture_history_pages(&path, None, &AtomicBool::new(false))
        .unwrap();
    assert!(p.canonical_read_bytes > 0);
    let mut other = observation("another-path");
    other.path = "copy:independent-history";
    let duplicate = w.retain_history_pages(p, &other, 131072).unwrap();
    assert_eq!(duplicate, first);
    c.execute_batch("INSERT INTO t VALUES(2)").unwrap();
    let p = capture
        .capture_history_pages(
            &path,
            w.history_basis(&observation("second")).unwrap(),
            &AtomicBool::new(false),
        )
        .unwrap();
    w.retain_history_pages(p, &observation("second"), 131072)
        .unwrap();
    assert!(
        w.known_history_snapshot(first_hash, &observation("again"))
            .unwrap()
            .is_none()
    );
    let reader = w.read_handle().unwrap();
    let first_source = w.retained_source(&first.id).unwrap().unwrap();
    let original = reader
        .read_source_range(&first.id, 1, first_hash, 0, first_source.input.length)
        .unwrap();
    // Restore only this owned provider after closing its SQLite writer.
    drop(c);
    std::fs::write(&path, original).unwrap();
    let p = capture
        .capture_history_pages(
            &path,
            w.history_basis(&observation("again")).unwrap(),
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(p.hash, first_hash);
    assert!(p.canonical_read_bytes > 0);
    let again = w
        .retain_history_pages(p, &observation("again"), 131072)
        .unwrap();
    assert_eq!(again, first);
    let observed: String = w
        .archive
        .query_row(
            "SELECT id FROM observation ORDER BY rowid DESC LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(observed, "again");
}

#[test]
fn cancelled_first_object_warming_is_retried_and_cached_reads_keep_metadata_fences() {
    use std::sync::atomic::Ordering;
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("provider.sqlite");
    let c = Connection::open(&path).unwrap();
    c.execute_batch("CREATE TABLE t(value BLOB);INSERT INTO t VALUES(zeroblob(8388608))")
        .unwrap();
    let mut w = Writer::open(&temp.path().join("archive")).unwrap();
    let p = w
        .blob_capture()
        .unwrap()
        .capture_history_pages(&path, None, &AtomicBool::new(false))
        .unwrap();
    let hash = p.hash;
    let n = p.length;
    let input = w
        .retain_history_pages(p, &observation("cancelled-warming"), 131072)
        .unwrap();
    let reader = w.read_handle().unwrap();
    let mut polls = 0;
    let result = reader.read_source_range_checked(&input.id, 1, hash, 0, 65536, || {
        polls += 1;
        if polls >= 10 {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    });
    assert!(matches!(result, Err(Error::Cancelled)));
    assert_eq!(
        reader
            .source_readers
            .explicit_verifications
            .load(Ordering::Relaxed),
        0
    );
    let bytes = reader.read_source_range(&input.id, 1, hash, 0, n).unwrap();
    assert_eq!(blake3::hash(&bytes).as_bytes(), hash.bytes());
    assert_eq!(
        reader
            .source_readers
            .explicit_verifications
            .load(Ordering::Relaxed),
        1
    );
    w.archive
        .execute("UPDATE blob SET state='deleting'", [])
        .unwrap();
    assert!(matches!(
        reader.read_source_range(&input.id, 1, hash, 0, 1),
        Err(Error::UnknownBlob)
    ));
    w.archive
        .execute("UPDATE blob SET state='present'", [])
        .unwrap();
    assert_eq!(
        reader.read_source_range(&input.id, 1, hash, 0, 1).unwrap(),
        bytes[..1]
    );
    assert_eq!(
        reader
            .source_readers
            .explicit_verifications
            .load(Ordering::Relaxed),
        2
    );
}
