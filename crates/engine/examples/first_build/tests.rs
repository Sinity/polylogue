use super::*;

#[test]
fn missing_or_bad_proc_readings_stay_unknown_and_measured_zero_stays_zero() {
    let io = Some("read_bytes: 0\nwrite_bytes: 4096\nwchar: broken\n");
    assert_eq!(io_count(io, "read_bytes:"), Some(0));
    assert_eq!(io_count(io, "write_bytes:"), Some(4096));
    assert_eq!(io_count(io, "wchar:"), None);
    assert_eq!(io_count(io, "missing:"), None);
    assert_eq!(io_count(None, "read_bytes:"), None);
    assert_eq!(peak_memory_mb(Some("VmHWM: 2048 kB\n")), Some(2));
    assert_eq!(peak_memory_mb(Some("VmHWM: 2048 MB\n")), None);
    assert_eq!(peak_memory_mb(Some("VmHWM: broken kB\n")), None);
    assert_eq!(peak_memory_mb(None), None);
}

#[test]
fn io_deltas_exclude_earlier_work_without_guessing_missing_or_reset_counters() {
    assert_eq!(counter_delta(Some(8192), Some(12288)), Some(4096));
    assert_eq!(counter_delta(Some(8192), Some(8192)), Some(0));
    assert_eq!(counter_delta(Some(8192), Some(4096)), None);
    assert_eq!(counter_delta(None, Some(4096)), None);
    assert_eq!(counter_delta(Some(4096), None), None);
}

#[test]
fn process_cpu_counts_survive_comm_parentheses_and_missing_fields_stay_unknown() {
    assert_eq!(
        cpu_ticks(Some("1 (worker ) name) S 1 2 3 4 5 6 7 8 9 10 100 20")),
        Some(120)
    );
    assert_eq!(cpu_ticks(Some("1 (worker) S 1")), None);
    assert_eq!(cpu_ticks(None), None);
}

#[test]
fn completed_build_measurement_pays_its_wal_and_refuses_a_live_reader() {
    let directory = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_or_create(directory.path()).unwrap();
    let writer = polylogue_store::archive::open_write(
        &archive.index_db(),
        polylogue_store::archive::cache::SETUP,
    )
    .unwrap();
    writer
        .set_db_config(
            rusqlite::config::DbConfig::SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE,
            true,
        )
        .unwrap();
    writer
        .execute_batch("CREATE TABLE measured(value INTEGER); INSERT INTO measured VALUES(1)")
        .unwrap();
    let reader = rusqlite::Connection::open(archive.index_db()).unwrap();
    reader.execute_batch("BEGIN").unwrap();
    assert_eq!(
        reader
            .query_row("SELECT value FROM measured", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    writer.execute("UPDATE measured SET value=2", []).unwrap();
    assert!(checkpoint_first_build(&archive).is_err());
    assert_eq!(
        reader
            .query_row("SELECT value FROM measured", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        1
    );
    reader.execute_batch("ROLLBACK").unwrap();
    let databases = checkpoint_first_build(&archive).unwrap();
    let index = databases
        .iter()
        .find(|database| database.name == "index")
        .unwrap();
    assert!(index.wal_before > 0);
    assert!(databases.iter().all(|database| database.wal_after == 0));
    assert_eq!(
        reader
            .query_row("SELECT value FROM measured", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        2
    );
}
