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
