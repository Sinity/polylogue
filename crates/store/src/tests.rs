use super::*;

#[test]
fn write_retry_distinguishes_transient_storage_from_refused_or_invalid_state() {
    for (code, retry) in [
        (rusqlite::ffi::SQLITE_BUSY, true),
        (rusqlite::ffi::SQLITE_LOCKED, true),
        (rusqlite::ffi::SQLITE_IOERR, true),
        (rusqlite::ffi::SQLITE_FULL, true),
        (rusqlite::ffi::SQLITE_READONLY, false),
        (rusqlite::ffi::SQLITE_PERM, false),
        (rusqlite::ffi::SQLITE_CORRUPT, false),
        (rusqlite::ffi::SQLITE_SCHEMA, false),
        (rusqlite::ffi::SQLITE_CONSTRAINT, false),
    ] {
        let error = StoreError::Sqlite(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(code),
            None,
        ));
        assert_eq!(error.retryable_write(), retry, "{error}");
    }
    for kind in [
        std::io::ErrorKind::PermissionDenied,
        std::io::ErrorKind::ReadOnlyFilesystem,
        std::io::ErrorKind::InvalidInput,
        std::io::ErrorKind::Unsupported,
    ] {
        assert!(!StoreError::Io(kind.into()).retryable_write());
    }
    for kind in [
        std::io::ErrorKind::Interrupted,
        std::io::ErrorKind::WouldBlock,
        std::io::ErrorKind::TimedOut,
        std::io::ErrorKind::Other,
    ] {
        assert!(StoreError::Io(kind.into()).retryable_write());
    }
    assert!(!StoreError::Invalid("refused content".into()).retryable_write());
}
