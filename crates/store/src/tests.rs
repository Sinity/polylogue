use super::*;

#[test]
fn sqlite_and_io_failures_are_classified_by_what_ends_them() {
    use polylogue_model::{Cause, Caused as _};
    for (code, cause) in [
        (rusqlite::ffi::SQLITE_BUSY, Cause::Transient),
        (rusqlite::ffi::SQLITE_LOCKED, Cause::Transient),
        (rusqlite::ffi::SQLITE_IOERR, Cause::Transient),
        (rusqlite::ffi::SQLITE_FULL, Cause::Environment),
        (rusqlite::ffi::SQLITE_READONLY, Cause::Environment),
        (rusqlite::ffi::SQLITE_PERM, Cause::Environment),
        (rusqlite::ffi::SQLITE_CORRUPT, Cause::StoredData),
        (rusqlite::ffi::SQLITE_INTERRUPT, Cause::Stopping),
        (rusqlite::ffi::SQLITE_CONSTRAINT, Cause::Bug),
    ] {
        let error = StoreError::Sqlite(rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(code),
            None,
        ));
        assert_eq!(error.cause(), cause, "{error}");
    }
    for (kind, cause) in [
        (std::io::ErrorKind::PermissionDenied, Cause::Environment),
        (std::io::ErrorKind::ReadOnlyFilesystem, Cause::Environment),
        (std::io::ErrorKind::InvalidInput, Cause::Bug),
        (std::io::ErrorKind::Interrupted, Cause::Transient),
        (std::io::ErrorKind::TimedOut, Cause::Transient),
    ] {
        assert_eq!(StoreError::Io(kind.into()).cause(), cause, "{kind:?}");
    }
    // A stop carried through an I/O error is still a stop.
    assert_eq!(
        StoreError::Io(std::io::Error::other(StoreError::Stopped)).cause(),
        Cause::Stopping
    );
    assert_eq!(
        StoreError::Invalid("refused content".into()).cause(),
        Cause::Request
    );
    assert_eq!(StoreError::internal("broken").cause(), Cause::Bug);
}

#[test]
fn a_group_member_keeps_the_group_failure_cause_and_code() {
    use polylogue_model::{Cause, Caused as _, Code};
    let busy = StoreError::Sqlite(rusqlite::Error::SqliteFailure(
        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_BUSY),
        None,
    ));
    let member = busy.for_group_member();
    assert_eq!(
        (member.cause(), member.code()),
        (Cause::Transient, Code::Busy)
    );
    assert!(!member.cause().recurs());
    let bug = StoreError::internal("broken").for_group_member();
    assert_eq!((bug.cause(), bug.code()), (Cause::Bug, Code::Internal));
    assert!(bug.cause().recurs());
    assert!(matches!(
        StoreError::Stopped.for_group_member(),
        StoreError::Stopped
    ));
}
