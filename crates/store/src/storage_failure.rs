//! A failed sync of the register. After a sync fails the kernel may have
//! marked the unwritten pages clean, so no later sync in this boot, in this
//! process or another, can show that they reached the disk: the page cache
//! that still holds them survives a process restart. Writing therefore stops
//! for the rest of the boot. The process records the failure here; the
//! writer refuses to start once one is recorded, and the daemon stops with
//! its storage exit code and remembers the boot (docs/operations.md,
//! storage failure).
use std::path::PathBuf;
use std::sync::{Condvar, Mutex};

/// Which file failed to sync, and how.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageFailure {
    pub path: PathBuf,
    pub message: String,
    /// The operating system's error number, when the failure came from it.
    pub errno: Option<i32>,
}

impl std::fmt::Display for StorageFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "storage I/O error syncing {}: {}",
            self.path.display(),
            self.message
        )?;
        if let Some(errno) = self.errno {
            write!(f, " (errno {errno})")?;
        }
        f.write_str("; writing stopped for this boot")
    }
}

impl StorageFailure {
    pub(crate) fn of(path: PathBuf, error: &crate::StoreError) -> Self {
        let errno = match error {
            crate::StoreError::Io(error) => error.raw_os_error(),
            _ => None,
        };
        Self {
            path,
            message: error.to_string(),
            errno,
        }
    }
}

static FAILED: Mutex<Option<StorageFailure>> = Mutex::new(None);
static RECORDED: Condvar = Condvar::new();

/// Records the first failure of this process; later ones keep the first.
pub(crate) fn record(failure: StorageFailure) {
    let mut failed = FAILED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if failed.is_none() {
        eprintln!("polylogue: {failure}");
        *failed = Some(failure);
        RECORDED.notify_all();
    }
}

/// The failure this process recorded, if any.
pub fn recorded() -> Option<StorageFailure> {
    FAILED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

/// Waits until this process records a failure.
pub fn wait() -> StorageFailure {
    let mut failed = FAILED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    loop {
        if let Some(failure) = failed.as_ref() {
            return failure.clone();
        }
        failed = RECORDED
            .wait(failed)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
    }
}
