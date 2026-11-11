//! A daemon whose register failed a disk sync stops writing for the rest of
//! the boot (docs/operations.md, storage failure). The failure is kept in
//! the archive's runtime directory with the boot's identifier: the runtime
//! directory does not survive a reboot, and a later boot's identifier
//! differs, so after a reboot the daemon starts again.
use polylogue_store::storage_failure::StorageFailure;
use std::path::{Path, PathBuf};

/// Why the daemon stopped, or refuses to start, in this boot.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct StorageStopped(pub String);

fn marker(root: &Path) -> PathBuf {
    crate::runtime_dir::runtime_dir(&crate::runtime_dir::archive_key(root)).join("storage-failure")
}

fn boot() -> Option<String> {
    std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .ok()
        .map(|id| id.trim().to_owned())
        .filter(|id| !id.is_empty())
}

/// The failure recorded for `root` in this boot, if there is one.
pub(crate) fn this_boot(root: &Path) -> Option<StorageStopped> {
    let text = std::fs::read_to_string(marker(root)).ok()?;
    let (recorded_boot, message) = text.split_once('\n')?;
    (Some(recorded_boot) == boot().as_deref()).then(|| {
        StorageStopped(format!(
            "{message}; refusing to start until the machine has rebooted (docs/operations.md, storage failure)"
        ))
    })
}

/// Keeps `failure` for the rest of this boot. A machine without a boot
/// identifier keeps nothing, and only the stopped process refuses.
pub(crate) fn remember(root: &Path, failure: &StorageFailure) {
    let Some(boot) = boot() else {
        return;
    };
    let path = marker(root);
    let written = path
        .parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| std::fs::write(&path, format!("{boot}\n{failure}")));
    if let Err(error) = written {
        eprintln!(
            "polylogue: record the storage failure at {}: {error}",
            path.display()
        );
    }
}

#[cfg(test)]
mod tests;
