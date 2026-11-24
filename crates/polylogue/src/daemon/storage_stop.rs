//! A daemon whose archive database (`archive.db`) failed a disk sync stops writing for the rest of
//! the boot (docs/operations.md, storage failure). The failure is kept in
//! the archive's runtime directory with the boot's identifier. A later
//! boot's identifier differs, so after a reboot the daemon starts again.
use polylogue_store::storage_failure::StorageFailure;
use std::path::{Path, PathBuf};

/// Why the daemon stopped, or refuses to start, in this boot.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct StorageStopped(pub String);

fn marker(paths: &crate::config::Paths, root: &Path) -> PathBuf {
    paths.runtime_dir(root).join("storage-failure")
}

/// The failure recorded for `root` in this boot, if there is one.
pub(crate) fn this_boot(paths: &crate::config::Paths, root: &Path) -> Option<StorageStopped> {
    let text = match std::fs::read_to_string(marker(paths, root)) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        // An unreadable marker may hide a failure of this boot: refuse.
        Err(error) => {
            return Some(StorageStopped(format!(
                "the storage failure record could not be read ({error}); refusing to start (docs/operations.md, storage failure)"
            )));
        }
    };
    let (recorded_boot, message) = text.split_once('\n')?;
    (Some(recorded_boot) == crate::live_events::boot_id().as_deref()).then(|| {
        StorageStopped(format!(
            "{message}; refusing to start until the machine has rebooted (docs/operations.md, storage failure)"
        ))
    })
}

/// Keeps `failure` for the rest of this boot. A machine without a boot
/// identifier keeps nothing, and only the stopped process refuses.
pub(crate) fn remember(paths: &crate::config::Paths, root: &Path, failure: &StorageFailure) {
    let Some(boot) = crate::live_events::boot_id() else {
        return;
    };
    let path = marker(paths, root);
    let written = path
        .parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| {
            // Whole or absent: a torn marker would not match this boot.
            polylogue_model::atomic_write::write_atomic(
                &path,
                format!("{boot}\n{failure}").as_bytes(),
                None,
            )
        });
    if let Err(error) = written {
        polylogue_store::log!(
            error,
            "record the storage failure at {}: {error}",
            path.display()
        );
    }
}

#[cfg(test)]
mod tests;
