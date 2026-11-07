//! The mailbox flag: an empty file `mail/<harness>/<session>` in the
//! archive's runtime directory, present while a delivery waits for the
//! session's next context-accepting hook event. A hook checks the one path
//! with a `stat` and asks the daemon only when it exists, so pings and
//! notices cost nothing to sessions with none pending.

use polylogue_model::SessionId;
use std::path::{Path, PathBuf};

#[cfg(test)]
mod tests;

/// The mailbox directory of the archive at `root`.
pub fn dir(root: &Path) -> PathBuf {
    crate::runtime_dir::runtime_dir(&crate::runtime_dir::archive_key(root)).join("mail")
}

/// The flag file of `session`.
pub fn flag(dir: &Path, session: &SessionId) -> PathBuf {
    dir.join(session.origin().as_str())
        .join(session.native().to_string())
}

/// Marks a delivery pending for `session`.
pub fn raise(dir: &Path, session: &SessionId) -> std::io::Result<()> {
    let path = flag(dir, session);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(path)
        .map(drop)
}

/// Clears the flag once nothing is pending.
pub fn lower(dir: &Path, session: &SessionId) -> std::io::Result<()> {
    match std::fs::remove_file(flag(dir, session)) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        other => other,
    }
}

/// The sessions whose flag is up.
pub fn flagged(dir: &Path) -> Vec<SessionId> {
    let Ok(origins) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for origin in origins.flatten() {
        let Ok(sessions) = std::fs::read_dir(origin.path()) else {
            continue;
        };
        for session in sessions.flatten() {
            if let Ok(id) = SessionId::parse(&format!(
                "{}:{}",
                origin.file_name().to_string_lossy(),
                session.file_name().to_string_lossy()
            )) {
                out.push(id);
            }
        }
    }
    out
}

/// Whether a delivery may be pending: one `stat`.
pub fn raised(dir: &Path, session: &SessionId) -> bool {
    std::fs::symlink_metadata(flag(dir, session)).is_ok()
}
