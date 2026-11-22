//! The mailbox flag: an empty file `mail/<harness>/<session>` in the
//! archive's runtime directory, present while a delivery waits for the
//! session's next context-accepting hook event. A hook checks the one path
//! with a `stat` and asks the daemon only when it exists, so pings and
//! notices cost nothing to sessions with none pending. A native ID is any
//! text its source wrote, so the file name is its percent-encoded form: an ID
//! holding `/` or `..` still names one file inside its harness directory.

use polylogue_model::SessionId;
use std::path::{Path, PathBuf};

/// The mailbox directory of the archive at `root`.
pub fn dir(paths: &crate::config::Paths, root: &Path) -> PathBuf {
    paths.runtime_dir(root).join("mail")
}

/// The flag file of `session`.
pub fn flag(dir: &Path, session: &SessionId) -> PathBuf {
    dir.join(session.origin().as_str())
        .join(file_name(session.native().as_str()))
}

/// A native ID as one file name: every byte but ASCII letters, digits, `-`
/// and `_` is written `%XX`.
fn file_name(native: &str) -> String {
    use std::fmt::Write as _;
    let mut name = String::with_capacity(native.len());
    for byte in native.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_') {
            name.push(byte as char);
        } else {
            let _ = write!(name, "%{byte:02X}");
        }
    }
    name
}

/// The native ID a flag's file name encodes; `None` for any other file.
fn native_of(name: &str) -> Option<String> {
    let mut bytes = Vec::with_capacity(name.len());
    let mut rest = name.as_bytes();
    while let Some((&byte, tail)) = rest.split_first() {
        if byte == b'%' {
            let (digits, tail) = tail.split_at_checked(2)?;
            if !digits.iter().all(u8::is_ascii_hexdigit) {
                return None;
            }
            bytes.push(u8::from_str_radix(std::str::from_utf8(digits).ok()?, 16).ok()?);
            rest = tail;
        } else {
            bytes.push(byte);
            rest = tail;
        }
    }
    String::from_utf8(bytes).ok()
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
            if let Some(native) = session.file_name().to_str().and_then(native_of)
                && let Ok(id) = SessionId::parse(&format!(
                    "{}:{native}",
                    origin.file_name().to_string_lossy()
                ))
            {
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

#[cfg(test)]
mod tests;
