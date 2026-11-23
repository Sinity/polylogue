//! The mailbox flag: an empty file `mailbox/<origin>/<session>` in the
//! archive's runtime directory, present while a delivery waits for the
//! session's next context-accepting hook event. A hook checks the one path
//! with a `stat` and asks the daemon only when it exists, so messages and
//! notices cost nothing to sessions with none pending. A native ID is any
//! text its source wrote, so the file name is its percent-encoded form: an ID
//! holding `/` or `..` still names one file inside its origin directory.

use polylogue_model::SessionId;
use std::path::{Path, PathBuf};

/// The mailbox directory of the archive at `root`.
pub fn dir(paths: &crate::config::Paths, root: &Path) -> PathBuf {
    paths.runtime_dir(root).join("mailbox")
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

/// The native ID a flag's file name encodes; `None` for any file that
/// `file_name` would not have written.
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
    let native = String::from_utf8(bytes).ok()?;
    (file_name(&native) == name).then_some(native)
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

/// The sessions whose flag is up. A mailbox that was never created has none;
/// any other read failure is an error.
pub fn flagged(dir: &Path) -> std::io::Result<Vec<SessionId>> {
    let origins = match std::fs::read_dir(dir) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        other => other?,
    };
    let mut out = Vec::new();
    for origin in origins {
        let origin = origin?;
        let sessions = match std::fs::read_dir(origin.path()) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            other => other?,
        };
        for session in sessions {
            let session = session?;
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
    Ok(out)
}

/// Whether a delivery may be pending: one `stat`. A flag that cannot be
/// read counts as up, so the daemon is asked.
pub fn raised(dir: &Path, session: &SessionId) -> bool {
    !matches!(
        std::fs::symlink_metadata(flag(dir, session)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound
    )
}

#[cfg(test)]
mod tests;
