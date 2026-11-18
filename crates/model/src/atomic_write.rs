//! Crash-safe replacement of a whole file.
use std::{
    fs,
    io::{self, Write},
    path::Path,
};

/// Replaces `path` after syncing its bytes, then syncs its directory. The
/// temporary file is private unless the caller supplies permissions.
pub fn write_atomic(
    path: &Path,
    bytes: &[u8],
    permissions: Option<fs::Permissions>,
) -> io::Result<()> {
    write(path, bytes, permissions, false)
}

/// Publishes a complete private file without replacing an existing destination.
pub fn write_atomic_if_absent(path: &Path, bytes: &[u8]) -> io::Result<()> {
    write(path, bytes, None, true)
}

fn write(
    path: &Path,
    bytes: &[u8],
    permissions: Option<fs::Permissions>,
    new: bool,
) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    if let Some(permissions) = permissions {
        file.as_file().set_permissions(permissions)?;
    }
    file.write_all(bytes)?;
    file.as_file().sync_all()?;
    if new {
        file.persist_noclobber(path).map_err(|error| error.error)?;
    } else {
        file.persist(path).map_err(|error| error.error)?;
    }
    fs::File::open(parent)?.sync_all()
}

#[cfg(test)]
mod tests;
