//! Durable replacement of a whole file.
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
    file.persist(path).map_err(|error| error.error)?;
    fs::File::open(parent)?.sync_all()
}

#[cfg(test)]
mod tests;
