//! The archive's runtime directory, `$XDG_RUNTIME_DIR/polylogue/<key>/`:
//! the web token and port, and the terminal row numbers.

use std::fs;
use std::path::{Path, PathBuf};

/// The first 12 hex digits of BLAKE3 over the archive directory's resolved
/// path: it names the archive's runtime directory and its cookie, so two
/// archives never share a token.
pub fn archive_key(root: &Path) -> String {
    let resolved = polylogue_store::Archive::resolved_root(root);
    let hash = blake3::hash(resolved.as_os_str().as_encoded_bytes());
    hash.to_hex()[..12].to_owned()
}

/// `$XDG_RUNTIME_DIR/polylogue/<key>/`.
pub fn runtime_dir(key: &str) -> PathBuf {
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    base.join("polylogue").join(key)
}

#[cfg(test)]
mod tests;
