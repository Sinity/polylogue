//! The archive key shared by runtime files and reader cookies.

use std::path::Path;

/// The first 12 hex digits of BLAKE3 over the archive directory's resolved
/// path: it names the archive's runtime directory and its cookie, so two
/// archives never share a token.
pub fn archive_key(root: &Path) -> String {
    let resolved = polylogue_store::Archive::resolved_root(root);
    let hash = blake3::hash(resolved.as_os_str().as_encoded_bytes());
    hash.to_hex()[..12].to_owned()
}

#[cfg(test)]
mod tests;
