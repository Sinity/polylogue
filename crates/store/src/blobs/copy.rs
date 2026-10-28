//! Copy a plain archived blob with bounded buffers and verify its whole identity.
use crate::{Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::Connection;
use std::io::{Read, Write};

/// Decode a catalogued plain blob using the private-destination rule of
/// [`copy_verified`].
pub(crate) fn copy_plain(
    archive: &Connection,
    store: &impl super::PlainBlobs,
    hash: BlobHash,
    output: impl Write,
) -> Result<()> {
    let size: i64 =
        archive.query_row("SELECT size FROM blob WHERE hash=?1", [&hash.0[..]], |r| {
            r.get(0)
        })?;
    let size = u64::try_from(size).map_err(|_| StoreError::BlobCorrupt(hash))?;
    let mut reader = store.open_plain(&hashed_blobs::Hash(hash.0), size)?;
    copy_verified(&mut reader, output, hash, Some(size))?;
    Ok(())
}

/// Use the copied bytes only after this returns successfully. An error can
/// leave an unverified prefix in the destination.
fn copy_verified(
    mut reader: impl Read,
    mut output: impl Write,
    hash: BlobHash,
    expected: Option<u64>,
) -> Result<u64> {
    let mut buffer = [0; 64 * 1024];
    let mut digest = blake3::Hasher::new();
    let mut length = 0u64;
    loop {
        let n = match reader.read(&mut buffer) {
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if n == 0 {
            break;
        }
        length = length
            .checked_add(n as u64)
            .ok_or(StoreError::BlobCorrupt(hash))?;
        if expected.is_some_and(|n| length > n) {
            return Err(StoreError::BlobCorrupt(hash));
        }
        digest.update(&buffer[..n]);
        output.write_all(&buffer[..n])?;
    }
    if expected.is_some_and(|n| length != n) || digest.finalize().as_bytes() != &hash.0 {
        return Err(StoreError::BlobCorrupt(hash));
    }
    Ok(length)
}
