//! Copy a plain archived blob with bounded buffers and verify its whole identity.
use crate::{Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::{Connection, MAIN_DB};
use std::io::{Read, Write};

pub(crate) struct CopyStats {
    pub bytes: u64,
    #[cfg(test)]
    pub decoded_bytes: u64,
}

/// Decode a catalogued plain blob using the private-destination rule of
/// [`copy_verified`].
pub(crate) fn copy_plain(
    archive: &Connection,
    store: &hashed_blobs::Store,
    hash: BlobHash,
    output: impl Write,
) -> Result<CopyStats> {
    let (rowid, place, encoding, size): (i64, String, String, i64) = archive.query_row(
        "SELECT rowid,place,encoding,size FROM blob WHERE hash=?1",
        [&hash.0[..]],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
    )?;
    if encoding != "plain" {
        return Err(StoreError::Invalid(format!(
            "blob {hash} has {encoding} encoding, not plain"
        )));
    }
    let size = u64::try_from(size).map_err(|_| StoreError::BlobCorrupt(hash))?;
    match place.as_str() {
        "inline" => {
            let blob = archive.blob_open(MAIN_DB, "blob", "data", rowid, true)?;
            let bytes = copy_verified(
                zstd::stream::read::Decoder::new(blob)?,
                output,
                hash,
                Some(size),
            )?;
            Ok(CopyStats {
                bytes,
                #[cfg(test)]
                decoded_bytes: bytes,
            })
        }
        "file" => {
            let mut reader =
                store.read_plain(&hashed_blobs::Hash(hash.0), Some(size), || Ok(()))?;
            let bytes = copy_verified(&mut reader, output, hash, Some(size))?;
            Ok(CopyStats {
                bytes,
                #[cfg(test)]
                decoded_bytes: reader.stats().decoded_bytes,
            })
        }
        _ => Err(StoreError::Invalid(format!("unknown blob place {place}"))),
    }
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
