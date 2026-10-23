//! Snapshot headers are structural bytes. Payloads share the operation's verified
//! plaintext owner; each header and digest is checked before replacement begins.
use super::*;
use polylogue_model::snapshot_log::{HEADER_LEN, Header, MAGIC};

pub(super) fn scan(
    archive: &rusqlite::Connection,
    store: &hashed_blobs::Store,
    bytes: &mut PlainBytes,
    plan: &Plan,
    key: &Chain,
    length: u64,
    needle: &str,
) -> Result<()> {
    let format: String =
        archive.query_row("SELECT format FROM input WHERE id=?1", [&key.0], |r| {
            r.get(0)
        })?;
    let unsupported = |detail: String| StoreError::ExcisionFormat {
        input: key.0.clone(),
        format: format.clone(),
        detail,
    };
    let mut scanner = TextMatches::new(needle)?;
    let mut reader = bytes.reader(plan, archive, store, key, 0..length)?;
    let mut prefix = [0; MAGIC.len()];
    let n = length.min(prefix.len() as u64) as usize;
    reader.read_exact(&mut prefix[..n])?;
    let prefix = &prefix[..n];
    if prefix.starts_with(b"SQLite format 3\0") {
        return Err(unsupported("SQLite page images need a structural text-value rewrite; logical JSON row exports are supported".into()));
    }
    if prefix.starts_with(b"PK\x03\x04")
        || prefix.starts_with(b"PK\x05\x06")
        || prefix.starts_with(b"PK\x07\x08")
    {
        return Err(unsupported("ZIP containers need member decoding and rewriting; imported plain member inputs are supported".into()));
    }
    if prefix.starts_with(b"%PDF-") {
        return Err(unsupported(
            "PDF streams need text decoding and rewriting".into(),
        ));
    }
    if prefix.starts_with(&[0x1f, 0x8b]) {
        return Err(unsupported(
            "gzip streams need decoding and rewriting".into(),
        ));
    }
    let mut buffer = [0; READ_CHUNK];
    {
        let mut feed = |chunk: &[u8]| {
            scanner
                .feed(chunk)
                .map_err(|e| unsupported(e.to_string()))?;
            plan.collect(&key.0, key.1, scanner.drain())
        };
        if prefix == MAGIC {
            plan.framed(key)?;
            feed(&[b' '; MAGIC.len()])?;
            let mut ordinal = 0;
            while reader.position() < length {
                let at = reader.position();
                let mut encoded = [0; HEADER_LEN];
                reader.read_exact(&mut encoded)?;
                let header = Header::decode(&encoded).map_err(StoreError::Invalid)?;
                let end = reader
                    .position()
                    .checked_add(header.length)
                    .filter(|end| *end <= length)
                    .ok_or_else(|| StoreError::Invalid("incomplete snapshot payload".into()))?;
                plan.frame(key, ordinal, at, &header)?;
                feed(&[b' '; HEADER_LEN])?;
                let mut before = blake3::Hasher::new();
                while reader.position() < end {
                    let take = buffer.len().min((end - reader.position()) as usize);
                    let n = reader.read(&mut buffer[..take])?;
                    if n == 0 {
                        return Err(StoreError::Invalid("incomplete snapshot payload".into()));
                    }
                    before.update(&buffer[..n]);
                    feed(&buffer[..n])?;
                }
                if before.finalize().to_hex().as_str() != header.digest {
                    return Err(StoreError::Invalid(
                        "snapshot payload digest mismatch".into(),
                    ));
                }
                ordinal += 1;
            }
        } else {
            if !prefix.is_empty() && MAGIC.starts_with(prefix) {
                return Err(StoreError::Invalid("incomplete snapshot log magic".into()));
            }
            feed(prefix)?;
            loop {
                let n = reader.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                feed(&buffer[..n])?;
            }
        }
    }
    plan.collect(
        &key.0,
        key.1,
        scanner.complete().map_err(|e| unsupported(e.to_string()))?,
    )?;
    plan.normalize(&key.0, key.1)
}

pub(super) fn rewrite(
    archive: &rusqlite::Connection,
    store: &hashed_blobs::Store,
    bytes: &mut PlainBytes,
    plan: &Plan,
    key: &Chain,
    fill: u8,
) -> Result<()> {
    plan.visit_frames(key, |ordinal, at, mut header| {
        let start = at + HEADER_LEN as u64;
        let end = start + header.length;
        let mut reader = bytes.reader(plan, archive, store, key, start..end)?;
        let mut edits = plan.edits(key, start)?;
        let mut after = blake3::Hasher::new();
        let mut buffer = [0; READ_CHUNK];
        loop {
            let at = reader.position();
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            edits.blank(at, &mut buffer[..n], fill)?;
            after.update(&buffer[..n]);
        }
        let digest = after.finalize().to_hex().to_string();
        if digest != header.digest {
            header.digest = digest;
            plan.patch(&key.0, key.1, ordinal, at, &header.encode())?;
        }
        Ok(())
    })
}
