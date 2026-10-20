//! Structural frame bytes are never text matches. Redaction changes the active
//! payload digest while keeping observed identity and all logical offsets.
use super::*;
use polylogue_model::snapshot_log::{HEADER_LEN, Header, MAGIC};

pub(super) struct Frame {
    at: u64,
    header: Header,
}

pub(super) fn read(
    archive: &rusqlite::Connection,
    store: &hashed_blobs::Store,
    input: &str,
    chain: i64,
    length: u64,
) -> Result<Vec<Frame>> {
    let bytes = crate::blobs::StoredChain::load(archive, "", input, chain, length)?;
    let prefix = bytes.read_range(store, 0, length.min(MAGIC.len() as u64))?;
    if prefix != MAGIC {
        if !prefix.is_empty() && MAGIC.starts_with(&prefix) {
            return Err(StoreError::Invalid("incomplete snapshot log magic".into()));
        }
        return Ok(Vec::new());
    }
    let mut frames = Vec::new();
    let mut at = MAGIC.len() as u64;
    while at < length {
        let body = at + HEADER_LEN as u64;
        if body > length {
            return Err(StoreError::Invalid("incomplete snapshot header".into()));
        }
        let header =
            Header::decode(&bytes.read_range(store, at, body)?).map_err(StoreError::Invalid)?;
        let end = body
            .checked_add(header.length)
            .filter(|end| *end <= length)
            .ok_or_else(|| StoreError::Invalid("incomplete snapshot payload".into()))?;
        frames.push(Frame { at, header });
        at = end;
    }
    Ok(frames)
}

pub(super) fn headers(frames: &[Frame]) -> Ranges {
    let mut ranges = Vec::new();
    if !frames.is_empty() {
        ranges.push((0, MAGIC.len() as u64));
    }
    ranges.extend(
        frames
            .iter()
            .map(|frame| (frame.at, frame.at + HEADER_LEN as u64)),
    );
    ranges
}

pub(super) fn rewrite(
    archive: &rusqlite::Connection,
    store: &hashed_blobs::Store,
    bytes: &mut PlainBytes,
    input: &str,
    chain: i64,
    frames: &[Frame],
    ranges: &Ranges,
    fill: u8,
) -> Result<Vec<(u64, Vec<u8>)>> {
    let mut patches = Vec::new();
    for frame in frames {
        let start = frame.at + HEADER_LEN as u64;
        let end = start + frame.header.length;
        let mut before = blake3::Hasher::new();
        let mut after = blake3::Hasher::new();
        let mut extents = archive.prepare("SELECT input, chain, offset, length, blob, blob_offset FROM extent
            WHERE input = ?1 AND chain = ?2 AND offset < ?4 AND offset + length > ?3 ORDER BY offset")?;
        let mut upto = start;
        for row in extents.query_map(params![input, chain, start as i64, end as i64], extent)? {
            let mut e = row?;
            let from = e.offset.max(start);
            let to = (e.offset + e.length).min(end);
            if from != upto {
                return Err(StoreError::Invalid("snapshot payload extent gap".into()));
            }
            e.blob_offset += from - e.offset;
            e.offset = from;
            e.length = to - from;
            bytes.visit(archive, store, &e, |chunk, at| {
                before.update(chunk);
                let mut chunk = chunk.to_vec();
                for &(from, to) in ranges {
                    let from = from.max(at);
                    let to = to.min(at + chunk.len() as u64);
                    if from < to {
                        for byte in &mut chunk[(from - at) as usize..(to - at) as usize] {
                            if *byte != b'\n' {
                                *byte = fill;
                            }
                        }
                    }
                }
                after.update(&chunk);
                Ok(())
            })?;
            upto = to;
        }
        if upto != end || before.finalize().to_hex().as_str() != frame.header.digest {
            return Err(StoreError::Invalid(
                "snapshot payload digest mismatch".into(),
            ));
        }
        let mut header = frame.header.clone();
        header.digest = after.finalize().to_hex().to_string();
        if header != frame.header {
            patches.push((frame.at, header.encode()));
        }
    }
    Ok(patches)
}
