//! Text matches belong to logical revision chains, not to isolated blob files.

use super::*;
use text_matches::TextMatches;

const READ_CHUNK: usize = 64 * 1024;
const WRITE_CHUNK: usize = 1024 * 1024;
type Chain = (String, i64);
type Ranges = Vec<(u64, u64)>;

pub(super) struct StoredText {
    ranges: BTreeMap<Chain, Ranges>,
    fill: u8,
}

#[derive(Clone)]
struct Extent {
    input: String,
    chain: i64,
    offset: u64,
    length: u64,
    blob: BlobHash,
    blob_offset: u64,
}

fn extent(r: &rusqlite::Row<'_>) -> rusqlite::Result<Extent> {
    let hash: Vec<u8> = r.get(4)?;
    let hash = hash.try_into().map_err(|_| rusqlite::Error::InvalidQuery)?;
    Ok(Extent {
        input: r.get(0)?,
        chain: r.get(1)?,
        offset: r.get::<_, i64>(2)? as u64,
        length: r.get::<_, i64>(3)? as u64,
        blob: BlobHash(hash),
        blob_offset: r.get::<_, i64>(5)? as u64,
    })
}

fn visit(
    archive: &rusqlite::Connection,
    store: &hashed_blobs::Store,
    e: &Extent,
    mut f: impl FnMut(&[u8], u64) -> Result<()>,
) -> Result<()> {
    let (place, encoding): (String, String) = archive.query_row(
        "SELECT place, encoding FROM blob WHERE hash = ?1",
        [&e.blob.0[..]],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    if encoding != "plain" {
        let format =
            archive.query_row("SELECT format FROM input WHERE id = ?1", [&e.input], |r| {
                r.get(0)
            })?;
        return Err(StoreError::ExcisionFormat {
            input: e.input.clone(),
            format,
            detail: format!("blob encoding {encoding} needs its own text-excision decoder"),
        });
    }
    let inline = if place == "inline" {
        Some(crate::blobs::read(archive, store, &e.blob)?)
    } else {
        None
    };
    let mut at = 0;
    while at < e.length {
        let take = (e.length - at).min(READ_CHUNK as u64);
        match &inline {
            Some(bytes) => {
                let start = (e.blob_offset + at) as usize;
                let bytes = bytes
                    .get(start..start + take as usize)
                    .ok_or_else(|| StoreError::Invalid("extent exceeds its stored blob".into()))?;
                f(bytes, e.offset + at)?;
            }
            None => {
                let bytes =
                    store.read_range(&hashed_blobs::Hash(e.blob.0), e.blob_offset + at, take)?;
                f(&bytes, e.offset + at)?;
            }
        }
        at += take;
    }
    Ok(())
}

/// Preflight every retained chain before any replacement is installed. Appends
/// share a chain: its longest revision includes every earlier prefix.
pub(super) fn find(t: &Transactions<'_, '_>, needle: &str) -> Result<StoredText> {
    find_in(t.archive, t.store, needle)
}

pub(super) fn find_in(
    archive: &rusqlite::Connection,
    store: &hashed_blobs::Store,
    needle: &str,
) -> Result<StoredText> {
    let mut ranges = BTreeMap::new();
    let mut chains = archive.prepare(
        "SELECT r.input, r.chain, max(r.length), i.format FROM revision r
         JOIN input i ON i.id = r.input GROUP BY r.input, r.chain ORDER BY r.input, r.chain",
    )?;
    let rows = chains.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, i64>(1)?,
            r.get::<_, i64>(2)? as u64,
            r.get::<_, String>(3)?,
        ))
    })?;
    for row in rows {
        let (input, chain, length, format) = row?;
        if format == "export-file" {
            return Err(StoreError::ExcisionFormat { input, format,
                detail: "uninterpreted export companions need a declared text/media decoder before their contents can be excised".into() });
        }
        let unsupported = |detail: String| StoreError::ExcisionFormat {
            input: input.clone(),
            format: format.clone(),
            detail,
        };
        let mut scanner = TextMatches::new(needle)?;
        let mut header = Vec::with_capacity(16);
        let mut pos = 0;
        let scan = (|| {
            let mut extents = archive.prepare(
                "SELECT input, chain, offset, length, blob, blob_offset FROM extent
                 WHERE input = ?1 AND chain = ?2 AND offset < ?3 ORDER BY offset",
            )?;
            for row in extents.query_map(params![input, chain, length as i64], extent)? {
                let mut e = row?;
                if e.offset != pos {
                    return Err(StoreError::Invalid(format!(
                        "revision bytes have a gap or overlap at {pos}"
                    )));
                }
                e.length = e.length.min(length - pos);
                visit(archive, store, &e, |bytes, _at| {
                    if header.len() < 16 {
                        header.extend_from_slice(&bytes[..bytes.len().min(16 - header.len())]);
                        if header.starts_with(b"SQLite format 3\0") {
                            return Err(unsupported("SQLite page images need a structural text-value rewrite; logical JSON row exports are supported".into()));
                        }
                        if header.starts_with(b"PK\x03\x04")
                            || header.starts_with(b"PK\x05\x06")
                            || header.starts_with(b"PK\x07\x08")
                        {
                            return Err(unsupported("ZIP containers need member decoding and rewriting; imported plain member inputs are supported".into()));
                        }
                        if header.starts_with(b"%PDF-") {
                            return Err(unsupported(
                                "PDF streams need text decoding and rewriting".into(),
                            ));
                        }
                        if header.starts_with(&[0x1f, 0x8b]) {
                            return Err(unsupported(
                                "gzip streams need decoding and rewriting".into(),
                            ));
                        }
                    }
                    scanner.feed(bytes).map_err(|e| unsupported(e.to_string()))
                })?;
                pos += e.length;
            }
            if pos != length {
                return Err(StoreError::Invalid(format!(
                    "revision ends at {pos}, before {length}"
                )));
            }
            scanner.finish().map_err(|e| unsupported(e.to_string()))
        })()?;
        if !scan.is_empty() {
            ranges.insert((input, chain), scan);
        }
    }
    Ok(StoredText {
        ranges,
        fill: text_matches::replacement(needle)?,
    })
}

impl StoredText {
    pub(super) fn blobs(&self, archive: &rusqlite::Connection) -> Result<Vec<BlobHash>> {
        let mut hashes = BTreeSet::new();
        let mut stmt = archive.prepare("SELECT DISTINCT blob FROM extent WHERE input = ?1 AND chain = ?2 AND offset < ?4 AND offset + length > ?3")?;
        for ((input, chain), ranges) in &self.ranges {
            for (start, end) in ranges {
                for row in stmt
                    .query_map(params![input, chain, *start as i64, *end as i64], |r| {
                        r.get::<_, Vec<u8>>(0)
                    })?
                {
                    let hash = row?
                        .try_into()
                        .map_err(|_| StoreError::Invalid("blob hash is not 32 bytes".into()))?;
                    hashes.insert(BlobHash(hash));
                }
            }
        }
        Ok(hashes.into_iter().collect())
    }

    pub(super) fn inputs(&self) -> BTreeSet<String> {
        self.ranges.keys().map(|(input, _)| input.clone()).collect()
    }

    pub(super) fn counts(&self) -> BTreeMap<String, u64> {
        let mut out = BTreeMap::new();
        for ((input, _), ranges) in &self.ranges {
            *out.entry(input.clone()).or_default() += ranges.len() as u64;
        }
        out
    }

    pub(super) fn overwrite(
        &self,
        t: &Transactions<'_, '_>,
        req: &ExciseRequest,
        refused: &mut RefusedJson,
    ) -> Result<()> {
        let mut blobs = BTreeSet::new();
        let mut observations: BTreeMap<String, Vec<serde_json::Value>> = BTreeMap::new();
        let mut affected = t.archive.prepare(
            "SELECT blob, offset, length, blob_offset FROM extent WHERE input = ?1 AND chain = ?2
             AND offset < ?4 AND offset + length > ?3",
        )?;
        for ((input, chain), ranges) in &self.ranges {
            for &(start, end) in ranges {
                for row in
                    affected.query_map(params![input, chain, start as i64, end as i64], |r| {
                        Ok((
                            r.get::<_, Vec<u8>>(0)?,
                            r.get::<_, i64>(1)? as u64,
                            r.get::<_, i64>(2)? as u64,
                            r.get::<_, i64>(3)? as u64,
                        ))
                    })?
                {
                    let (hash, offset, length, blob_offset) = row?;
                    let from = start.max(offset) - offset + blob_offset;
                    let to = end.min(offset + length) - offset + blob_offset;
                    observations
                        .entry(input.clone())
                        .or_default()
                        .push(serde_json::json!([hex(&hash), from, to - from]));
                    blobs.insert(hash);
                }
            }
        }
        drop(affected);
        // A shared blob may contribute only a fragment of a match to one input.
        // Copy each referring extent, applying only that chain's matches. This
        // leaves unrelated input bytes exact and drops unused blob padding.
        for hash in blobs {
            let extents: Vec<Extent> = t.archive.prepare(
                "SELECT input, chain, offset, length, blob, blob_offset FROM extent WHERE blob = ?1")?
                .query_map([&hash], extent)?.collect::<rusqlite::Result<_>>()?;
            for e in extents {
                let ranges = self.ranges.get(&(e.input.clone(), e.chain));
                let mut batch = t.store.batch(hashed_blobs::Durability::PerFile);
                let mut buffer = Vec::with_capacity(WRITE_CHUNK);
                let mut pieces = Vec::new();
                visit(t.archive, t.store, &e, |bytes, at| {
                    let mut bytes = bytes.to_vec();
                    if let Some(ranges) = ranges {
                        for &(start, end) in ranges {
                            let start = start.max(at);
                            let end = end.min(at + bytes.len() as u64);
                            if start < end {
                                for b in &mut bytes[(start - at) as usize..(end - at) as usize] {
                                    if *b != b'\n' {
                                        *b = self.fill;
                                    }
                                }
                            }
                        }
                    }
                    buffer.extend_from_slice(&bytes);
                    if buffer.len() == WRITE_CHUNK {
                        pieces.push(Piece::prepare_in(&mut batch, &buffer)?);
                        buffer.clear();
                    }
                    Ok(())
                })?;
                if !buffer.is_empty() {
                    pieces.push(Piece::prepare_in(&mut batch, &buffer)?);
                }
                batch.commit()?;
                t.archive.execute(
                    "DELETE FROM extent WHERE input = ?1 AND chain = ?2 AND offset = ?3",
                    params![e.input, e.chain, e.offset as i64],
                )?;
                let mut offset = e.offset;
                for mut piece in pieces {
                    piece.settle(t.store)?;
                    record_piece(t, &piece)?;
                    t.archive.execute(
                        "INSERT INTO extent(input, chain, offset, length, blob, blob_offset)
                        VALUES (?1, ?2, ?3, ?4, ?5, 0)",
                        params![
                            e.input,
                            e.chain,
                            offset as i64,
                            piece.size as i64,
                            &piece.hash.0[..]
                        ],
                    )?;
                    offset += piece.size;
                }
            }
            refused.blobs.push(hex(&hash));
        }
        for (input, spans) in observations {
            insert_excised_observations(t, &input, req, Some(&serde_json::Value::Array(spans)))?;
        }
        Ok(())
    }
}
