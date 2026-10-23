//! Text matches belong to logical revision chains, not to isolated blob files.

use super::*;
use text_matches::TextMatches;

mod bytes;
mod framing;
mod plan;
use bytes::PlainBytes;
use plan::Plan;

const READ_CHUNK: usize = 64 * 1024;
const WRITE_CHUNK: usize = 1024 * 1024;
type Chain = (String, i64);
type Ranges = Vec<(u64, u64)>;

pub(super) struct StoredText {
    plan: Plan,
    fill: u8,
    bytes: PlainBytes,
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
    let plan = Plan::new()?;
    let fill = text_matches::replacement(needle)?;
    let mut bytes = PlainBytes::new()?;
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
        let frames = framing::read(archive, store, &input, chain, length)?;
        let frame_headers = framing::headers(&frames);
        let mut framing = frame_headers.iter().peekable();
        let mut scanner = TextMatches::new(needle)?;
        let mut header = Vec::with_capacity(16);
        let mut pos = 0;
        (|| {
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
                bytes.visit(&plan, archive, store, &e, |bytes, at| {
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
                    while framing.peek().is_some_and(|(_, end)| *end <= at) { framing.next(); }
                    let mut masked = None;
                    for (start, end) in framing.clone() {
                        if *start >= at + bytes.len() as u64 { break; }
                        let copy = masked.get_or_insert_with(|| bytes.to_vec());
                        let from = start.saturating_sub(at) as usize;
                        let upto = (*end - at).min(bytes.len() as u64) as usize;
                        copy[from..upto].fill(b' ');
                    }
                    scanner.feed(masked.as_deref().unwrap_or(bytes)).map_err(|e| unsupported(e.to_string()))?;
                    plan.collect(&input, chain, scanner.drain())
                })?;
                pos += e.length;
            }
            if pos != length {
                return Err(StoreError::Invalid(format!(
                    "revision ends at {pos}, before {length}"
                )));
            }
            plan.collect(
                &input,
                chain,
                scanner.complete().map_err(|e| unsupported(e.to_string()))?,
            )
        })()?;
        plan.normalize(&input, chain)?;
        let rewritten = framing::rewrite(
            archive,
            store,
            &mut bytes,
            &plan,
            &(input.clone(), chain),
            &frames,
            fill,
        )?;
        plan.patches(&input, chain, &rewritten)?;
    }
    Ok(StoredText { plan, fill, bytes })
}

impl StoredText {
    pub(super) fn blobs(&self, archive: &rusqlite::Connection) -> Result<Vec<BlobHash>> {
        self.plan.stage_affected(archive)?;
        let mut hashes = Vec::new();
        self.plan.visit_blobs(|hash| {
            hashes.push(hash);
            Ok(())
        })?;
        Ok(hashes)
    }

    pub(super) fn inputs(&self) -> Result<BTreeSet<String>> {
        self.plan.inputs()
    }

    pub(super) fn counts(&self) -> Result<BTreeMap<String, u64>> {
        self.plan.counts()
    }

    pub(super) fn overwrite(
        &mut self,
        t: &Transactions<'_, '_>,
        req: &ExciseRequest,
        refused: &mut RefusedJson,
    ) -> Result<()> {
        self.plan.stage_affected(t.archive)?;
        // A shared blob may contribute only a fragment of a match to one input.
        // Copy each referring extent, applying only that chain's matches. This
        // leaves unrelated input bytes exact and drops unused blob padding.
        self.plan.visit_blobs(|hash| {
            self.plan.freeze_extents(t.archive, hash)?;
            self.plan.visit_extents(|e| {
                let key = (e.input.clone(), e.chain);
                let mut batch = t.store.batch(hashed_blobs::Durability::PerFile);
                let mut buffer = Vec::with_capacity(WRITE_CHUNK);
                let mut pieces = Vec::new();
                self.bytes
                    .visit(&self.plan, t.archive, t.store, &e, |bytes, at| {
                        let mut bytes = bytes.to_vec();
                        self.plan.rewrite(&key, at, &mut bytes, self.fill)?;
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
                Ok(())
            })?;
            refused.blobs.push(hex(&hash.0));
            Ok(())
        })?;
        self.plan.visit_observations(|input, spans| {
            insert_excised_observations(t, input, req, Some(&spans))
        })?;
        Ok(())
    }
}
