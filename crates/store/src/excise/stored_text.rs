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
                detail: "input was captured as opaque; text/media excision for that byte format remains unsupported after later interpretation".into() });
        }
        let key = (input, chain);
        framing::scan(archive, store, &mut bytes, &plan, &key, length, needle)?;
        framing::rewrite(archive, store, &mut bytes, &plan, &key, fill)?;
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
        // A blob another blob is a delta of cannot go; refuse before writing.
        self.plan
            .visit_blobs(|hash| crate::blobs::refuse_delta_base(t.archive, &hash))?;
        // A shared blob may contribute only a fragment of a match to one input.
        // Copy each referring extent, applying only that chain's matches. This
        // leaves unrelated input bytes exact and drops unused blob padding.
        self.plan.visit_blobs(|hash| {
            self.plan.freeze_extents(t.archive, hash)?;
            self.plan.visit_extents(|e| {
                let key = (e.input.clone(), e.chain);
                let mut edits = self.plan.edits(&key, e.offset)?;
                let mut buffer = Vec::with_capacity(WRITE_CHUNK);
                let mut offset = e.offset;
                t.archive.execute(
                    "DELETE FROM extent WHERE input = ?1 AND chain = ?2 AND offset = ?3",
                    params![e.input, e.chain, e.offset as i64],
                )?;
                self.bytes
                    .visit(&self.plan, t.archive, t.store, &e, |chunk, at| {
                        let mut chunk = chunk.to_vec();
                        edits.rewrite(at, &mut chunk, self.fill)?;
                        let mut remaining = chunk.as_slice();
                        while !remaining.is_empty() {
                            let take = remaining.len().min(WRITE_CHUNK - buffer.len());
                            buffer.extend_from_slice(&remaining[..take]);
                            remaining = &remaining[take..];
                            if buffer.len() == WRITE_CHUNK {
                                install_piece(t, req, &e, &mut offset, &buffer)?;
                                buffer.clear();
                            }
                        }
                        Ok(())
                    })?;
                if !buffer.is_empty() {
                    install_piece(t, req, &e, &mut offset, &buffer)?;
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

/// A piece is durable before its canonical rows are written. A later failure
/// rolls back those rows and the extent deletion. Prepared files may remain
/// unreferenced after rollback, as with the other blob-writing paths.
fn install_piece(
    t: &Transactions<'_, '_>,
    req: &ExciseRequest,
    e: &Extent,
    offset: &mut u64,
    bytes: &[u8],
) -> Result<()> {
    crate::fault::check("excision_text.before_piece", &req.id)?;
    let piece = Piece::prepare(t.store, bytes)?;
    record_piece(t, &piece)?;
    t.archive.execute(
        "INSERT INTO extent(input,chain,offset,length,blob,blob_offset) VALUES (?1,?2,?3,?4,?5,0)",
        params![
            e.input,
            e.chain,
            *offset as i64,
            piece.size as i64,
            &piece.hash.0[..]
        ],
    )?;
    *offset += piece.size;
    Ok(())
}
