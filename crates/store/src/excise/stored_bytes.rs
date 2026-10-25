//! Excision in stored bytes: finding the text in blobs, overwriting a
//! session's records or the text with spaces of the same length, and
//! blanking a shared document's texts where they stand.

use super::*;

mod erasures;
mod plain_blob;
mod records;

/// Blobs holding part of a match of `needle` in a stored chain, including
/// decoded JSON spellings and matches spanning several blobs. Uses the
/// excision's first pass, so a format excision cannot rewrite returns an error.
pub fn stored_blobs_containing(archive: &crate::Archive, needle: &str) -> Result<Vec<BlobHash>> {
    let conn = crate::archive::open_read(&archive.archive_db())?;
    let store = hashed_blobs::Store::open(archive.root.clone(), vec![], 0o600)?;
    stored_text::find_in(&conn, &store, needle)?.blobs(&conn)
}

/// Every `table.column` of `index.db` holding `needle` in some row, as
/// written or as JSON escapes it; compressed values are read unpacked,
/// and the search vocabularies are checked for its words. How a text
/// excision is checked afterwards: the answer must be empty.
/// Recognized compressed frames use the row codec; unreadable frames are errors.
pub fn index_columns_containing(archive: &crate::Archive, needle: &str) -> Result<Vec<String>> {
    let conn = crate::archive::open_read(&archive.index_db())?;
    let needles = needles(needle);
    let tables: Vec<String> = conn
        .prepare(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND sql NOT LIKE 'CREATE VIRTUAL%'
             AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '%_fts_%'
             AND name NOT LIKE 'tool_call_trigram_%'",
        )?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let mut out = Vec::new();
    for table in tables {
        let cols: Vec<String> = conn
            .prepare(&format!("SELECT name FROM pragma_table_info('{table}')"))?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        for col in cols {
            let mut stmt = conn.prepare(&format!("SELECT \"{col}\" FROM \"{table}\""))?;
            let mut rows = stmt.query([])?;
            while let Some(row) = rows.next()? {
                let bytes = match row.get_ref(0)? {
                    rusqlite::types::ValueRef::Text(t) => t.to_vec(),
                    rusqlite::types::ValueRef::Blob(b)
                        if b.starts_with(&zstd::zstd_safe::MAGICNUMBER.to_le_bytes()) =>
                    {
                        crate::encoding::unpack_bytes(b)?
                    }
                    rusqlite::types::ValueRef::Blob(b) => b.to_vec(),
                    _ => continue,
                };
                if !find_all(&bytes, &needles).is_empty() {
                    out.push(format!("{table}.{col}"));
                    break;
                }
            }
        }
    }
    let words = crate::search::words(needle);
    for vocab in ["fts_vocab_text", "fts_vocab_output"] {
        let mut stmt = conn.prepare(&format!(
            "SELECT 1 FROM {vocab} WHERE term = ?1 AND cnt > 0"
        ))?;
        for w in &words {
            if stmt.exists([w])? {
                out.push(format!("{vocab}.term"));
                break;
            }
        }
    }
    Ok(out)
}

/// The same number of spaces as `n` has bytes.
pub(super) fn blank(n: &[u8]) -> String {
    " ".repeat(n.len())
}

/// The forms a string takes in stored bytes: as written, and as JSON
/// escapes it when it holds quotes, backslashes or control characters.
pub(super) fn needles(text: &str) -> Vec<Vec<u8>> {
    let mut out = vec![text.as_bytes().to_vec()];
    if let Ok(json) = serde_json::to_string(text) {
        let inner = json.as_bytes()[1..json.len() - 1].to_vec();
        if inner != out[0] {
            out.push(inner);
        }
    }
    out
}

pub(super) fn find_all(hay: &[u8], needles: &[Vec<u8>]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for n in needles {
        if n.is_empty() || n.len() > hay.len() {
            continue;
        }
        let mut i = 0;
        while let Some(p) = hay[i..].windows(n.len()).position(|w| w == n.as_slice()) {
            out.push((i + p, i + p + n.len()));
            i += p + n.len();
            if i + n.len() > hay.len() {
                break;
            }
        }
    }
    out.sort();
    out
}

/// A blob replaced by an overwritten copy, marked for deletion.
pub(super) struct Gone {
    pub(super) hash: [u8; 32],
}

fn record_replacement(
    t: &Transactions<'_, '_>,
    old: &BlobHash,
    piece: Piece,
) -> Result<Option<Gone>> {
    if piece.hash == *old {
        return Ok(None);
    }
    crate::blobs::refuse_delta_base(t.archive, old)?;
    record_piece(t, &piece)?;
    t.archive.execute(
        "UPDATE extent SET blob = ?2 WHERE blob = ?1",
        params![&old.0[..], &piece.hash.0[..]],
    )?;
    crate::blobs::mark_for_deletion(t.archive, old)?;
    Ok(Some(Gone { hash: old.0 }))
}

/// Overwrites a shared input's spans in every blob its chains use.
pub(super) fn overwrite(
    t: &Transactions<'_, '_>,
    input: &str,
    spans: &Spans,
    req: &ExciseRequest,
    refused: &mut RefusedJson,
) -> Result<()> {
    let erasures = erasures::Erasures::new()?;
    let mut ext = t.archive.prepare(
        "SELECT offset, length, blob, blob_offset FROM extent
         WHERE input = ?1 AND chain = ?2 AND offset < ?4 AND offset + length > ?3",
    )?;
    spans.visit(input, |s| {
        let rows = ext.query_map(params![input, s.chain, s.start, s.end], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, Vec<u8>>(2)?,
                r.get::<_, i64>(3)?,
            ))
        })?;
        for row in rows {
            let (offset, length, blob, blob_offset) = row?;
            let from = s.start.max(offset) - offset + blob_offset;
            let to = s.end.min(offset + length) - offset + blob_offset;
            erasures.request(&blob, from as usize, to as usize)?;
        }
        Ok(())
    })?;
    records::preflight(t, input, spans, &erasures)?;
    if req.dry_run {
        return Ok(());
    }
    let mut spans_json = Vec::new();
    erasures.visit(|hash| {
        let arr = <[u8; 32]>::try_from(hash)
            .map_err(|_| StoreError::Invalid("invalid source blob hash".into()))?;
        let old = BlobHash(arr);
        let bytes = plain_blob::PlainBlob::load(t, input, old)?;
        erasures.visit_ranges(hash, |s, e| {
            spans_json.push(serde_json::json!([hex(hash), s, e - s]));
            Ok(())
        })?;
        if let Some(gone) = bytes.replace(t, &old, |blank| erasures.visit_ranges(hash, blank))? {
            refused.blobs.push(hex(&gone.hash));
        }
        Ok(())
    })?;
    insert_excised_observations(t, input, req, Some(&serde_json::Value::Array(spans_json)))?;
    Ok(())
}

/// Shortest session text blanked in a shared whole document: a shorter
/// complete string value ("ok", "user") could be another record's value
/// or the document's own structure.
pub(super) const MIN_BLANKED: usize = 12;

/// The forms a text takes as a complete JSON string value: as `serde_json`
/// writes it, and with non-ASCII escaped as `\uXXXX` (Python's default).
pub(super) fn quoted(text: &str) -> Vec<Vec<u8>> {
    let Ok(plain) = serde_json::to_string(text) else {
        return Vec::new();
    };
    let mut ascii = String::with_capacity(plain.len());
    for ch in plain.chars() {
        if ch.is_ascii() {
            ascii.push(ch);
        } else {
            let mut units = [0u16; 2];
            for u in ch.encode_utf16(&mut units) {
                ascii.push_str(&format!("\\u{u:04x}"));
            }
        }
    }
    let mut out = vec![plain.into_bytes()];
    if ascii.as_bytes() != out[0].as_slice() {
        out.push(ascii.into_bytes());
    }
    out
}

/// A shared whole document: blanks the session's own texts where they
/// stand as complete JSON string values, quotes kept. Returns how many of
/// those texts are still in it (too short, or not found whole).
pub(super) fn blank_document(
    t: &Transactions<'_, '_>,
    input: &str,
    texts: &SessionTexts,
    req: &ExciseRequest,
    refused: &mut RefusedJson,
    done: &mut Excised,
) -> Result<u64> {
    let matches = erasures::Erasures::new()?;
    matches.freeze_literals(t.archive, input)?;
    texts.reset_found()?;
    let mut spans_json = Vec::new();
    matches.literal_blobs(|hash, ranges| {
        ranges.begin()?;
        let arr = <[u8; 32]>::try_from(hash)
            .map_err(|_| StoreError::Invalid("invalid source blob hash".into()))?;
        let old = BlobHash(arr);
        let bytes = plain_blob::PlainBlob::load(t, input, old)?;
        texts.visit(MIN_BLANKED, |text| {
            let mut found = false;
            for needle in quoted(text) {
                bytes.find(&needle, |start, end| {
                    // Keep quotes, and preserve every hit for counts and observations.
                    ranges.insert(start + 1, end - 1)?;
                    found = true;
                    Ok(())
                })?;
            }
            if found {
                texts.found(text)?;
            }
            Ok(())
        })?;
        let count = ranges.finish()?;
        if count == 0 {
            return Ok(());
        }
        *done.inputs_overwritten.entry(input.to_owned()).or_default() += count;
        if req.dry_run {
            return Ok(());
        }
        ranges.ranges(|start, end| {
            spans_json.push(serde_json::json!([hex(hash), start, end - start]));
            Ok(())
        })?;
        if let Some(gone) = bytes.replace(t, &old, |blank| ranges.ranges(blank))? {
            refused.blobs.push(hex(&gone.hash));
        }
        Ok(())
    })?;
    if !req.dry_run && !spans_json.is_empty() {
        insert_excised_observations(t, input, req, Some(&serde_json::Value::Array(spans_json)))?;
    }
    texts.missing()
}

pub(super) fn record_piece(t: &Transactions<'_, '_>, piece: &Piece) -> Result<()> {
    let (place, size, data) = match &piece.stored {
        Stored::Inline(z) => ("inline", z.len() as i64, Some(z.as_slice())),
        Stored::File { stored_size } => ("file", *stored_size as i64, None),
    };
    t.archive.execute(
        "INSERT OR IGNORE INTO blob(hash, size, stored_size, place, encoding, state, data)
        VALUES (?1, ?2, ?3, ?4, 'plain', 'present', ?5)",
        params![&piece.hash.0[..], piece.size as i64, size, place, data],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;
