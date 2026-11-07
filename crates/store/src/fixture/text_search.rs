//! Checks that a text is gone, by plain searches that share no code with
//! retained bytes: every stored chain, and every column of `index.db`.

use crate::{Result, StoreError};
use polylogue_model::BlobHash;
use rusqlite::params;
use std::collections::BTreeSet;
use std::io::Read;
use std::sync::Arc;

/// The forms a text takes in stored bytes: as written, and as JSON escapes
/// it.
fn spellings(needle: &str) -> Result<Vec<Vec<u8>>> {
    let escaped = serde_json::to_string(needle)?;
    let mut out = vec![needle.as_bytes().to_vec()];
    let inner = escaped.as_bytes()[1..escaped.len() - 1].to_vec();
    if inner != out[0] {
        out.push(inner);
    }
    Ok(out)
}

/// Blobs holding part of `needle` in a stored chain.
pub fn stored_blobs_containing(archive: &crate::Archive, needle: &str) -> Result<Vec<BlobHash>> {
    let conn = crate::archive::open_read(&archive.archive_db())?;
    let store = Arc::new(archive.blobs()?);
    let chains: Vec<(String, i64, i64)> = conn
        .prepare("SELECT input, chain, max(length) FROM revision GROUP BY input, chain")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    let mut out = BTreeSet::new();
    for (input, chain, length) in chains {
        let bytes = crate::blobs::StoredChain::load(&conn, "", &input, chain, length as u64)?;
        let mut all = Vec::new();
        bytes.reader(Arc::clone(&store)).read_to_end(&mut all)?;
        for spelling in spellings(needle)? {
            for found in memchr::memmem::find_iter(&all, &spelling) {
                let (start, end) = (found as i64, (found + spelling.len()) as i64);
                let mut held = conn.prepare(
                    "SELECT blob FROM extent WHERE input = ?1 AND chain = ?2
                     AND offset < ?4 AND offset + length > ?3",
                )?;
                for blob in held.query_map(params![input, chain, start, end], |r| {
                    r.get::<_, Vec<u8>>(0)
                })? {
                    out.insert(BlobHash(
                        blob?
                            .try_into()
                            .map_err(|_| StoreError::Invalid("invalid blob hash".into()))?,
                    ));
                }
            }
        }
    }
    Ok(out.into_iter().collect())
}

/// Every `table.column` of `index.db` holding `needle` in some row;
/// compressed values are read unpacked, and the search vocabularies are
/// checked for its words.
pub fn index_columns_containing(archive: &crate::Archive, needle: &str) -> Result<Vec<String>> {
    let conn = crate::archive::open_read(&archive.index_db())?;
    let spellings = spellings(needle)?;
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
        let columns: Vec<String> = conn
            .prepare(&format!("SELECT name FROM pragma_table_info('{table}')"))?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        for column in columns {
            let mut stmt = conn.prepare(&format!("SELECT \"{column}\" FROM \"{table}\""))?;
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
                if spellings
                    .iter()
                    .any(|s| memchr::memmem::find(&bytes, s).is_some())
                {
                    out.push(format!("{table}.{column}"));
                    break;
                }
            }
        }
    }
    let words = crate::search::words(needle);
    for vocabulary in ["fts_vocab_text", "fts_vocab_output"] {
        let mut stmt = conn.prepare(&format!(
            "SELECT 1 FROM {vocabulary} WHERE term = ?1 AND cnt > 0"
        ))?;
        for word in &words {
            if stmt.exists([word])? {
                out.push(format!("{vocabulary}.term"));
                break;
            }
        }
    }
    Ok(out)
}
