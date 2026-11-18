//! Stored text bodies. Texts are compressed together in groups of about
//! `GROUP_BYTES`: each `text_group` row is one zstd frame holding the UTF-8
//! of its texts one after another, and each `text` row names its group, start
//! and length. Compressing neighbouring texts together keeps most of zstd's
//! ratio, which texts compressed one by one lose; reading one text decodes
//! its group. Groups are made on the parse workers, before the writer.
use crate::{Result, StoreError};
use rusqlite::Connection;
use std::cell::RefCell;
use std::collections::HashMap;

/// Uncompressed bytes a group collects before it is closed. A text longer
/// than this is a group of its own.
const GROUP_BYTES: usize = 64 * 1024;
/// Workers compress; the writer and readers only copy and decode.
const LEVEL: i32 = 3;
/// The decoded group a thread last read is kept for the next read when it is
/// no larger than this, so a group holding one very large text is not held.
const KEPT_GROUP_BYTES: usize = 4 * GROUP_BYTES;

/// Texts a parse step brings, compressed into groups on the worker. The
/// writer stores a group once one of its texts is new to the index.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PreparedTexts {
    texts: HashMap<[u8; 32], PreparedText>,
    groups: Vec<PreparedGroup>,
}

#[cfg(test)]
thread_local! {
    static PACK_INPUT_BYTES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Where one prepared text lies in its step's groups.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PreparedText {
    group: usize,
    start: u64,
    length: u64,
    pub(crate) normalized_hash: [u8; 32],
}

/// One compressed group and the hash that names it while its step is
/// written.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PreparedGroup {
    hash: [u8; 32],
    body: Vec<u8>,
}

/// A prepared text with its group, as the writer stores them.
#[derive(Clone, Copy)]
pub(crate) struct Prepared<'a> {
    pub(crate) text: &'a PreparedText,
    pub(crate) group: &'a PreparedGroup,
}

impl PreparedTexts {
    #[cfg(test)]
    pub(crate) fn count_input_bytes<T>(run: impl FnOnce() -> T) -> (T, usize) {
        PACK_INPUT_BYTES.with(|bytes| bytes.set(0));
        let result = run();
        (result, PACK_INPUT_BYTES.with(std::cell::Cell::get))
    }

    /// Prepares `texts` in order; repeated and already prepared texts are
    /// skipped.
    pub(crate) fn pack<'t>(texts: impl IntoIterator<Item = &'t str>) -> Result<Self> {
        let mut prepared = Self::default();
        prepared.add(texts)?;
        Ok(prepared)
    }

    /// Adds the texts not yet prepared, in new groups after the existing ones.
    pub(crate) fn add<'t>(&mut self, texts: impl IntoIterator<Item = &'t str>) -> Result<()> {
        let mut pending = Vec::new();
        // Counted apart from the bytes: an empty text still needs its group.
        let mut members = 0;
        for text in texts {
            #[cfg(test)]
            PACK_INPUT_BYTES.with(|bytes| bytes.set(bytes.get() + text.len()));
            let hash = *blake3::hash(text.as_bytes()).as_bytes();
            if self.texts.contains_key(&hash) {
                continue;
            }
            if members > 0 && pending.len() + text.len() > GROUP_BYTES {
                self.close(&mut pending)?;
                members = 0;
            }
            self.texts.insert(
                hash,
                PreparedText {
                    group: self.groups.len(),
                    start: pending.len() as u64,
                    length: text.len() as u64,
                    normalized_hash: crate::encoding::normalized_text_hash(text),
                },
            );
            pending.extend_from_slice(text.as_bytes());
            members += 1;
        }
        if members > 0 {
            self.close(&mut pending)?;
        }
        Ok(())
    }

    fn close(&mut self, pending: &mut Vec<u8>) -> Result<()> {
        let body = COMPRESSOR.with(|compressor| {
            let mut compressor = compressor.borrow_mut();
            let compressor = match &mut *compressor {
                Some(compressor) => compressor,
                empty => empty.insert(zstd::bulk::Compressor::new(LEVEL)?),
            };
            compressor.compress(pending)
        })?;
        self.groups.push(PreparedGroup {
            hash: *blake3::hash(&body).as_bytes(),
            body,
        });
        pending.clear();
        Ok(())
    }

    pub(crate) fn get(&self, hash: &[u8; 32]) -> Option<Prepared<'_>> {
        let text = self.texts.get(hash)?;
        Some(Prepared {
            text,
            group: &self.groups[text.group],
        })
    }

    #[cfg(any(test, feature = "fixture"))]
    pub(crate) fn contains(&self, hash: &[u8; 32]) -> bool {
        self.texts.contains_key(hash)
    }
}

/// The stored form of prepared texts in a staged step file.
#[derive(serde::Serialize, serde::Deserialize)]
struct StoredTexts(Vec<([u8; 32], PreparedText)>, Vec<PreparedGroup>);

impl serde::Serialize for PreparedTexts {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let texts = self.texts.iter().map(|(h, t)| (*h, t.clone())).collect();
        StoredTexts(texts, self.groups.clone()).serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for PreparedTexts {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let StoredTexts(texts, groups) = StoredTexts::deserialize(deserializer)?;
        if texts.iter().any(|(_, text)| text.group >= groups.len()) {
            return Err(serde::de::Error::custom("a prepared text names no group"));
        }
        Ok(Self {
            texts: texts.into_iter().collect(),
            groups,
        })
    }
}

/// The group rows the writer stored in its current transaction, by the hash
/// that names each prepared group. A step is written in one transaction, so
/// the map ends with it.
pub(crate) struct StoredGroups {
    rows: RefCell<HashMap<[u8; 32], i64>>,
    entries: usize,
}

impl Default for StoredGroups {
    fn default() -> Self {
        Self::new(1 << 20)
    }
}

impl StoredGroups {
    pub(crate) fn new(bytes: u64) -> Self {
        Self {
            rows: RefCell::new(HashMap::new()),
            entries: (bytes / 128).max(1) as usize,
        }
    }
    /// Stores `text`'s group unless this transaction already did, and returns
    /// its row and whether this call inserted it.
    pub(crate) fn row(&self, conn: &Connection, prepared: Prepared<'_>) -> Result<(i64, bool)> {
        if let Some(&id) = self.rows.borrow().get(&prepared.group.hash) {
            return Ok((id, false));
        }
        conn.prepare_cached("INSERT INTO text_group(body) VALUES (?1)")?
            .execute([&prepared.group.body])?;
        let id = conn.last_insert_rowid();
        let mut rows = self.rows.borrow_mut();
        if rows.len() >= self.entries {
            rows.clear();
        }
        rows.insert(prepared.group.hash, id);
        Ok((id, true))
    }

    pub(crate) fn clear(&self) {
        self.rows.borrow_mut().clear();
    }
}

/// Tries to insert a prepared text, returning no row when its identity or
/// hash is already stored. A group used only by that skipped insert is removed.
pub(crate) fn insert(
    conn: &Connection,
    groups: &StoredGroups,
    id: Option<i64>,
    hash: &[u8; 32],
    prepared: Prepared<'_>,
    indexed: i64,
) -> Result<Option<i64>> {
    let (group, inserted) = groups.row(conn, prepared)?;
    let row = conn
        .prepare_cached(
            "INSERT OR IGNORE INTO text(id,hash,normalized_hash,indexed,group_id,start,length)
             VALUES (?1,?2,?3,?4,?5,?6,?7) RETURNING id",
        )?
        .query_row(
            rusqlite::params![
                id,
                hash.as_slice(),
                prepared.text.normalized_hash.as_slice(),
                indexed,
                group,
                prepared.text.start as i64,
                prepared.text.length as i64
            ],
            |r| r.get::<_, i64>(0),
        );
    match row {
        Ok(id) => Ok(Some(id)),
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            if inserted {
                remove_empty_group(conn, group)?;
                groups.clear();
            }
            Ok(None)
        }
        Err(error) => Err(error.into()),
    }
}

/// Removes a text row, and its group once no text is left in it.
pub(crate) fn delete(conn: &Connection, id: i64) -> Result<()> {
    let group: Option<i64> = rusqlite::OptionalExtension::optional(
        conn.prepare_cached("DELETE FROM text WHERE id=?1 RETURNING group_id")?
            .query_row([id], |r| r.get(0)),
    )?;
    if let Some(group) = group {
        remove_empty_group(conn, group)?;
    }
    Ok(())
}

fn remove_empty_group(conn: &Connection, group: i64) -> Result<()> {
    conn.prepare_cached(
        "DELETE FROM text_group WHERE id=?1 AND NOT EXISTS(SELECT 1 FROM text WHERE group_id=?1)",
    )?
    .execute([group])?;
    Ok(())
}

/// The text in columns `at`, `at + 1` and `at + 2` of `row` (its group's
/// body, start and length), or none where a left join found no text.
pub(crate) fn column(row: &rusqlite::Row<'_>, at: usize) -> Result<Option<String>> {
    let Some(body) = row
        .get_ref(at)?
        .as_blob_or_null()
        .map_err(rusqlite::Error::from)?
    else {
        return Ok(None);
    };
    slice(body, row.get(at + 1)?, row.get(at + 2)?).map(Some)
}

/// The text an inner join selected into columns `at..at + 3`.
pub(crate) fn joined(row: &rusqlite::Row<'_>, at: usize) -> Result<String> {
    column(row, at)?.ok_or_else(|| crate::StoreError::stored(0, "a text row has no group"))
}

/// One stored text.
pub(crate) fn read(conn: &Connection, id: i64) -> Result<String> {
    let (body, start, length) = conn
        .prepare_cached(
            "SELECT g.body,t.start,t.length FROM text t JOIN text_group g ON g.id=t.group_id
             WHERE t.id=?1",
        )?
        .query_row([id], |r| <(Vec<u8>, _, _)>::try_from(r))?;
    slice(&body, start, length)
}

thread_local! {
    /// Each worker reuses its compression context: creating one per step
    /// costs more than compressing a small step.
    static COMPRESSOR: RefCell<Option<zstd::bulk::Compressor<'static>>> = const { RefCell::new(None) };
    /// The compressed and decoded bytes of the group this thread last read.
    /// Groups are compared by their bytes, so the entry is right for any
    /// connection and any snapshot.
    static LAST_GROUP: RefCell<(Vec<u8>, Vec<u8>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

/// The text at `start..start + length` of the group `body`. Consecutive reads
/// from one group decode it once.
pub(crate) fn slice(body: &[u8], start: i64, length: i64) -> Result<String> {
    let range = text_range(start, length)?;
    LAST_GROUP.with(|last| {
        let mut last = last.borrow_mut();
        if last.0 != body {
            let decoded = decode(body)?;
            if decoded.len() > KEPT_GROUP_BYTES {
                return text_of(&decoded, range);
            }
            *last = (body.to_vec(), decoded);
        }
        text_of(&last.1, range)
    })
}

pub(super) fn decode(body: &[u8]) -> Result<Vec<u8>> {
    Ok(crate::encoding::unpack_bytes(body)?)
}

pub(super) fn slice_decoded(group: &[u8], start: i64, length: i64) -> Result<String> {
    text_of(group, text_range(start, length)?)
}

fn text_range(start: i64, length: i64) -> Result<std::ops::Range<usize>> {
    usize::try_from(start)
        .ok()
        .zip(usize::try_from(length).ok())
        .and_then(|(start, length)| Some(start..start.checked_add(length)?))
        .ok_or_else(|| crate::StoreError::stored(0, "a stored text has an invalid place"))
}

fn text_of(group: &[u8], range: std::ops::Range<usize>) -> Result<String> {
    let bytes = group
        .get(range)
        .ok_or_else(|| crate::StoreError::stored(0, "a stored text lies outside its group"))?;
    String::from_utf8(bytes.to_vec()).map_err(|e| StoreError::Io(std::io::Error::other(e)))
}

#[cfg(test)]
mod tests;
