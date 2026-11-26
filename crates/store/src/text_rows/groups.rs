//! Stored text bodies. Texts are compressed together in content groups
//! ([`crate::content`]) of their own: each holds the UTF-8 of its texts one
//! after another, and each `text` row names its group, start and length.
//! Compressing neighbouring texts together keeps most of zstd's ratio, which
//! texts compressed one by one lose; reading one text decodes its group.
//! Groups are made on the parse workers, before the writer.
use crate::content::{Group, Packer};
use crate::row::{Rows as _, row};
use crate::{Result, StoreError};
use rusqlite::Connection;
use std::collections::HashMap;

/// Texts a parse step brings, compressed into groups on the worker. The
/// writer stores a group once one of its texts is new to the index.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct PreparedTexts {
    texts: HashMap<[u8; 32], PreparedText>,
    groups: Vec<Group>,
}

/// Where one prepared text lies in its step's groups.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct PreparedText {
    group: usize,
    start: u64,
    length: u64,
    pub(crate) normalized_hash: [u8; 32],
}

/// A prepared text with its group, as the writer stores them.
#[derive(Clone, Copy)]
pub(crate) struct Prepared<'a> {
    pub(crate) text: &'a PreparedText,
    pub(crate) group: &'a Group,
}

impl PreparedTexts {
    /// Prepares `texts` in order; repeated and already prepared texts are
    /// skipped.
    pub(crate) fn pack<'t>(texts: impl IntoIterator<Item = &'t str>) -> Result<Self> {
        let mut prepared = Self::default();
        prepared.add(texts)?;
        Ok(prepared)
    }

    /// Adds the texts not yet prepared, in new groups after the existing ones.
    pub(crate) fn add<'t>(&mut self, texts: impl IntoIterator<Item = &'t str>) -> Result<()> {
        let mut packer = Packer::continuing(crate::content::Groups::from(std::mem::take(
            &mut self.groups,
        )));
        for text in texts {
            let hash = *blake3::hash(text.as_bytes()).as_bytes();
            if self.texts.contains_key(&hash) {
                continue;
            }
            let placed = packer.add(text.as_bytes())?;
            self.texts.insert(
                hash,
                PreparedText {
                    group: placed.group,
                    start: placed.start,
                    length: placed.length,
                    normalized_hash: crate::encoding::normalized_text_hash(text),
                },
            );
        }
        self.groups = packer.finish()?.into();
        Ok(())
    }

    pub(crate) fn get(&self, hash: &[u8; 32]) -> Option<Prepared<'_>> {
        let text = self.texts.get(hash)?;
        Some(Prepared {
            text,
            group: &self.groups[text.group],
        })
    }
}

/// The stored form of prepared texts in a staged step file.
#[derive(serde::Serialize, serde::Deserialize)]
struct StoredTexts(Vec<([u8; 32], PreparedText)>, Vec<Group>);

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

row! {
    /// A text row as it is inserted; `id` and `indexed` take their defaults.
    struct NewText {
        hash: [u8; 32],
        normalized_hash: [u8; 32],
        group_id: i64,
        start: i64,
        length: i64,
    }
    statements {
        const INSERT = "INSERT INTO text(hash, normalized_hash, group_id, start, length) VALUES (?1, ?2, ?3, ?4, ?5)";
    }
}

/// Inserts a prepared text and returns its key. Its group is stored once
/// ([`crate::content::store`]).
pub(crate) fn insert(conn: &Connection, hash: &[u8; 32], prepared: Prepared<'_>) -> Result<i64> {
    let row = NewText {
        hash: *hash,
        normalized_hash: prepared.text.normalized_hash,
        group_id: crate::content::store(conn, prepared.group)?,
        start: prepared.text.start as i64,
        length: prepared.text.length as i64,
    };
    conn.insert(&NewText::INSERT, &row)?;
    Ok(conn.last_insert_rowid())
}

/// Removes a text row, and its group once nothing names it.
pub(crate) fn delete(conn: &Connection, id: i64) -> Result<()> {
    let group: Option<i64> = conn.value("DELETE FROM text WHERE id=?1 RETURNING group_id", [id])?;
    if let Some(group) = group {
        crate::content::release(conn, group)?;
    }
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
    slice(body, row.get(at + 1)?, row.get(at + 2)?)
        .map(Some)
        .map_err(|error| match error {
            // The error names the column the text was read from.
            StoreError::StoredValue { reason, .. } => StoreError::stored(at, reason),
            other => other,
        })
}

/// The text an inner join selected into columns `at..at + 3`.
pub(crate) fn joined(row: &rusqlite::Row<'_>, at: usize) -> Result<String> {
    column(row, at)?.ok_or_else(|| StoreError::stored(at, "a text row has no group"))
}

row! {
    /// A text's group body and its place there.
    struct StoredText {
        body: Vec<u8>,
        start: i64,
        length: i64,
    }
    statements {
        const OF_ID = "SELECT g.body, t.start, t.length FROM text t JOIN content_group g ON g.id=t.group_id
            WHERE t.id=?1";
    }
}

/// One stored text.
pub(crate) fn read(conn: &Connection, id: i64) -> Result<String> {
    let text = conn.one(&StoredText::OF_ID, [id])?;
    slice(&text.body, text.start, text.length)
}

/// The text at `start..start + length` of the group `body`. Consecutive reads
/// from one group decode it once.
pub(crate) fn slice(body: &[u8], start: i64, length: i64) -> Result<String> {
    crate::content::text(body, start, length)
}

pub(super) fn decode(body: &[u8]) -> Result<Vec<u8>> {
    crate::content::decode(body)
}

pub(super) fn slice_decoded(group: &[u8], start: i64, length: i64) -> Result<String> {
    let bytes = group
        .get(crate::content::range(start, length)?)
        .ok_or_else(|| StoreError::stored(0, "a stored text lies outside its group"))?;
    String::from_utf8(bytes.to_vec())
        .map_err(|_| StoreError::stored(0, "a stored text is not valid UTF-8"))
}

#[cfg(test)]
mod tests;

/// The rows this module declares statements for, for the every-statement test.
#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<StoredText>();
    check.row::<NewText>();
}
