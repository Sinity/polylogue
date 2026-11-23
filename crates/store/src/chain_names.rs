//! A chain's committed rows, read to settle the names that belong to the
//! chain when its reading is extended (`polylogue_sources::chain_names`),
//! and the alias rows readers resolve a record's name through.
//!
//! Three things a reading's rows keep for this, each as the source wrote
//! it: the response key of a message (`message_source.response_key`), the
//! content name of a content-named message (`message_source.content_name`)
//! and the alias rows (`message_alias`).
use crate::ConnExt as _;
use crate::{Archive, Result, StoreError};
use polylogue_model::{IdKind, MessageId, MessageRow, SessionId};
use rusqlite::{Connection, OptionalExtension, params};

/// The committed rows of one chain whose reading stands at a cursor.
pub struct ChainRows {
    conn: Connection,
    input_pk: i64,
    chain: i64,
    has_aliases: Option<bool>,
}

impl ChainRows {
    /// The rows of `chain` of `input`, when its committed reading is under
    /// `reading_version` and stands exactly at `cursor`. `None` otherwise:
    /// the step does not continue that reading and is read from the start.
    pub fn open(
        archive: &Archive,
        input: &str,
        chain: i64,
        reading_version: &str,
        cursor: u64,
    ) -> Result<Option<Self>> {
        let conn = archive.open_index_read()?;
        let cursor = i64::try_from(cursor)
            .map_err(|_| StoreError::Invalid("a cursor exceeds SQLite's integer range".into()))?;
        let input_pk = conn
            .query_row(
                "SELECT i.pk FROM input_ref i JOIN parse_state p ON p.input_pk=i.pk
                 WHERE i.id=?1 AND p.chain=?2 AND p.reading_version=?3 AND p.parsed_upto=?4",
                params![input, chain, reading_version, cursor],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        Ok(input_pk.map(|input_pk| Self {
            conn,
            input_pk,
            chain,
            has_aliases: None,
        }))
    }

    /// The message the chain's rows of `session` with this response key form.
    pub fn response(&mut self, session: &SessionId, key: &str) -> Result<Option<MessageRow>> {
        let rows = self
            .conn
            .prepare_cached(
                "SELECT s.version_pk,v.version_hash,s.source_start,s.source_end
                 FROM message_source s JOIN message_version v ON v.pk=s.version_pk
                 WHERE s.input_pk=?1 AND s.chain=?2 AND s.response_key=?3",
            )?
            .query_map(params![self.input_pk, self.chain, key], |row| {
                <(i64, Vec<u8>, i64, i64)>::try_from(row)
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for (pk, hash, offset, end) in rows {
            let (message, _) = crate::versions::read_message_by_key(&self.conn, pk, &hash)?
                .ok_or_else(|| {
                    StoreError::Invalid("a response's message version is absent".into())
                })?;
            if message.session != *session {
                continue;
            }
            let position = |value: i64| {
                u64::try_from(value).map_err(|_| StoreError::stored(0, "negative source position"))
            };
            return Ok(Some(MessageRow {
                offset: position(offset)?,
                end: position(end)?,
                id_kind: IdKind::Native,
                page_text: false,
                response: Some(key.to_owned()),
                message,
            }));
        }
        Ok(None)
    }

    /// What an alias row of the chain says `name` stands for.
    pub fn alias(&mut self, name: &MessageId) -> Result<Option<Option<MessageId>>> {
        let has_aliases = match self.has_aliases {
            Some(has) => has,
            None => {
                let has = self.conn.exists(
                    "SELECT EXISTS(SELECT 1 FROM message_alias WHERE input_pk=?1 AND chain=?2)",
                    params![self.input_pk, self.chain],
                )?;
                *self.has_aliases.insert(has)
            }
        };
        if !has_aliases {
            return Ok(None);
        }
        let target = self
            .conn
            .prepare_cached(
                "SELECT target FROM message_alias WHERE input_pk=?1 AND chain=?2 AND name=?3",
            )?
            .query_row(
                params![self.input_pk, self.chain, name.to_string()],
                |row| row.get::<_, Option<String>>(0),
            )
            .optional()?;
        target
            .map(|target| {
                target
                    .map(|target| MessageId::parse(&target).map_err(StoreError::invalid))
                    .transpose()
            })
            .transpose()
    }

    /// How many messages of the chain have this content name.
    pub fn content_count(&mut self, name: &str) -> Result<u64> {
        self.count(name, i64::MAX)
    }

    /// How many messages of the chain have this content name and start
    /// before `position`.
    pub fn content_before(&mut self, name: &str, position: u64) -> Result<u64> {
        self.count(name, i64::try_from(position).unwrap_or(i64::MAX))
    }

    fn count(&self, name: &str, before: i64) -> Result<u64> {
        let count: i64 = self
            .conn
            .prepare_cached(
                "SELECT count(*) FROM message_source
                 WHERE input_pk=?1 AND chain=?2 AND content_name=?3 AND source_start<?4",
            )?
            .query_row(params![self.input_pk, self.chain, name, before], |row| {
                row.get(0)
            })?;
        Ok(count as u64)
    }
}

/// The content name of a settled content ID (`<name>-<number>`), which the
/// chain's rows keep to number the next message with that name.
pub(crate) fn content_name(row: &MessageRow) -> Option<&str> {
    (row.id_kind == IdKind::Content)
        .then(|| row.message.id.native().as_str().rsplit_once('-'))
        .flatten()
        .map(|(name, _)| name)
}

/// A reference to a record by its own name, resolved through the alias rows
/// to the message it stands for. A name that is a message's own ID, or that
/// no alias row holds, stays as it is. Authored targets keep the name they
/// were written with; only lookup follows the alias.
pub(crate) fn resolved_message_sql(
    id: &MessageId,
    bind: &mut dyn FnMut(String) -> String,
) -> String {
    let original = bind(id.to_string());
    format!(
        "CASE WHEN EXISTS(SELECT 1 FROM message direct JOIN message_source ds ON ds.message_pk=direct.pk WHERE direct.id={original}) THEN {original} ELSE pl_source_alias((SELECT json_group_array(DISTINCT a.target) FROM message_alias a JOIN parse_state ps ON ps.input_pk=a.input_pk AND ps.chain=a.chain WHERE a.name={original}),{original}) END"
    )
}

pub(crate) fn resolved_message(conn: &Connection, id: &MessageId) -> Result<MessageId> {
    let mut params = Vec::new();
    let expression = resolved_message_sql(id, &mut |value| {
        params.push(value);
        format!("?{}", params.len())
    });
    let value: String = conn.scalar(
        &format!("SELECT {expression}"),
        rusqlite::params_from_iter(params),
    )?;
    MessageId::parse(&value).map_err(StoreError::invalid)
}
