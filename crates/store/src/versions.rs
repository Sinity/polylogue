//! Packed readings and their shared text. Source rows choose what is shown.
use crate::{Result, encoding, first_build::FirstBuild};
use polylogue_model::{BlobHash, Block, Message, bundle::MessageProjection};
use rusqlite::{Connection, OptionalExtension, params};
use serde::de::DeserializeOwned;
use std::collections::HashMap;

#[cfg(test)]
mod tests;

/// Each kind declares its source key, stored payload and shown-version order.
#[derive(Clone, Copy)]
pub(crate) enum VersionKind {
    Message,
    Document,
    Event,
    Link,
}

impl VersionKind {
    pub(crate) const ALL: [Self; 4] = [Self::Message, Self::Document, Self::Event, Self::Link];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Message => "message",
            Self::Document => "document",
            Self::Event => "event",
            Self::Link => "link",
        }
    }

    fn content(self) -> bool {
        matches!(self, Self::Message | Self::Document)
    }

    pub(crate) fn retract(self, conn: &Connection, input: i64, chain: i64) -> Result<()> {
        let name = self.name();
        conn.prepare_cached(&format!(
            "DELETE FROM {name}_source WHERE input_pk=?1 AND chain=?2"
        ))?
        .execute(params![input, chain])?;
        Ok(())
    }
}

pub(crate) struct TextRows<'a> {
    pub packed: &'a HashMap<[u8; 32], Vec<u8>>,
    pub first_build: Option<&'a FirstBuild>,
    pub keys: &'a crate::writer::resolve::ResolvedKeys,
}

pub(crate) fn store(
    conn: &Connection,
    kind: VersionKind,
    hash: &[u8],
    body: &[u8],
    blocks: &[Block],
    texts: &[Option<[u8; 32]>],
    text_rows: TextRows<'_>,
) -> Result<()> {
    store_value(
        conn,
        kind,
        ValueRows {
            hash,
            body,
            blocks,
            texts,
        },
        text_rows,
        false,
    )
    .map(drop)
}

struct ValueRows<'a> {
    hash: &'a [u8],
    body: &'a [u8],
    blocks: &'a [Block],
    texts: &'a [Option<[u8; 32]>],
}

fn store_value(
    conn: &Connection,
    kind: VersionKind,
    ValueRows {
        hash,
        body,
        blocks,
        texts,
    }: ValueRows<'_>,
    text_rows: TextRows<'_>,
    replace: bool,
) -> Result<bool> {
    let TextRows {
        packed,
        first_build,
        keys,
    } = text_rows;
    let kind = kind.name();
    let sql = if replace {
        format!("INSERT INTO {kind}_version(version_hash,body) VALUES (?1,?2)
            ON CONFLICT(version_hash) DO UPDATE SET body=excluded.body WHERE body IS NOT excluded.body")
    } else {
        format!("INSERT OR IGNORE INTO {kind}_version(version_hash,body) VALUES (?1,?2)")
    };
    let inserted = conn.prepare_cached(&sql)?.execute(params![hash, body])?;
    if inserted == 0 && !replace {
        return Ok(false);
    }
    if replace {
        conn.execute(
            &format!("DELETE FROM {kind}_version_text WHERE version_hash=?1"),
            [hash],
        )?;
    }
    for (idx, text_hash) in texts.iter().enumerate() {
        if let Some(text_hash) = text_hash {
            let key = keys.get(crate::writer::resolve::Kind::Text, text_hash)?;
            if let Some(body) = packed.get(text_hash) {
                crate::text_rows::store_prepared(conn, key.pk, text_hash, body, first_build)?;
            } else if !key.existing {
                return Err(crate::StoreError::Invalid(
                    "a new text has no prepared bytes".into(),
                ));
            }
            conn.prepare_cached(&format!(
                "INSERT INTO {kind}_version_text(version_hash,idx,text_id) VALUES (?1,?2,?3)"
            ))?
            .execute(params![hash, idx as i64, key.pk])?;
        }
    }
    for block in blocks {
        let attachments: &[polylogue_model::Attachment] = match block {
            Block::Image(a) | Block::File(a) => std::slice::from_ref(a),
            Block::ToolResult { attachments, .. } => attachments,
            _ => &[],
        };
        for attachment in attachments {
            if let Some(blob) = attachment.blob {
                conn.prepare_cached(&format!("INSERT OR IGNORE INTO {kind}_version_blob(version_hash,blob_hash) VALUES (?1,?2)"))?.execute(params![hash,blob.0.as_slice()])?;
            }
        }
    }
    Ok(true)
}

pub(crate) fn store_message(
    conn: &Connection,
    message: &Message,
    prepared: &encoding::PreparedMessage,
    text_rows: TextRows<'_>,
    replace: bool,
    admitted: Option<u64>,
) -> Result<()> {
    prepared.projection.validate(message.blocks.len())?;
    if prepared.hash != prepared.projection.source_version.0 {
        return Err(polylogue_model::bundle::BundleError::SourceVersionMismatch.into());
    }
    let inserted = store_value(
        conn,
        VersionKind::Message,
        ValueRows {
            hash: &prepared.hash,
            body: &prepared.body,
            blocks: &message.blocks,
            texts: &prepared.texts,
        },
        text_rows,
        replace,
    )?;
    if inserted || replace {
        for (idx, (body, data)) in prepared.blocks.iter().zip(&prepared.data).enumerate() {
            let value_pk = store_block_payload(conn, body, data.as_deref())?;
            conn.prepare_cached(
                "INSERT INTO message_version_block(version_hash,idx,value_pk) VALUES (?1,?2,?3)
                 ON CONFLICT(version_hash,idx) DO UPDATE SET value_pk=excluded.value_pk
                 WHERE value_pk IS NOT excluded.value_pk",
            )?
            .execute(params![prepared.hash.as_slice(), idx as i64, value_pk])?;
        }
        conn.prepare_cached("DELETE FROM message_version_block WHERE version_hash=?1 AND idx>=?2")?
            .execute(params![
                prepared.hash.as_slice(),
                prepared.blocks.len() as i64
            ])?;
    }
    // A stored version no source row chose is deleted before the commit.
    conn.prepare_cached(
        "INSERT OR IGNORE INTO temp.unreferenced_version(kind,version_hash) VALUES ('message',?1)",
    )?
    .execute([prepared.hash.as_slice()])?;
    let layout = message_layout(&prepared.projection)?;
    conn.execute(
        "INSERT INTO message_projection(version_hash,layout,admitted) VALUES (?1,?2,?3)
         ON CONFLICT(version_hash) DO UPDATE SET layout=excluded.layout,
         admitted=CASE WHEN message_projection.admitted IS NULL THEN excluded.admitted
             WHEN excluded.admitted IS NULL THEN message_projection.admitted
             ELSE max(message_projection.admitted,excluded.admitted) END
         WHERE layout IS NOT excluded.layout OR (excluded.admitted IS NOT NULL
             AND (message_projection.admitted IS NULL OR excluded.admitted>message_projection.admitted))",
        params![
            prepared.hash.as_slice(),
            layout,
            admitted
                .map(i64::try_from)
                .transpose()
                .map_err(|_| crate::StoreError::Invalid(
                    "message admission exceeds SQLite's integer range".into()
                ))?
        ],
    )?;
    Ok(())
}

/// Writes an encoded payload once, shared by version history and selected rows.
pub(crate) fn store_block_payload(
    conn: &Connection,
    body: &[u8],
    data: Option<&[u8]>,
) -> Result<i64> {
    let hash = encoding::block_value_hash(body, data);
    if let Some(pk) = conn
        .prepare_cached("SELECT pk FROM block_value WHERE hash=?1")?
        .query_row([hash.as_slice()], |row| row.get(0))
        .optional()?
    {
        return Ok(pk);
    }
    conn.prepare_cached("INSERT INTO block_value(hash,body,data) VALUES (?1,?2,?3) RETURNING pk")?
        .query_row(params![hash.as_slice(), body, data], |row| row.get(0))
        .map_err(Into::into)
}

pub(crate) fn message_layout(projection: &MessageProjection) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&(
        projection.block_count,
        &projection.block_ordinals,
    ))?)
}

pub(crate) fn read<T: DeserializeOwned>(
    conn: &Connection,
    kind: VersionKind,
    hash: &[u8],
) -> Result<T> {
    let kind = kind.name();
    let body: Vec<u8> = conn
        .prepare_cached(&format!(
            "SELECT body FROM {kind}_version WHERE version_hash=?1"
        ))?
        .query_row([hash], |r| r.get(0))?;
    read_body(conn, kind, hash, &body)
}

fn read_body<T: DeserializeOwned>(
    conn: &Connection,
    kind: &str,
    hash: &[u8],
    body: &[u8],
) -> Result<T> {
    let mut value: serde_json::Value = encoding::unpack_json(body)?;
    if kind == "message" {
        let mut statement = conn.prepare_cached(
            "SELECT b.body,b.data FROM message_version_block v JOIN block_value b ON b.pk=v.value_pk
             WHERE v.version_hash=?1 ORDER BY v.idx",
        )?;
        let blocks = statement
            .query_map([hash], |row| {
                Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Option<Vec<u8>>>(1)?))
            })?
            .map(|row| {
                let (body, data) = row?;
                let mut block = encoding::unpack_block(&body)?;
                if let Some(data) = data {
                    encoding::restore_data(&mut block, &data)?;
                }
                Ok(serde_json::to_value(block)?)
            })
            .collect::<Result<Vec<_>>>()?;
        value["blocks"] = blocks.into();
    }
    let mut statement = conn.prepare_cached(&format!(
        "SELECT v.idx,t.body FROM {kind}_version_text v JOIN text t ON t.id=v.text_id WHERE v.version_hash=?1 ORDER BY v.idx"
    ))?;
    let mut rows = statement.query([hash])?;
    while let Some(row) = rows.next()? {
        let idx: i64 = row.get(0)?;
        let packed: Vec<u8> = row.get(1)?;
        let block = &mut value["blocks"][idx as usize];
        let field = if block.get("output").is_some() {
            "output"
        } else if block.get("written_text").is_some() {
            "written_text"
        } else {
            "text"
        };
        block[field] = encoding::unpack_text(&packed)?.into();
    }
    Ok(serde_json::from_value(value)?)
}

/// The value and its block positions belong to the caller's one read snapshot.
pub(crate) fn read_message(
    conn: &Connection,
    hash: &[u8],
) -> Result<Option<(Message, MessageProjection)>> {
    let stored = conn
        .query_row(
            "SELECT v.body,p.layout FROM message_version v
         LEFT JOIN message_projection p USING(version_hash) WHERE v.version_hash=?1",
            [hash],
            |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Option<Vec<u8>>>(1)?)),
        )
        .optional()?;
    let Some((body, layout)) = stored else {
        return Ok(None);
    };
    let message: Message = read_body(conn, "message", hash, &body)?;
    let projection = message_projection(hash, layout.as_deref(), message.blocks.len())?;
    projection.validate(message.blocks.len())?;
    if projection.block_count == message.blocks.len() as u64
        && message.content_hash()? != projection.source_version
    {
        return Err(polylogue_model::bundle::BundleError::SourceVersionMismatch.into());
    }
    Ok(Some((message, projection)))
}

pub(crate) fn message_projection(
    hash: &[u8],
    layout: Option<&[u8]>,
    blocks: usize,
) -> Result<MessageProjection> {
    let source_version = BlobHash(hash.try_into().map_err(|_| {
        crate::StoreError::Invalid("a message source version is not a BLAKE3 hash".into())
    })?);
    let (block_count, block_ordinals) = match layout {
        Some(layout) => encoding::read_json::<(u64, Vec<u64>)>(layout)?,
        None => (blocks as u64, (0..blocks as u64).collect()),
    };
    let projection = MessageProjection {
        source_version,
        block_count,
        block_ordinals,
    };
    projection.validate(projection.block_ordinals.len())?;
    Ok(projection)
}

/// Records, for the connection's current transaction, every version body,
/// value and text whose reference a statement removed, so
/// [`remove_unreferenced`] can delete those left without any before the
/// transaction commits. Temporary: the candidates belong to one connection's
/// transaction and roll back with it, and nothing is kept between
/// transactions or across a restart.
const UNREFERENCED: &str = "
CREATE TEMP TABLE IF NOT EXISTS unreferenced_version(
  kind TEXT NOT NULL, version_hash BLOB NOT NULL, PRIMARY KEY(kind,version_hash)) WITHOUT ROWID;
CREATE TEMP TABLE IF NOT EXISTS unreferenced_value(
  kind TEXT NOT NULL, value_pk INTEGER NOT NULL, PRIMARY KEY(kind,value_pk)) WITHOUT ROWID;
CREATE TEMP TABLE IF NOT EXISTS unreferenced_text(text_id INTEGER PRIMARY KEY);
";

/// One trigger per reference column. An UPSERT's conflict policy can override
/// a trigger's own, so each insert selects only absent keys.
fn reference_triggers() -> String {
    let mut sql = String::from(UNREFERENCED);
    let mut trigger = |name: &str, table: &str, column: &str, target: &str, kind: Option<&str>| {
        let (columns, key) = match kind {
            Some(kind) => (
                format!("'{kind}',OLD.{column}"),
                format!("kind='{kind}' AND {}=OLD.{column}", target_key(target)),
            ),
            None => (format!("OLD.{column}"), format!("text_id=OLD.{column}")),
        };
        let insert = format!(
            "INSERT INTO {target} SELECT {columns} WHERE OLD.{column} IS NOT NULL
               AND NOT EXISTS(SELECT 1 FROM {target} WHERE {key});"
        );
        sql.push_str(&format!(
            "CREATE TEMP TRIGGER IF NOT EXISTS {name}_removed AFTER DELETE ON {table}
             BEGIN {insert} END;
             CREATE TEMP TRIGGER IF NOT EXISTS {name}_changed AFTER UPDATE OF {column} ON {table}
             WHEN OLD.{column} IS NOT NEW.{column} BEGIN {insert} END;\n"
        ));
    };
    for (name, kind) in [("message", "message"), ("document", "document")] {
        trigger(
            &format!("{name}_source"),
            &format!("{name}_source"),
            "version_hash",
            "unreferenced_version",
            Some(kind),
        );
    }
    // Context a parser keeps holds an earlier message body after the shown
    // source moved on; releasing the context makes the body a candidate.
    trigger(
        "parser_context",
        "parser_context",
        "version_hash",
        "unreferenced_version",
        Some("message"),
    );
    for name in ["event", "link"] {
        trigger(
            &format!("{name}_source"),
            &format!("{name}_source"),
            "value_pk",
            "unreferenced_value",
            Some(name),
        );
    }
    for table in ["block", "message_version_block"] {
        trigger(
            &format!("{table}_payload"),
            table,
            "value_pk",
            "unreferenced_value",
            Some("block"),
        );
    }
    for table in [
        "block",
        "document_block",
        "message_version_text",
        "document_version_text",
    ] {
        trigger(table, table, "text_id", "unreferenced_text", None);
    }
    sql
}

fn target_key(target: &str) -> &'static str {
    if target == "unreferenced_value" {
        "value_pk"
    } else {
        "version_hash"
    }
}

/// Installs the reference triggers on a connection that writes the index.
pub(crate) fn track_unreferenced(conn: &Connection) -> Result<()> {
    conn.execute_batch(&reference_triggers())?;
    Ok(())
}

/// Deletes the version bodies, values and texts whose last reference this
/// transaction removed; each row-writing transaction calls it before it
/// commits. Texts go last: deleting a version removes its text references.
/// In a first build, the search tables are built from the remaining texts
/// when it ends, so a removed text has no search entry to remove.
pub(crate) fn remove_unreferenced(conn: &Connection, bulk: bool) -> Result<()> {
    for kind in VersionKind::ALL {
        let name = kind.name();
        let (table, stored_key, key, candidates) = if kind.content() {
            (
                format!("{name}_version"),
                "version_hash",
                "version_hash",
                "unreferenced_version",
            )
        } else {
            (
                format!("{name}_value"),
                "pk",
                "value_pk",
                "unreferenced_value",
            )
        };
        let context = if matches!(kind, VersionKind::Message) {
            " AND NOT EXISTS(SELECT 1 FROM parser_context c WHERE c.version_hash=t.version_hash)"
        } else {
            ""
        };
        conn.prepare_cached(&format!(
            "DELETE FROM {table} AS t WHERE {stored_key} IN (SELECT {key} FROM temp.{candidates} WHERE kind=?1)
             AND NOT EXISTS(SELECT 1 FROM {name}_source s WHERE s.{key}=t.{stored_key}){context}"
        ))?
        .execute([name])?;
    }
    conn.prepare_cached("DELETE FROM temp.unreferenced_version")?
        .execute([])?;
    conn.prepare_cached(
        "DELETE FROM block_value AS v
         WHERE pk IN (SELECT value_pk FROM temp.unreferenced_value WHERE kind='block')
         AND NOT EXISTS(SELECT 1 FROM block b WHERE b.value_pk=v.pk)
         AND NOT EXISTS(SELECT 1 FROM message_version_block b WHERE b.value_pk=v.pk)",
    )?
    .execute([])?;
    conn.prepare_cached("DELETE FROM temp.unreferenced_value")?
        .execute([])?;
    // Candidates drive the lookup. An ordinary JOIN can scan every retained
    // text and its references before checking even an empty candidate table.
    let texts = conn
        .prepare_cached(
            "SELECT t.id,t.indexed FROM temp.unreferenced_text u CROSS JOIN text t ON t.id=u.text_id
             WHERE NOT EXISTS(SELECT 1 FROM block b WHERE b.text_id=t.id)
             AND NOT EXISTS(SELECT 1 FROM document_block b WHERE b.text_id=t.id)
             AND NOT EXISTS(SELECT 1 FROM message_version_text v WHERE v.text_id=t.id)
             AND NOT EXISTS(SELECT 1 FROM document_version_text v WHERE v.text_id=t.id)",
        )?
        .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for (id, indexed) in texts {
        if bulk {
            conn.prepare_cached("DELETE FROM text WHERE id=?1")?
                .execute([id])?;
        } else {
            crate::text_rows::remove(conn, id, indexed)?;
        }
    }
    conn.prepare_cached("DELETE FROM temp.unreferenced_text")?
        .execute([])?;
    Ok(())
}
