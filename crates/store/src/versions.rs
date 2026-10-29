//! Packed readings and their shared text. Source rows choose what is shown.
use crate::{Result, encoding, first_build::FirstBuild};
use polylogue_model::{BlobHash, Block, Message, bundle::MessageProjection};
use rusqlite::{Connection, OptionalExtension, params};
use serde::de::DeserializeOwned;
use std::collections::HashMap;

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

    pub(crate) fn retract(self, conn: &Connection, input: i64, chain: i64) -> Result<Vec<i64>> {
        let name = self.name();
        let touched = conn
            .prepare_cached(&format!(
                "SELECT {name}_pk FROM {name}_source WHERE input_pk=?1 AND chain=?2"
            ))?
            .query_map(params![input, chain], |row| row.get(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        conn.prepare_cached(&format!(
            "DELETE FROM {name}_source WHERE input_pk=?1 AND chain=?2"
        ))?
        .execute(params![input, chain])?;
        Ok(touched)
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
) -> Result<()> {
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
    if inserted == 0 {
        return Ok(());
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
    Ok(())
}

pub(crate) fn store_message(
    conn: &Connection,
    message: &Message,
    prepared: &encoding::PreparedMessage,
    text_rows: TextRows<'_>,
    replace: bool,
) -> Result<()> {
    prepared.projection.validate(message.blocks.len())?;
    if prepared.hash != prepared.projection.source_version.0 {
        return Err(polylogue_model::bundle::BundleError::SourceVersionMismatch.into());
    }
    store_value(
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
    let layout = message_layout(&prepared.projection)?;
    conn.execute(
        "INSERT INTO message_projection(version_hash,layout) VALUES (?1,?2)
         ON CONFLICT(version_hash) DO UPDATE SET layout=excluded.layout WHERE layout IS NOT excluded.layout",
        params![prepared.hash.as_slice(), layout],
    )?;
    Ok(())
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

/// Recheck only readings whose source references changed, using their hash indexes.
pub(crate) fn remove_readings(conn: &Connection) -> Result<()> {
    for kind in VersionKind::ALL {
        let name = kind.name();
        let (table, key, changed) = if kind.content() {
            (format!("{name}_version"), "version_hash", "changed_version")
        } else {
            (format!("{name}_value"), "value_pk", "changed_value")
        };
        let stored_key = if kind.content() { "version_hash" } else { "pk" };
        conn.execute(&format!("DELETE FROM {table} WHERE {stored_key} IN (SELECT {key} FROM {changed} WHERE kind=?1)
            AND NOT EXISTS(SELECT 1 FROM {name}_source s WHERE s.{key}={table}.{stored_key})"), [name])?;
        conn.execute(&format!("DELETE FROM {changed} WHERE kind=?1"), [name])?;
    }
    Ok(())
}

pub(crate) fn remove_unused(conn: &Connection, bulk: bool) -> Result<()> {
    remove_readings(conn)?;
    let mut statement = conn.prepare(
        "SELECT t.id,t.indexed FROM changed_text u JOIN text t ON t.id=u.text_id
         WHERE NOT EXISTS(SELECT 1 FROM block b WHERE b.text_id=t.id)
         AND NOT EXISTS(SELECT 1 FROM document_block b WHERE b.text_id=t.id)
         AND NOT EXISTS(SELECT 1 FROM message_version_text v WHERE v.text_id=t.id)
         AND NOT EXISTS(SELECT 1 FROM document_version_text v WHERE v.text_id=t.id)",
    )?;
    let texts = statement
        .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(statement);
    for (id, indexed) in texts {
        if bulk {
            conn.execute("DELETE FROM text WHERE id=?1", [id])?;
        } else {
            crate::text_rows::remove(conn, id, indexed)?;
        }
    }
    conn.execute("DELETE FROM changed_text", [])?;
    Ok(())
}
