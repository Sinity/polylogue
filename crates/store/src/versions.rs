//! Packed readings and their shared text. Source rows choose what is shown.
use crate::{Result, encoding, first_build::FirstBuild};
use polylogue_model::Block;
use rusqlite::{Connection, OptionalExtension, Row, params};
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

    fn order(self) -> &'static str {
        match self {
            Self::Message => {
                "s.main DESC,s.complete DESC,s.arrival_rank,s.version_hash,i.id,s.chain DESC"
            }
            Self::Document => {
                "s.source_time IS NULL,s.source_time DESC,s.arrival_rank,s.version_hash DESC,i.id,s.chain DESC"
            }
            Self::Event | Self::Link => "s.arrival_rank,i.id,s.chain DESC",
        }
    }

    /// Selection reads the stored source rank; codecs only read the selected value.
    pub(crate) fn select<T>(
        self,
        conn: &Connection,
        pk: i64,
        columns: &str,
        read: impl FnOnce(&Row<'_>) -> rusqlite::Result<T>,
    ) -> Result<Option<T>> {
        let name = self.name();
        let (join, newest) = match self {
            Self::Message => (
                "".to_owned(),
                "AND NOT EXISTS(SELECT 1 FROM message_source n WHERE n.message_pk=s.message_pk AND n.input_pk=s.input_pk AND n.chain>s.chain)",
            ),
            Self::Document => ("".to_owned(), ""),
            Self::Event | Self::Link => (format!("JOIN {name}_value v ON v.pk=s.value_pk"), ""),
        };
        Ok(conn
            .prepare_cached(&format!(
                "SELECT {columns} FROM {name}_source s JOIN input_ref i ON i.pk=s.input_pk {join}
             WHERE s.{name}_pk=?1 {newest} ORDER BY {} LIMIT 1",
                self.order()
            ))?
            .query_row([pk], read)
            .optional()?)
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

pub(crate) fn store(
    conn: &Connection,
    kind: VersionKind,
    hash: &[u8],
    body: &[u8],
    blocks: &[Block],
    packed: &HashMap<[u8; 32], Vec<u8>>,
    first_build: Option<&FirstBuild>,
) -> Result<()> {
    let kind = kind.name();
    let inserted = conn
        .prepare_cached(&format!(
            "INSERT OR IGNORE INTO {kind}_version(version_hash,body) VALUES (?1,?2)"
        ))?
        .execute(params![hash, body])?;
    if inserted == 0 {
        return Ok(());
    }
    for (idx, block) in blocks.iter().enumerate() {
        if let Some(text) = encoding::block_text(block) {
            let id = crate::text_rows::intern(conn, text, 0, Some(packed), first_build)?;
            conn.prepare_cached(&format!(
                "INSERT INTO {kind}_version_text(version_hash,idx,text_id) VALUES (?1,?2,?3)"
            ))?
            .execute(params![hash, idx as i64, id])?;
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
    let mut value: serde_json::Value = encoding::unpack_json(&body)?;
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
