//! Explicit scans. Coverage and findings belong only to the rebuildable index.
use crate::{Reader, Result, StoreError};
use polylogue_model::masking::{SCANNER_VERSION, SecretScanner};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub session: String,
    pub target: String,
    pub block: Option<i64>,
    /// Decoded field: arrays use indices, objects use sorted member numbers
    /// followed by key or value. Offsets count UTF-8 bytes in that string.
    pub field: String,
    pub pattern: String,
    pub start: usize,
    pub end: usize,
    pub byte_length: usize,
    pub fingerprint: String,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SessionScan {
    pub found: bool,
    pub blocks_scanned: u64,
    pub events_scanned: u64,
    pub findings: Vec<Finding>,
}

/// Includes the application's target selection and the shared detector rules.
pub fn version() -> String {
    format!("polylogue-1:{SCANNER_VERSION}")
}

pub fn validate(
    session: Option<&str>,
    all: bool,
    status: bool,
    origin: Option<&str>,
    limit: Option<u64>,
) -> Result<()> {
    if usize::from(session.is_some()) + usize::from(all) + usize::from(status) != 1 {
        return Err(StoreError::Invalid(
            "choose exactly one of session, all or status".into(),
        ));
    }
    if (session.is_some() && origin.is_some()) || (limit.is_some() && !all) || limit == Some(0) {
        return Err(StoreError::Invalid(
            "origin requires all or status; a positive limit requires all".into(),
        ));
    }
    Ok(())
}

impl Reader<'_> {
    /// One pending session, in stable ID order. No captured content is read.
    pub fn pending_secret_scan(
        &self,
        origin: Option<&str>,
        after: Option<&str>,
    ) -> Result<Option<String>> {
        Ok(self
            .conn()?
            .query_row(
                "SELECT s.id FROM session s LEFT JOIN secret_scan c ON c.session_pk=s.pk
             WHERE (?1 IS NULL OR s.origin=?1) AND
             (c.session_pk IS NULL OR c.version<>?2 OR c.from_changed<>s.changed)
             AND (?3 IS NULL OR s.id > ?3)
             ORDER BY s.id LIMIT 1",
                params![origin, version(), after],
                |r| r.get(0),
            )
            .optional()?)
    }

    pub fn secret_scan_pending_count(&self, origin: Option<&str>) -> Result<u64> {
        Ok(self.conn()?.query_row(
            "SELECT count(*) FROM session s LEFT JOIN secret_scan c ON c.session_pk=s.pk
             WHERE (?1 IS NULL OR s.origin=?1) AND
             (c.session_pk IS NULL OR c.version<>?2 OR c.from_changed<>s.changed)",
            params![origin, version()],
            |r| r.get::<_, i64>(0).map(|count| count as u64),
        )?)
    }
}

/// Each session commits independently. An interruption leaves the current
/// session pending, while earlier sessions remain covered.
pub(crate) fn scan(index: &mut Connection, session: &str) -> Result<SessionScan> {
    let tx = index.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let identity: Option<(i64, i64)> = tx
        .query_row(
            "SELECT pk, changed FROM session WHERE id=?1",
            [session],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    let Some((pk, changed)) = identity else {
        return Ok(SessionScan::default());
    };
    let mut output = SessionScan {
        found: true,
        ..Default::default()
    };
    let mut scanner = SecretScanner::new();
    let mut offset = 0;
    let mut statement = tx.prepare(
        "SELECT m.id, b.idx, b.body, t.body, b.data FROM message m
         JOIN block b ON b.message_pk=m.pk LEFT JOIN text t ON t.id=b.text_id
         WHERE m.session_pk=?1
         ORDER BY m.order_input_id, m.seq, m.record_position, m.id, b.idx",
    )?;
    let mut rows = statement.query([pk])?;
    while let Some(row) = rows.next()? {
        let message = format!("message:{}", row.get::<_, String>(0)?);
        let block_index = row.get::<_, Option<i64>>(1)?;
        let mut block = crate::encoding::unpack_block(&row.get::<_, Vec<u8>>(2)?)?;
        if let Some(text) = row.get::<_, Option<Vec<u8>>>(3)? {
            crate::encoding::restore_text(&mut block, crate::encoding::unpack_text(&text)?);
        }
        if let Some(data) = row.get::<_, Option<Vec<u8>>>(4)? {
            crate::encoding::restore_data(&mut block, &data)?;
        }
        let primary = match &block {
            polylogue_model::Block::Text { .. } | polylogue_model::Block::Thinking { .. } => {
                Some("text")
            }
            polylogue_model::Block::ToolCall {
                command: Some(_), ..
            } => Some("command"),
            polylogue_model::Block::ToolCall { path: Some(_), .. } => Some("path"),
            polylogue_model::Block::ToolCall {
                input: Value::String(_),
                ..
            } => Some("input"),
            polylogue_model::Block::ToolResult { .. } => Some("output"),
            _ => None,
        };
        let Value::Object(mut fields) = serde_json::to_value(block)? else {
            unreachable!("block is an object")
        };
        fields.remove("type");
        if let Some(primary) = primary
            && let Some(Value::String(text)) = fields.remove(primary)
        {
            scanner.push(&text)?;
            append_findings(
                &mut output,
                (session, &message, block_index),
                primary,
                scanner.block_boundary()?,
                offset,
            );
            offset += text.len();
        }
        for (name, value) in &fields {
            scan_value(&mut output, session, &message, block_index, name, value)?;
        }
        output.blocks_scanned += 1;
    }
    let _ = scanner.finish()?;
    drop(rows);
    drop(statement);
    let mut events =
        tx.prepare("SELECT id, data FROM event WHERE session_pk=?1 ORDER BY seq, id")?;
    let mut rows = events.query([pk])?;
    while let Some(row) = rows.next()? {
        let target = format!("event:{}", row.get::<_, String>(0)?);
        let data: Value = crate::encoding::unpack_json(&row.get::<_, Vec<u8>>(1)?)?;
        scan_value(&mut output, session, &target, None, "data", &data)?;
        output.events_scanned += 1;
    }
    drop(rows);
    drop(events);
    tx.execute("INSERT INTO secret_scan(session_pk, version, from_changed, output) VALUES (?1,?2,?3,?4)
        ON CONFLICT(session_pk) DO UPDATE SET version=excluded.version, from_changed=excluded.from_changed, output=excluded.output",
        params![pk, version(), changed, serde_json::to_string(&output)?])?;
    tx.commit()?;
    Ok(output)
}

fn scan_value(
    output: &mut SessionScan,
    session: &str,
    target: &str,
    block: Option<i64>,
    name: &str,
    value: &Value,
) -> Result<()> {
    let mut strings = vec![(name.to_owned(), value)];
    while let Some((field, value)) = strings.pop() {
        match value {
            Value::String(text) => scan_string(output, session, target, block, &field, text)?,
            Value::Array(values) => strings.extend(
                values
                    .iter()
                    .enumerate()
                    .rev()
                    .map(|(i, v)| (format!("{field}/{i}"), v)),
            ),
            Value::Object(values) => {
                // Member numbers keep credentials in object keys out of the result path.
                for (i, (key, value)) in values.iter().enumerate().rev() {
                    scan_string(
                        output,
                        session,
                        target,
                        block,
                        &format!("{field}/{i}/key"),
                        key,
                    )?;
                    strings.push((format!("{field}/{i}/value"), value));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn scan_string(
    output: &mut SessionScan,
    session: &str,
    message: &str,
    block: Option<i64>,
    field: &str,
    text: &str,
) -> Result<()> {
    let mut scanner = SecretScanner::new();
    scanner.push(text)?;
    append_findings(
        output,
        (session, message, block),
        field,
        scanner.finish()?,
        0,
    );
    Ok(())
}

fn append_findings(
    output: &mut SessionScan,
    target: (&str, &str, Option<i64>),
    field: &str,
    findings: Vec<polylogue_model::masking::SecretFinding>,
    offset: usize,
) {
    let (session, message, block) = target;
    for finding in findings {
        output.findings.push(Finding {
            session: session.into(),
            target: message.into(),
            block,
            field: field.into(),
            pattern: finding.pattern.into(),
            start: finding.start - offset,
            end: finding.end - offset,
            byte_length: finding.end - finding.start,
            fingerprint: finding.fingerprint,
        });
    }
}

#[cfg(test)]
mod tests;
