//! Parsed AI sessions as Sinex events. After each derive step the loop sends
//! every session whose rows changed since the last send: the session, its own
//! messages, its tool calls and its turns, read from `index.db` as Polylogue
//! shows them. Each event is keyed by Polylogue's ID and anchored at the
//! record it came from, in the material holding that record's bytes.
//!
//! Events go in frames of [`CHUNK`], named by their content, so a chunk Sinex
//! holds already is not sent again: a growing session sends only its changed
//! tail, and a changed parser re-sends what it reads differently, which
//! replaces the older readings. Where the last send stopped is kept beside
//! the index (`sinex-sent.json`): the index file and its write counter.
use crate::parts::content_name;
use crate::records::time;
use crate::{Error, Result, Sinex};
use polylogue_model::{AttachmentDirection, Block, MessageId, SessionId, Timestamp};
use polylogue_store::{ReadPool, Reader, SqlValue};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sinex_model::event_types::polylogue as event;
use sinex_model::{Class, MaterialId, RecordLocation};
use sinex_protocol::{Ingress, ProducerEvent};
use std::collections::HashMap;

/// Events per frame.
pub(crate) const CHUNK: usize = 200;

/// Where the last send stood.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Sent {
    /// The index file's device and inode: a rebuilt index sends everything.
    file: (u64, u64),
    counter: i64,
}

pub(crate) fn send(sinex: &Sinex, reads: &ReadPool) -> Result<()> {
    let reader = reads.reader()?;
    let file = reader.index_identity()?;
    let counter = reader.change_position()?.index;
    let sent: Sent = match std::fs::read(&sinex.sent) {
        Ok(bytes) => serde_json::from_slice(&bytes).unwrap_or_default(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Sent::default(),
        Err(e) => return Err(e.into()),
    };
    let after = if sent.file == file { sent.counter } else { 0 };
    let changed = reader.query(
        "SELECT id FROM session WHERE changed > ?1 ORDER BY id",
        &[SqlValue::Int(after)],
    )?;
    for row in changed {
        let id = row["id"]
            .as_str()
            .and_then(|id| SessionId::parse(id).ok())
            .ok_or_else(|| Error::Invalid(format!("session row {row}")))?;
        let events = session_events(&reader, &id)?;
        send_chunks(sinex, events)?;
    }
    let done = Sent { file, counter };
    let path = sinex.sent.with_extension("json.new");
    std::fs::write(&path, serde_json::to_vec(&done)?)?;
    std::fs::rename(&path, &sinex.sent)?;
    Ok(())
}

/// Sends the chunks of one session's events Sinex does not hold yet.
fn send_chunks(sinex: &Sinex, events: Vec<ProducerEvent>) -> Result<()> {
    let chunks: Vec<(String, Vec<ProducerEvent>)> = events
        .chunks(CHUNK)
        .map(|chunk| {
            let mut hash = blake3::Hasher::new();
            for e in chunk {
                hash.update(e.event_type.as_bytes());
                hash.update(&[0]);
                hash.update(e.payload.to_string().as_bytes());
                hash.update(&[0]);
                if let Some(at) = e.location {
                    hash.update(at.material.0.as_bytes());
                    hash.update(&at.anchor.to_le_bytes());
                }
            }
            (
                format!("events:{}", hash.finalize().to_hex()),
                chunk.to_vec(),
            )
        })
        .collect();
    if chunks.is_empty() {
        return Ok(());
    }
    let reading_order = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));
    sinex.with(async |client| {
        let names: Vec<String> = chunks.iter().map(|(name, _)| name.clone()).collect();
        let held = client.have(&names).await?;
        for (name, mut events) in chunks {
            if let Some(acknowledgment) = held.get(&name) {
                crate::client::checked(acknowledgment.clone())?;
                continue;
            }
            for e in &mut events {
                e.reading_order = reading_order;
            }
            client
                .send(Ingress::Events {
                    frame: name,
                    events,
                    failures: vec![],
                })
                .await?;
        }
        Ok(())
    })
}

/// Where a record starting at `start` of `input`'s chain `chain` is: the
/// material holding that byte, and the record's position in it.
struct Anchors<'r, 'a> {
    reader: &'r Reader<'a>,
    found: HashMap<(String, i64, i64), Option<RecordLocation>>,
}

impl Anchors<'_, '_> {
    fn at(&mut self, input: &str, chain: i64, start: i64) -> Result<Option<RecordLocation>> {
        let key = (input.to_owned(), chain, start);
        if let Some(found) = self.found.get(&key) {
            return Ok(*found);
        }
        let rows = self.reader.query(
            "SELECT id, offset FROM a.material WHERE input = ?1 AND chain = ?2 AND offset <= ?3
             ORDER BY offset DESC LIMIT 1",
            &[
                SqlValue::Text(input.to_owned()),
                SqlValue::Int(chain),
                SqlValue::Int(start),
            ],
        )?;
        let location = match rows.first() {
            Some(row) => {
                let material: MaterialId = row["id"]
                    .as_str()
                    .and_then(|id| id.parse().ok())
                    .map(MaterialId)
                    .ok_or_else(|| Error::Invalid(format!("material row {row}")))?;
                let offset = row["offset"].as_i64().unwrap_or(0);
                Some(RecordLocation {
                    material,
                    anchor: u64::try_from(start - offset)
                        .map_err(|_| Error::Invalid("a record before its part".into()))?,
                })
            }
            None => None,
        };
        self.found.insert(key, location);
        Ok(location)
    }

    /// The location of a message's shown version.
    fn message(&mut self, row: &Value) -> Result<Option<RecordLocation>> {
        match (
            row["input"].as_str(),
            row["chain"].as_i64(),
            row["start"].as_i64(),
        ) {
            (Some(input), Some(chain), Some(start)) => self.at(input, chain, start),
            _ => Ok(None),
        }
    }
}

fn ms(value: &Value) -> Result<Option<sinex_model::Timestamp>> {
    value
        .as_i64()
        .and_then(Timestamp::from_unix_ms)
        .map(time)
        .transpose()
}

fn text(value: &Value) -> Option<String> {
    value.as_str().map(str::to_owned)
}

fn producer_event(
    event_type: &str,
    payload: impl Serialize,
    location: RecordLocation,
) -> Result<ProducerEvent> {
    Ok(ProducerEvent {
        event_type: event_type.into(),
        payload: serde_json::to_value(payload)?,
        location: Some(location),
        ordinal: 0,
        parents: vec![],
        reading_order: 0,
        class: Class::Open,
    })
}

/// The events of one session as the index holds it now. Anything without a
/// record in Sinex (a session only a hook has named so far) is left out.
pub(crate) fn session_events(reader: &Reader<'_>, id: &SessionId) -> Result<Vec<ProducerEvent>> {
    let Some(export) = reader.session_export(id)? else {
        return Ok(Vec::new());
    };
    let session = reader.query(
        "SELECT s.pk, s.origin, s.kind, coalesce(s.title, s.derived_title) AS title, s.cwd_json,
                s.repo, s.branch, s.harness_version, s.mode, s.started_at, s.last_activity_at,
                (SELECT o.machine FROM a.observation o
                 WHERE o.input = (SELECT i.id FROM message m JOIN input_ref i ON i.pk = m.input_pk
                                  WHERE m.session_pk = s.pk LIMIT 1)
                 ORDER BY o.rowid DESC LIMIT 1) AS machine
         FROM session s WHERE s.id = ?1",
        &[SqlValue::Text(id.to_string())],
    )?;
    let Some(session) = session.first() else {
        return Ok(Vec::new());
    };
    let pk = session["pk"].as_i64().unwrap_or_default();
    let mut anchors = Anchors {
        reader,
        found: HashMap::new(),
    };
    let rows = reader.query(
        "SELECT m.pk, m.id, m.written_by, m.cost_nusd, i.id AS input, m.source_chain AS chain,
                m.source_start AS start
         FROM message m JOIN input_ref i ON i.pk = m.input_pk
         WHERE m.session_pk = ?1",
        &[SqlValue::Int(pk)],
    )?;
    let by_id: HashMap<String, &Value> = rows
        .iter()
        .filter_map(|row| Some((row["id"].as_str()?.to_owned(), row)))
        .collect();
    let copies = reader.query(
        "SELECT m.id, i.id AS input, s.chain, s.source_start AS start
         FROM message_source s JOIN message m ON m.pk = s.message_pk
         JOIN input_ref i ON i.pk = s.input_pk
         WHERE m.session_pk = ?1 AND s.preferred = 0
         ORDER BY m.id, i.id, s.chain",
        &[SqlValue::Int(pk)],
    )?;
    let mut copies_of: HashMap<String, Vec<event::RecordRef>> = HashMap::new();
    for copy in &copies {
        if let (Some(message), Some(location)) = (copy["id"].as_str(), anchors.message(copy)?) {
            copies_of
                .entry(message.to_owned())
                .or_default()
                .push(event::RecordRef {
                    material: location.material,
                    anchor: location.anchor,
                });
        }
    }

    let mut messages = Vec::new();
    let mut first: Option<RecordLocation> = None;
    for m in &export.messages {
        let Some(row) = by_id.get(&m.id.to_string()) else {
            continue;
        };
        let Some(location) = anchors.message(row)? else {
            continue;
        };
        first.get_or_insert(location);
        messages.push(producer_event(
            "polylogue.message",
            message_payload(
                m,
                row,
                copies_of.remove(&m.id.to_string()).unwrap_or_default(),
            )?,
            location,
        )?);
    }
    let Some(session_location) = first else {
        return Ok(Vec::new());
    };

    let mut events = vec![producer_event(
        "polylogue.session",
        event::Session {
            session: id.to_string(),
            origin: text(&session["origin"]).unwrap_or_default(),
            kind: text(&session["kind"]).unwrap_or_default(),
            title: text(&session["title"]),
            cwd: session["cwd_json"]
                .as_str()
                .and_then(|json| serde_json::from_str(json).ok())
                .unwrap_or_default(),
            repo: text(&session["repo"]),
            branch: text(&session["branch"]),
            tool_version: text(&session["harness_version"]),
            mode: text(&session["mode"]),
            machine: text(&session["machine"]),
            started: ms(&session["started_at"])?,
            last_activity: ms(&session["last_activity_at"])?,
            links: export
                .links
                .iter()
                .map(|link| event::SessionLink {
                    kind: match link.kind {
                        polylogue_model::LinkKind::Fork => "fork",
                        polylogue_model::LinkKind::Spawn => "spawn",
                        polylogue_model::LinkKind::Continuation => "continuation",
                        polylogue_model::LinkKind::Copy => "copy",
                    }
                    .into(),
                    parent: link.parent.to_string(),
                    child: link.child.to_string(),
                    at: link.at.as_ref().map(MessageId::to_string),
                    call_id: link.call_id.clone(),
                })
                .collect(),
        },
        session_location,
    )?];
    events.extend(messages);
    events.extend(tool_calls(reader, &mut anchors, id, pk)?);
    events.extend(turns(reader, &mut anchors, id, pk)?);
    Ok(events)
}

fn message_payload(
    m: &polylogue_model::Message,
    row: &Value,
    copies: Vec<event::RecordRef>,
) -> Result<event::Message> {
    let text: Vec<&str> = m
        .blocks
        .iter()
        .filter_map(|b| match b {
            Block::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    let attachments = m
        .blocks
        .iter()
        .filter_map(|b| match b {
            Block::Image(a) | Block::File(a) => Some(event::Attachment {
                name: a.name.clone(),
                media_type: a.media_type.clone(),
                size: a.size,
                blob: a.blob.as_ref().map(content_name),
                direction: match a.direction {
                    AttachmentDirection::Input => "input",
                    AttachmentDirection::Output => "output",
                    AttachmentDirection::Unknown => "unknown",
                }
                .into(),
            }),
            _ => None,
        })
        .collect();
    Ok(event::Message {
        message: m.id.to_string(),
        session: m.session.to_string(),
        parent: m.parent.as_ref().map(MessageId::to_string),
        role: m.role.as_str().to_owned(),
        written_by: row["written_by"].as_str().unwrap_or("unknown").to_owned(),
        prompt_kind: m.prompt_kind.map(|k| k.as_str().to_owned()),
        at: m.at.map(time).transpose()?,
        model: m.model.clone(),
        text: text.join("\n\n"),
        usage: event::Usage {
            input: m.usage.input,
            cached_input: m.usage.cached_input,
            cache_write: m.usage.cache_write,
            output: m.usage.output,
            reasoning: m.usage.reasoning,
        },
        cost_nusd: row["cost_nusd"].as_i64(),
        stop_reason: m.stop_reason.clone(),
        duration_ms: m.duration_ms,
        attachments,
        copies,
    })
}

fn tool_calls(
    reader: &Reader<'_>,
    anchors: &mut Anchors<'_, '_>,
    session: &SessionId,
    pk: i64,
) -> Result<Vec<ProducerEvent>> {
    let rows = reader.query(
        "SELECT t.call_id, t.tool, t.action, t.command, t.path, t.outcome, t.outcome_reason,
                t.exit_code, t.at, cb.idx,
                cm.id AS message, ci.id AS input, cm.source_chain AS chain, cm.source_start AS start,
                ri.id AS result_input, rm.source_chain AS result_chain,
                rm.source_start AS result_start
         FROM tool_call t
         JOIN block cb ON cb.pk = t.call_block_pk
         JOIN message cm ON cm.pk = cb.message_pk
         JOIN input_ref ci ON ci.pk = cm.input_pk
         LEFT JOIN block rb ON rb.pk = t.result_block_pk
         LEFT JOIN message rm ON rm.pk = rb.message_pk
         LEFT JOIN input_ref ri ON ri.pk = rm.input_pk
         WHERE t.session_pk = ?1
         ORDER BY cm.order_input_id, cm.seq, cm.record_position, cm.id, cb.idx",
        &[SqlValue::Int(pk)],
    )?;
    let mut out = Vec::new();
    for row in rows {
        let Some(location) = anchors.message(&row)? else {
            continue;
        };
        let message = text(&row["message"]).unwrap_or_default();
        let call = match row["call_id"].as_str() {
            Some(call) => call.to_owned(),
            None => format!("{message}/{}", row["idx"]),
        };
        let result = match (
            row["result_input"].as_str(),
            row["result_chain"].as_i64(),
            row["result_start"].as_i64(),
        ) {
            (Some(input), Some(chain), Some(start)) => {
                anchors.at(input, chain, start)?.map(|l| event::RecordRef {
                    material: l.material,
                    anchor: l.anchor,
                })
            }
            _ => None,
        };
        out.push(producer_event(
            "polylogue.tool_call",
            event::ToolCall {
                call: format!("tool:{session}#{call}"),
                session: session.to_string(),
                message,
                tool: text(&row["tool"]).unwrap_or_default(),
                action: text(&row["action"]).unwrap_or_else(|| "other".into()),
                command: text(&row["command"]),
                path: text(&row["path"]),
                outcome: text(&row["outcome"]).unwrap_or_else(|| "unknown".into()),
                outcome_reason: text(&row["outcome_reason"]),
                exit_code: row["exit_code"].as_i64(),
                at: ms(&row["at"])?,
                result,
            },
            location,
        )?);
    }
    Ok(out)
}

fn turns(
    reader: &Reader<'_>,
    anchors: &mut Anchors<'_, '_>,
    session: &SessionId,
    pk: i64,
) -> Result<Vec<ProducerEvent>> {
    let rows = reader.query(
        &format!(
            "SELECT t.id, t.start, t.\"end\", t.prompts, t.answer, t.slash_command, t.tool_calls,
                    t.errors, t.input_tokens, t.output_tokens, t.cost_nusd, t.abandoned,
                    i.id AS input, m.source_chain AS chain, m.source_start AS start
             FROM {} t JOIN message m ON m.id = t.id JOIN input_ref i ON i.pk = m.input_pk
             WHERE t.session_pk = ?1 ORDER BY t.start, t.id",
            polylogue_query::units::TURNS
        ),
        &[SqlValue::Int(pk)],
    )?;
    let mut out = Vec::new();
    for row in rows {
        let Some(location) = anchors.message(&row)? else {
            continue;
        };
        out.push(producer_event(
            "polylogue.turn",
            event::Turn {
                turn: text(&row["id"]).unwrap_or_default(),
                session: session.to_string(),
                start: ms(&row["start"])?,
                end: ms(&row["end"])?,
                prompts: row["prompts"]
                    .as_str()
                    .and_then(|json| serde_json::from_str(json).ok())
                    .unwrap_or_default(),
                answer: text(&row["answer"]),
                slash_command: row["slash_command"].as_i64() == Some(1),
                tool_calls: row["tool_calls"].as_u64().unwrap_or(0),
                errors: row["errors"].as_u64().unwrap_or(0),
                input_tokens: row["input_tokens"].as_u64(),
                output_tokens: row["output_tokens"].as_u64(),
                cost_nusd: row["cost_nusd"].as_i64(),
                abandoned: row["abandoned"].as_i64() == Some(1),
            },
            location,
        )?);
    }
    Ok(out)
}
