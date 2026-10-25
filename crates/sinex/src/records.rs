//! Records Polylogue writes as Sinex materials: a look at an input that found
//! no new bytes (a part of that input's history, under its own name), and
//! each request a person or agent makes and each excision (under Polylogue's
//! requests on this host). The material's bytes are the record as JSON. A
//! note, setting, kept result, delivery or excision also becomes its typed
//! event, anchored at the record.
use crate::client::Client;
use crate::parts::{RECORDS, parser_version};
use crate::{Error, Result};
use polylogue_model::Timestamp;
use polylogue_store::primary::Record;
use polylogue_store::user_writes::{UserRequest, UserWrite};
use sinex_model::event_types::polylogue as event;
use sinex_model::{Class, InputRef, MaterialId, RecordLocation};
use sinex_protocol::{Ingress, MaterialFrame, ProducerEvent};

/// Sends `record` and its event, and waits until Sinex acknowledged both.
/// Returns the material holding the record.
pub(crate) async fn send(client: &mut Client, host: &str, record: &Record) -> Result<String> {
    let (name, input) = match record {
        Record::Observation { input, stored } => (
            format!(
                "observation:{}",
                stored
                    .observation
                    .as_ref()
                    .ok_or_else(|| Error::Invalid(
                        "an observation record has no observation".into()
                    ))?
                    .id
            ),
            InputRef {
                kind: input.origin.clone(),
                locator: input.path.clone(),
                host: input.host.clone(),
            },
        ),
        Record::Request { request, .. } => (format!("request:{}", request.id), requests(host)),
        Record::Excision(excision) => (format!("excision:{}", excision.id), requests(host)),
    };
    let bytes = serde_json::to_vec(record)?;
    let blob = format!("blake3:{}", blake3::hash(&bytes).to_hex());
    let length = bytes.len() as u64;
    client.stage(&name, &blob, length, bytes.as_slice())?;
    let material = client
        .send(Ingress::Material {
            frame: name.clone(),
            material: Box::new(MaterialFrame {
                id: MaterialId::new(),
                input,
                blob,
                length,
                parser: RECORDS.into(),
                parser_version: parser_version(),
                part_index: 0,
                previous_part: None,
                manifest: serde_json::json!({ "media_type": "application/json" }),
            }),
        })
        .await?
        .material
        .ok_or_else(|| Error::Invalid(format!("{name} registered no material")))?;
    let event = match record {
        Record::Request { request, root } => request_event(request, root.as_deref())?
            .map(|(event_type, payload)| (event_type, payload, request.at.unix_ms())),
        Record::Excision(excision) => Some((
            "polylogue.excision",
            serde_json::to_value(event::Excision {
                excision: excision.id.clone(),
                scope: excision.scope.clone(),
                reason: excision.reason.clone(),
                created_at: time(
                    Timestamp::from_unix_ms(excision.created_at)
                        .ok_or_else(|| Error::Invalid("an excision time out of range".into()))?,
                )?,
            })?,
            excision.created_at,
        )),
        Record::Observation { .. } => None,
    };
    let Some((event_type, payload, reading_order)) = event else {
        return Ok(material.to_string());
    };
    client
        .send(Ingress::Events {
            frame: format!("{name}:event"),
            events: vec![ProducerEvent {
                event_type: event_type.into(),
                payload,
                location: Some(RecordLocation {
                    material,
                    anchor: 0,
                }),
                ordinal: 0,
                parents: vec![],
                reading_order,
                class: Class::Open,
            }],
            failures: vec![],
        })
        .await?;
    Ok(material.to_string())
}

/// What Polylogue's own records are read from: its requests on this host.
fn requests(host: &str) -> InputRef {
    InputRef {
        kind: "polylogue".into(),
        locator: "requests".into(),
        host: host.to_owned(),
    }
}

pub(crate) fn time(at: Timestamp) -> Result<sinex_model::Timestamp> {
    sinex_model::Timestamp::from_unix_ns(at.unix_ns())
        .map_err(|e| Error::Invalid(format!("time {}: {e}", at.unix_ms())))
}

/// The typed event a request makes, if it makes one.
pub(crate) fn request_event(
    r: &UserRequest,
    root: Option<&str>,
) -> Result<Option<(&'static str, serde_json::Value)>> {
    let at = time(r.at)?;
    let root = || {
        root.map(str::to_owned)
            .ok_or_else(|| Error::Invalid(format!("note request {} has no family", r.id)))
    };
    Ok(Some(match &r.write {
        UserWrite::AddNote { note } => (
            "polylogue.note",
            serde_json::to_value(event::Note {
                note: r.id.clone(),
                root: root()?,
                kind: note.kind.as_str().to_owned(),
                key: note.key.clone(),
                value: note.value.clone(),
                author: note.author.to_string(),
                machine: r.machine.clone(),
                created_at: at,
                valid_until: note.valid_until.map(time).transpose()?,
                targets: note.targets.iter().map(ToString::to_string).collect(),
                relations: note
                    .revises
                    .iter()
                    .map(|previous| event::NoteRelation {
                        kind: "revises".into(),
                        other: format!("note:{previous}"),
                    })
                    .chain(
                        note.relations
                            .iter()
                            .map(|(relation, other)| event::NoteRelation {
                                kind: relation.as_str().to_owned(),
                                other: other.clone(),
                            }),
                    )
                    .collect(),
            })?,
        ),
        UserWrite::Withdraw { author, reason, .. } => {
            let root = root()?;
            (
                "polylogue.note",
                serde_json::to_value(event::Note {
                    note: r.id.clone(),
                    root: r.id.clone(),
                    kind: "withdrawal".into(),
                    key: None,
                    value: reason.clone(),
                    author: author.to_string(),
                    machine: r.machine.clone(),
                    created_at: at,
                    valid_until: None,
                    targets: vec![],
                    relations: vec![event::NoteRelation {
                        kind: "withdraws".into(),
                        other: format!("note:{root}"),
                    }],
                })?,
            )
        }
        UserWrite::Deliver(d) => (
            "polylogue.delivery",
            serde_json::to_value(event::Delivery {
                delivery: r.id.clone(),
                recipient_kind: d.recipient_kind.clone(),
                recipient_id: d.recipient_id.clone(),
                sender: d.sender.clone(),
                event: d.event.clone(),
                cwd: d.cwd.clone(),
                at,
                content_hash: format!("blake3:{}", blake3::hash(d.text.as_bytes()).to_hex()),
                bytes: d.text.len() as u64,
                refs: d.refs.clone(),
                left_out: d
                    .left_out
                    .iter()
                    .map(|l| event::LeftOut {
                        what: l.what.clone(),
                        reason: l.reason.clone(),
                    })
                    .collect(),
            })?,
        ),
        UserWrite::DeliveryEvent {
            delivery,
            what,
            detail,
        } => (
            "polylogue.delivery_outcome",
            serde_json::to_value(event::DeliveryOutcome {
                delivery: delivery.clone(),
                what: what.as_str().to_owned(),
                at,
                detail: detail.clone(),
            })?,
        ),
        UserWrite::KeepResult {
            name,
            query,
            rows_json,
            view,
        } => (
            "polylogue.kept_result",
            serde_json::to_value(event::KeptResult {
                result: r.id.clone(),
                name: name.clone(),
                query: query.clone(),
                ran_at: at,
                view: serde_json::from_str(view)?,
                rows: serde_json::from_str(rows_json)?,
            })?,
        ),
        UserWrite::SetSetting {
            key,
            value,
            author,
            machine,
        } => (
            "polylogue.setting",
            serde_json::to_value(event::Setting {
                setting: r.id.clone(),
                machine: machine.clone().unwrap_or_else(|| "*".into()),
                key: key.clone(),
                value: value.clone(),
                author: author.to_string(),
                set_at: at,
            })?,
        ),
        // The browser's chat-action queue is Polylogue's working state: Sinex
        // keeps the requests, with no event of their own.
        UserWrite::ChatEnqueue(_)
        | UserWrite::ChatClaim { .. }
        | UserWrite::ChatReport { .. }
        | UserWrite::ChatDecide { .. } => return Ok(None),
    }))
}
