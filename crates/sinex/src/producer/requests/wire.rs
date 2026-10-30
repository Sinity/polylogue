//! The frozen Source3 envelope, exercised against the actual predecessor codec.
//! Only an owner-verified retained material can reach this private continuation.
use crate::{Error, Result};
use polylogue_model::{authored::ImportAttribution, historical::QueuedPublicationCustody};
use polylogue_store::primary::SourceLocation;
use serde::Deserialize;
use serde_json::{Value, json, value::RawValue};
use sinex_model::{Class, KindRef, MaterialId, RecordLocation};
use sinex_protocol::{Ingress, MaterialFrame, ProducerEvent};
use std::io::Read;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Prepared<'a> {
    #[serde(borrow)]
    request: &'a RawValue,
    #[serde(borrow)]
    outcome: &'a RawValue,
    #[serde(borrow)]
    effect: &'a RawValue,
}
#[derive(Deserialize)]
struct RequestMeta {
    id: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Write {
    product: Box<RawValue>,
    location: Option<SourceLocation>,
    parents: Vec<String>,
    machine: Option<String>,
    output_index: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Restore<'a> {
    product: &'a str,
    #[serde(borrow)]
    prepared: &'a RawValue,
    claims: Vec<ImportAttribution>,
    current: ImportAttribution,
}

pub(super) fn build(
    custody: &QueuedPublicationCustody,
    material: &MaterialFrame,
    body: impl Read,
) -> Result<Ingress> {
    crate::client::confirmed::source3::validate_source(custody)?;
    if material.id.to_string() != custody.material
        || material.source.name != custody.source
        || material.source.version != custody.source_version
        || material.parser != custody.parser
        || material.parser_version != custody.parser_version
        || material.blob != format!("blake3:{}", custody.blob)
        || material.length != custody.length
        || material.part_index != 0
        || material.previous_part.is_some()
        || material.manifest.role != sinex_model::MaterialRole::Records
        || material.manifest.media_type.as_deref() != Some("application/json")
        || !material.manifest.details.is_empty()
    {
        return Err(Error::Invalid(
            "retained envelope differs from its verified material binding".into(),
        ));
    }
    let events = if custody.parser == "polylogue-request" {
        if material.input.kind != "polylogue" || material.input.locator != "requests" {
            return Err(Error::Invalid(
                "retained request has no original producer encoding metadata".into(),
            ));
        }
        let raw: Box<RawValue> = crate::spool::decode_json(body)?;
        let (accepted, id) = conserve(custody, &raw)?;
        let fields: Prepared<'_> = crate::client::confirmed::source3::parse(raw.get().as_bytes())?;
        let effect: Value = crate::spool::decode_json(fields.effect.get().as_bytes())?;
        if effect
            .get("chats")
            .and_then(Value::as_array)
            .is_some_and(|chats| {
                chats.iter().any(|chat| {
                    chat.pointer("/action/attachments")
                        .and_then(Value::as_array)
                        .is_some_and(|attachments| !attachments.is_empty())
                })
            })
        {
            return Err(Error::Refused {
                name: custody.frame.clone(),
                reason: "attachment continuation requires its actual retained original event group"
                    .into(),
            });
        }
        if custody.frame != format!("request:{id}") {
            return Err(Error::Invalid(
                "retained request differs from its original operation key".into(),
            ));
        }
        let kind = polylogue_store::assertions::conversion::retained_source3_kind(&accepted)?;
        let mut payload =
            polylogue_store::assertions::conversion::retained_source3_payload(&accepted)?;
        let natural = json!([id]);
        let record = json!({"product":"request", "request":fields.request, "outcome":fields.outcome, "effect":fields.effect});
        finish(&mut payload, kind, natural, record)?;
        vec![event(
            kind,
            payload,
            Some(RecordLocation {
                material: material.id,
                anchor: 0,
            }),
            Vec::new(),
            0,
        )?]
    } else {
        if material.input.kind != "polylogue-observation"
            || custody.frame.strip_suffix(":observation") != Some(material.input.locator.as_str())
        {
            return Err(Error::Invalid(
                "retained restore has no original producer operation metadata".into(),
            ));
        }
        // The original ProductWrite array retains output indices, locations,
        // parents and importing metadata. None come from the current operation.
        let mut events = Vec::new();
        crate::client::confirmed::source3::visit_writes(body, |raw| {
            let write: Write = crate::spool::decode_json(raw.get().as_bytes())?;
            let restore: Restore<'_> =
                crate::client::confirmed::source3::parse(write.product.get().as_bytes())?;
            if restore.product != "restored_request" {
                return Err(Error::Invalid(
                    "retained restore contains another product".into(),
                ));
            }
            let (_accepted, id) = conserve(custody, restore.prepared)?;
            let mut payload =
                json!({"request":id, "at":restore.current.at, "machine":restore.current.machine});
            let natural = json!([
                id,
                restore.claims,
                restore.current.input,
                restore.current.chain,
                restore.current.machine,
                restore.current.actor
            ]);
            finish(
                &mut payload,
                "polylogue.request_restore",
                natural,
                crate::client::confirmed::source3::parse(write.product.get().as_bytes())?,
            )?;
            let location =
                match write.location {
                    Some(location) => Some(RecordLocation {
                        material: MaterialId(location.material.parse().map_err(|error| {
                            Error::Invalid(format!("retained location: {error}"))
                        })?),
                        anchor: location.anchor,
                    }),
                    None if write.parents.is_empty() => Some(RecordLocation {
                        material: material.id,
                        anchor: 0,
                    }),
                    None => None,
                };
            let parents = write
                .parents
                .into_iter()
                .map(|parent| {
                    parent
                        .parse()
                        .map(sinex_model::EventId)
                        .map_err(|error| Error::Invalid(format!("retained parent: {error}")))
                })
                .collect::<Result<_>>()?;
            // Request summaries never use ProductWrite.machine. It remains in
            // the original material body and is not a current host claim.
            let _original_machine = write.machine;
            events.push(event(
                "polylogue.request_restore",
                payload,
                location,
                parents,
                write.output_index,
            )?);
            Ok(())
        })?;
        events
    };
    let frame = if custody.parser == "polylogue-request" {
        format!(
            "capture-products:{}",
            blake3::hash(&serde_json::to_vec(&[&custody.frame])?).to_hex()
        )
    } else {
        custody
            .frame
            .strip_suffix(":observation")
            .ok_or_else(|| {
                Error::Invalid("retained restore has no original observation operation".into())
            })?
            .into()
    };
    Ok(Ingress::Events {
        frame,
        source: Some(material.source.clone()),
        coverage: None,
        events,
        failures: Vec::new(),
    })
}

fn conserve(
    custody: &QueuedPublicationCustody,
    raw: &RawValue,
) -> Result<(polylogue_model::historical::ConservedAccepted, String)> {
    let prepared: Prepared<'_> = crate::client::confirmed::source3::parse(raw.get().as_bytes())?;
    let accepted = polylogue_store::assertions::conversion::conserve_request(
        custody.source_schema(),
        prepared.request.get().as_bytes(),
        Some(prepared.outcome.get().as_bytes()),
        Some(prepared.effect.get().as_bytes()),
        None,
        None,
    )?;
    require_source(&accepted)?;
    let meta: RequestMeta = crate::spool::decode_json(prepared.request.get().as_bytes())?;
    Ok((accepted, meta.id))
}
fn finish(payload: &mut Value, kind: &str, natural: Value, record: Value) -> Result<()> {
    let source: Option<Value> = None;
    let key = blake3::hash(&serde_json::to_vec(&(kind, source, natural))?)
        .to_hex()
        .to_string();
    payload["record_key"] = key.into();
    payload["record"] = record;
    Ok(())
}
fn event(
    kind: &str,
    payload: Value,
    location: Option<RecordLocation>,
    parents: Vec<sinex_model::EventId>,
    output_index: u64,
) -> Result<ProducerEvent> {
    let kind = KindRef {
        name: kind.into(),
        version: 4,
    };
    crate::client::confirmed::source3::header(&kind)?;
    Ok(ProducerEvent {
        kind,
        payload,
        location,
        output_index,
        parents,
        reading_order: None,
        class: Class::Open,
    })
}

/// The predecessor SDK refused signal-bearing Source publication too. A
/// retained material cannot turn that derived-only content into a Source event.
pub(super) fn require_source(
    accepted: &polylogue_model::historical::ConservedAccepted,
) -> Result<()> {
    use polylogue_model::{
        historical::{ConservedAccepted, Intent},
        user::UserWrite,
    };
    let (signal_action, effect) = match accepted {
        ConservedAccepted::Compatible(prepared) => (
            matches!(prepared.request.write, UserWrite::SignalCommit(_)),
            Some(&prepared.effect),
        ),
        ConservedAccepted::Historical(history) => (
            matches!(&history.intent,Some(Intent::Other{write}) if matches!(write.as_ref(),UserWrite::SignalCommit(_))),
            history
                .conserved_effects
                .as_ref()
                .map(|effects| &effects.current),
        ),
    };
    if signal_action
        || effect.is_some_and(|effect| {
            !effect.signal_checkpoints.is_empty()
                || !effect.signal_events.is_empty()
                || effect
                    .deliveries
                    .iter()
                    .any(|change| change.delivery.signal.is_some())
        })
    {
        return Err(Error::Refused {
            name: accepted.id().into(),
            reason: "retained signal content still requires derived admission evidence".into(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
