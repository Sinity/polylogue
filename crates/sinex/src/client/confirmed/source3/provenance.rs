//! Actual owner-fetched metadata stays separate from accepted-intent evidence.
use crate::{Error, Result};
use polylogue_model::historical::{NativeProjectionProvenance, NativeRecordLocation};
use sinex_protocol::ConfirmedEvent;

pub(super) fn from_event(
    event: &ConfirmedEvent,
    projected_payload: Vec<u8>,
) -> Result<NativeProjectionProvenance> {
    Ok(NativeProjectionProvenance {
        event: event.id.to_string(),
        run: event.run.to_string(),
        commit_seq: event.commit_seq,
        output_index: event.output_index,
        record_hash: event.record_hash.clone(),
        projection_epoch: event.projection.policy_epoch,
        projection_payload_hash: event.projection.payload_hash.clone(),
        location: event
            .location
            .as_ref()
            .map(|location| NativeRecordLocation {
                material: location.material.to_string(),
                anchor: location.anchor,
            }),
        parents: event.parents.iter().map(ToString::to_string).collect(),
        producer_family: event.producer_family.clone(),
        producer_identity: event.producer_identity.clone(),
        ts_orig: event.ts_orig.map(timestamp).transpose()?,
        ts_quality: event.ts_quality.clone(),
        ts_coided: event.ts_coided.map(timestamp).transpose()?,
        ts_persisted: timestamp(event.ts_persisted)?,
        record_state: event.record_state.clone(),
        class: wire_name(&event.class)?,
        validation: wire_name(&event.validation)?,
        equivalence_key: event.equivalence_key.clone(),
        reading_order: event.reading_order,
        projected_payload,
    })
}

fn timestamp(value: sinex_model::Timestamp) -> Result<polylogue_model::Timestamp> {
    polylogue_model::Timestamp::from_unix_ns(value.unix_ns()).ok_or_else(|| {
        Error::Invalid("native timestamp is outside the retained timestamp range".into())
    })
}

fn wire_name(value: &impl serde::Serialize) -> Result<String> {
    serde_json::from_value(serde_json::to_value(value)?).map_err(Error::from)
}
