//! Frozen Run2 readings retain their old JSON without current cursor authority.
use super::{contract, envelope, provenance};
use crate::{Error, Result};
use polylogue_model::{
    BlobHash,
    historical::{NativeProjectionHistory, NativeProjectionObservation, SourceSchema},
};
use polylogue_store::primary::Product;
use query_language::identity::QuestionId;
use serde::Deserialize;
use serde_json::value::RawValue;
use sinex_protocol::ConfirmedEvent;

#[derive(Deserialize)]
struct Payload<'a> {
    #[serde(borrow)]
    record: &'a RawValue,
    #[serde(default)]
    binding_hash: Option<BlobHash>,
    #[serde(default)]
    run_hash: Option<BlobHash>,
}
#[derive(Deserialize)]
struct Record<'a> {
    product: &'a str,
    #[serde(borrow)]
    run: &'a RawValue,
}

pub(super) fn decode(event: &ConfirmedEvent) -> Result<Option<Product>> {
    if !contract::matches_run(event)? {
        return Ok(None);
    }
    let encoded_payload = serde_json::to_vec(&event.payload)?;
    let payload: Payload<'_> = envelope::parse(&encoded_payload)?;
    let record: Record<'_> = envelope::parse(payload.record.get().as_bytes())?;
    if record.product != "query_run" {
        return Err(Error::Invalid(
            "frozen Run2 reading contains another product".into(),
        ));
    }
    let body: serde_json::Value = crate::spool::decode_json(record.run.get().as_bytes())?;
    // Source3 had no exact-object Run and required an actual QuestionId. New
    // optional questions or bindings cannot be accepted through this contract.
    if !body.get("question").is_some_and(|question| {
        !question.is_null() && serde_json::from_value::<QuestionId>(question.clone()).is_ok()
    }) || !matches!(
        body.pointer("/binding/kind")
            .and_then(serde_json::Value::as_str),
        Some("original" | "continuation")
    ) {
        return Err(Error::Invalid(
            "Run2 reading does not match its frozen question/binding shape".into(),
        ));
    }
    let source = SourceSchema::NativeProjection {
        source: contract::declaration()?.source.name.clone(),
        source_version: contract::declaration()?.source.version,
        source_fingerprint: contract::SOURCE_FINGERPRINT.into(),
        kind: event.event_type.clone(),
        kind_version: event.kind_version,
        kind_definition_hash: event.kind_definition_hash.clone(),
    };
    let mut native = NativeProjectionHistory::default();
    native
        .observe(NativeProjectionObservation {
            source: source.clone(),
            provenance: provenance::from_event(event, encoded_payload.clone())?,
        })
        .map_err(|error| Error::Invalid(error.to_string()))?;
    let run = polylogue_store::query_objects::conserve_historical_run(
        source,
        record.run.get().as_bytes(),
        payload.binding_hash,
        payload.run_hash,
        native,
    )?;
    Ok(Some(Product::HistoricalQueryRun { run: Box::new(run) }))
}

#[cfg(test)]
mod tests;
