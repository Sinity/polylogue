//! A stored question is data; current executable binding remains a separate step.
use super::{contract, envelope};
use crate::{Error, Result};
use polylogue_model::query_objects::QueryObject;
use polylogue_store::primary::Product;
use serde::Deserialize;
use serde_json::value::RawValue;
use sinex_protocol::ConfirmedEvent;

#[derive(Deserialize)]
struct Payload<'a> {
    #[serde(borrow)]
    record: &'a RawValue,
}
#[derive(Deserialize)]
struct Record<'a> {
    product: &'a str,
    #[serde(borrow)]
    object: &'a RawValue,
}

pub(super) fn decode(event: &ConfirmedEvent) -> Result<Option<Product>> {
    if !contract::matches_query(event)? {
        return Ok(None);
    }
    // No old/new schema superset inference: the unchanged data codec is used
    // only while the current immutable kind definition is exactly the old one.
    let current = crate::kinds::declarations("polylogue")
        .into_iter()
        .find(|kind| kind.kind.name == event.event_type && kind.kind.version == event.kind_version)
        .ok_or_else(|| Error::Invalid("the frozen query has no current data codec".into()))?;
    if current.definition_hash() != event.kind_definition_hash {
        return Err(Error::Invalid(
            "the frozen query data codec changed its immutable definition".into(),
        ));
    }
    let encoded = serde_json::to_vec(&event.payload)?;
    let payload: Payload<'_> = envelope::parse(&encoded)?;
    let record: Record<'_> = envelope::parse(payload.record.get().as_bytes())?;
    if record.product != "query" {
        return Err(Error::Invalid(
            "the frozen query kind contains another product".into(),
        ));
    }
    let object: QueryObject = crate::spool::decode_json(record.object.get().as_bytes())?;
    // Confirmed projection applies integrity-only restoration. Neither this
    // decode nor a metadata getter accepts the old catalog for execution.
    Ok(Some(Product::Query {
        object: Box::new(object),
    }))
}

#[cfg(test)]
mod tests;
