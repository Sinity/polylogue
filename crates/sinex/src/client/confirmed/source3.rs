//! One-way read projection for the immutable predecessor request contract.
//! This path never prepares a frame, accepts a request, or grants execution.
mod contract;
pub(crate) use contract::request_kind;
mod envelope;
pub(crate) use envelope::{parse, visit_writes};
mod provenance;
mod queries;
mod queued;
mod runs;
pub(crate) use queued::{validate_source, visit_body};
#[cfg(test)]
mod tests;

use crate::{Error, Result};
use polylogue_model::historical::SourceSchema;
use polylogue_store::primary::Product;
use sinex_protocol::ConfirmedEvent;

pub(super) fn decode(event: &ConfirmedEvent) -> Result<Option<Product>> {
    if let Some(product) = queries::decode(event)? {
        return Ok(Some(product));
    }
    if let Some(product) = runs::decode(event)? {
        return Ok(Some(product));
    }
    if !contract::matches(event)? {
        return Ok(None);
    }
    let decode = || -> Result<Product> {
        let encoded = serde_json::to_vec(&event.payload)?;
        let fields = envelope::extract(&event.event_type, &encoded)?;
        let source = event
            .source_ref
            .as_ref()
            .ok_or_else(|| Error::Invalid("historical native source is absent".into()))?;
        let fingerprint = event.source_fingerprint.as_ref().ok_or_else(|| {
            Error::Invalid("historical native source fingerprint is absent".into())
        })?;
        let accepted = polylogue_store::assertions::conversion::conserve_request(
            SourceSchema::NativeProjection {
                source: source.name.clone(),
                source_version: source.version,
                source_fingerprint: fingerprint.clone(),
                kind: event.event_type.clone(),
                kind_version: event.kind_version,
                kind_definition_hash: event.kind_definition_hash.clone(),
            },
            fields.request,
            Some(fields.outcome),
            Some(fields.effect),
            None,
            Some(provenance::from_event(event, encoded.clone())?),
        )?;
        Ok(Product::ConservedRequest {
            accepted,
            claims: fields.claims,
        })
    };
    decode().map(Some).map_err(|error| Error::Refused {
        name: format!("{}@{}", event.event_type, event.kind_version),
        reason: format!("historical accepted request cannot be conserved: {error}"),
    })
}

pub(in crate::client) fn origin_supported(event: &ConfirmedEvent) -> Result<bool> {
    contract::origin_supported(event)
}

pub(crate) fn historical_request_declarations() -> Result<Vec<sinex_model::KindDeclaration>> {
    Ok(contract::declaration()?
        .kinds
        .iter()
        .filter(|kind| kind.kind.version == 4 && contract::request_kind(&kind.kind.name))
        .cloned()
        .collect())
}

/// Frozen producer contract identity for a private retained continuation.
pub(crate) fn source_fingerprint() -> &'static str {
    contract::SOURCE_FINGERPRINT
}

pub(crate) fn header(kind: &sinex_model::KindRef) -> Result<sinex_protocol::RecordedHeaderUse> {
    let declaration = contract::declaration()?
        .kinds
        .iter()
        .find(|declaration| &declaration.kind == kind)
        .ok_or_else(|| Error::Invalid("retained event has no frozen kind declaration".into()))?;
    Ok(sinex_protocol::RecordedHeaderUse {
        kind: kind.clone(),
        kind_definition_hash: declaration.definition_hash(),
        class: sinex_model::Class::Open,
    })
}

/// Exact Run2 declarations are read-only and never current registration.
pub(crate) fn historical_run_declarations() -> Result<Vec<sinex_model::KindDeclaration>> {
    Ok(contract::declaration()?
        .kinds
        .iter()
        .filter(|kind| kind.kind.name == "polylogue.query_run" && kind.kind.version == 2)
        .cloned()
        .collect())
}

/// Immutable Query1 data is read-only, independent of current catalog support.
pub(crate) fn historical_query_declarations() -> Result<Vec<sinex_model::KindDeclaration>> {
    Ok(contract::declaration()?
        .kinds
        .iter()
        .filter(|kind| kind.kind.name == "polylogue.query" && kind.kind.version == 1)
        .cloned()
        .collect())
}
