//! Immutable producer source formats projected into the current executable model.
//!
//! Native fact identity remains the captured kind, definition and payload digest.
//! The same structured-query conversion also owns the numbered archive upgrade.
use super::{Product, decode};
mod progress_v5;
use crate::{Error, Result};
use polylogue_model::query_objects::QueryObject;
use serde_json::Value;
use sinex_model::{KindDeclaration, KindRef};

fn historical_declarations() -> &'static [KindDeclaration] {
    static DECLARATIONS: std::sync::LazyLock<Vec<KindDeclaration>> =
        std::sync::LazyLock::new(|| {
            serde_json::from_str(include_str!("source_formats/structured_query_v1.json"))
                .expect("checked historical producer declarations")
        });
    &DECLARATIONS
}

fn known_historical_definition(kind: &KindRef, definition_hash: &str) -> bool {
    static DEFINITIONS: std::sync::LazyLock<Vec<(KindRef, String)>> =
        std::sync::LazyLock::new(|| {
            historical_declarations()
                .iter()
                .map(|declaration| (declaration.kind.clone(), declaration.definition_hash()))
                .collect()
        });
    DEFINITIONS
        .iter()
        .any(|(known, hash)| known == kind && hash == definition_hash)
}

#[cfg(test)]
pub(crate) fn historical_source_declarations() -> Vec<KindDeclaration> {
    historical_declarations().to_vec()
}

/// Current declarations are admitted as current types. Historical inputs select
/// one exact immutable producer declaration before any representation conversion.
pub(crate) fn decode_source(
    kind: &KindRef,
    definition_hash: &str,
    mut payload: Value,
) -> Result<Product> {
    if let Some(current) = crate::kinds::declared_definition(kind) {
        if current != definition_hash {
            return Err(Error::Invalid(
                "a confirmed event's kind definition differs from Polylogue's".into(),
            ));
        }
        return decode(&kind.name, payload);
    }
    let progress_v5 = progress_v5::declaration();
    if *kind == progress_v5.kind && definition_hash == progress_v5::definition_hash() {
        // Its unchanged fields have the current typed representation. Absence
        // of stopped_attempt stays unrecorded, never an inferred old attempt.
        return progress_v5::project(payload);
    }
    if !known_historical_definition(kind, definition_hash) {
        return Err(Error::Invalid(format!(
            "unsupported Polylogue kind {}@{}",
            kind.name, kind.version
        )));
    }
    let record = payload
        .get_mut("record")
        .ok_or_else(|| Error::Invalid("historical query fact has no complete record".into()))?;
    let field = match kind.name.as_str() {
        "polylogue.query" => "object",
        "polylogue.query_run" => "run",
        _ => {
            return Err(Error::Invalid(
                "unsupported historical query representation".into(),
            ));
        }
    };
    let source = record
        .get_mut(field)
        .ok_or_else(|| Error::Invalid("historical query record has no query body".into()))?;
    *source = match kind.name.as_str() {
        "polylogue.query" => serde_json::to_value(
            QueryObject::from_structured_v1(source.clone()).map_err(Error::Invalid)?,
        )?,
        "polylogue.query_run"
            if kind.version == 2
                && (source["binding"]["kind"] == "continuation"
                    || source["binding"]["execution"]["presentation"]["kind"] == "read") =>
        {
            serde_json::to_value(
                polylogue_model::query_objects::QueryRun::from_retired_read_v2(source.clone())
                    .map_err(Error::Invalid)?,
            )?
        }
        "polylogue.query_run" => serde_json::to_value(
            polylogue_store::primary::products::query_run_from_structured_v1(source.clone())?,
        )?,
        _ => unreachable!("historical declaration selected above"),
    };
    let product = decode(&kind.name, payload.clone())?;
    if let Product::QueryRun { run } = &product {
        for (key, actual) in [
            ("binding_hash", run.binding_hash()?),
            ("run_hash", run.record_hash()?),
        ] {
            if let Some(expected) = payload.get(key).filter(|value| !value.is_null())
                && *expected != serde_json::to_value(actual)?
            {
                return Err(Error::Invalid(format!(
                    "historical query run {key} differs from its exact source custody"
                )));
            }
        }
    }
    Ok(product)
}

#[cfg(test)]
mod tests;
