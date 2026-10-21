//! Borrow the manifest's JSON bytes instead of building an intermediate tree.
use super::ExcisionPublication;
use crate::Result;
use serde_json::value::RawValue;
use std::collections::BTreeMap;

pub(super) fn publication(json: &str) -> Result<Option<ExcisionPublication>> {
    // A new excision starts with a refusal list; no publication exists yet.
    if !json.trim_ascii_start().starts_with('{') {
        let _: serde::de::IgnoredAny = serde_json::from_str(json)?;
        return Ok(None);
    }
    // Map decoding keeps last-value semantics for repeated outer field names.
    // An explicit null is present and must fail the typed manifest decoder.
    let fields: BTreeMap<String, &RawValue> = serde_json::from_str(json)?;
    fields
        .get("publication")
        .map(|raw| {
            let publication: ExcisionPublication = serde_json::from_str(raw.get())?;
            publication.validate()?;
            Ok(publication)
        })
        .transpose()
}

#[cfg(test)]
mod tests;
