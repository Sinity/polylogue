//! Privacy assessment precedes model-page framing. Non-scalar strings retain
//! their exact UTF-16 units only when the scalar assessment is unchanged;
//! a transformation with uncertain reconstruction refuses the whole value.
use super::{deadline, unchanged_key, write_string};
use credential_output::Masker;
use polylogue_model::RetainedJson;
use serde_json::{Value, json};
use std::{io, time::Instant};

pub(super) fn present(
    value: &RetainedJson,
    until: Instant,
    preserve: bool,
) -> io::Result<(RetainedJson, bool)> {
    deadline(until)?;
    match value {
        RetainedJson::String(raw) => {
            let units = raw.native().utf16();
            let scalar = String::from_utf16(&units);
            let assessment = scalar
                .as_ref()
                .cloned()
                .unwrap_or_else(|_| String::from_utf16_lossy(&units));
            let mut encoded = Vec::new();
            write_string(&mut encoded, &assessment, &mut Masker::new(), until, false)?;
            let masked: String = serde_json::from_slice(&encoded)?;
            if masked == assessment {
                return Ok((value.clone(), false));
            }
            if preserve || scalar.is_err() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "retained string requires an unavailable privacy transformation",
                ));
            }
            Ok((RetainedJson::from(masked), true))
        }
        RetainedJson::Array(items) => {
            let mut changed = false;
            let mut result = Vec::with_capacity(items.len());
            for item in items {
                let (item, masked) = present(item, until, preserve)?;
                changed |= masked;
                result.push(item);
            }
            Ok((RetainedJson::Array(result), changed))
        }
        RetainedJson::Object(fields) => {
            let mut changed = false;
            let mut result = std::collections::BTreeMap::new();
            for (key, value) in fields {
                // Never rename a structural key, including one with exact
                // non-scalar units. The assessment is not emitted as data.
                unchanged_key(&String::from_utf16_lossy(&key.utf16()), until)?;
                let (value, masked) = present(value, until, preserve)?;
                changed |= masked;
                result.insert(key.clone(), value);
            }
            Ok((RetainedJson::Object(result), changed))
        }
        _ => Ok((value.clone(), false)),
    }
}

pub(crate) fn envelope(value: &RetainedJson, until: Instant, preserve: bool) -> io::Result<Value> {
    let (value, masked) = present(value, until, preserve)?;
    let canonical = String::from_utf8(value.canonical()).map_err(io::Error::other)?;
    Ok(json!({"encoding":"canonical-json-v1","json":canonical,"masked":masked}))
}

#[cfg(test)]
mod tests;
