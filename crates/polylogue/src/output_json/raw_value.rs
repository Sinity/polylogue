//! Credential masking for `RawJson` values. A string with lone surrogates is
//! checked through its lossy text and kept unchanged when that text needs no
//! masking; if it does, the whole value is refused, since masking it could not
//! keep its exact UTF-16 units.
use super::{deadline, unchanged_key, write_string};
use credential_output::Masker;
use polylogue_model::RawJson;
use serde_json::{Value, json};
use std::{io, time::Instant};

pub(super) fn mask(value: &RawJson, until: Instant, preserve: bool) -> io::Result<(RawJson, bool)> {
    deadline(until)?;
    match value {
        RawJson::String(raw) => {
            let units = raw.native().utf16();
            let scalar = String::from_utf16(&units);
            let text = scalar
                .as_ref()
                .cloned()
                .unwrap_or_else(|_| String::from_utf16_lossy(&units));
            let mut encoded = Vec::new();
            write_string(&mut encoded, &text, &mut Masker::new(), until, false)?;
            let masked: String = serde_json::from_slice(&encoded)?;
            if masked == text {
                return Ok((value.clone(), false));
            }
            if preserve || scalar.is_err() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "string needs masking but has lone surrogates",
                ));
            }
            Ok((RawJson::from(masked), true))
        }
        RawJson::Array(items) => {
            let mut changed = false;
            let mut result = Vec::with_capacity(items.len());
            for item in items {
                let (item, masked) = mask(item, until, preserve)?;
                changed |= masked;
                result.push(item);
            }
            Ok((RawJson::Array(result), changed))
        }
        RawJson::Object(fields) => {
            let mut changed = false;
            let mut result = std::collections::BTreeMap::new();
            for (key, value) in fields {
                // Keys are never masked; one that needs masking is refused.
                // The lossy text is only checked, never written.
                unchanged_key(&String::from_utf16_lossy(&key.utf16()), until)?;
                let (value, masked) = mask(value, until, preserve)?;
                changed |= masked;
                result.insert(key.clone(), value);
            }
            Ok((RawJson::Object(result), changed))
        }
        _ => Ok((value.clone(), false)),
    }
}

pub(crate) fn envelope(value: &RawJson, until: Instant, preserve: bool) -> io::Result<Value> {
    let (value, masked) = mask(value, until, preserve)?;
    let canonical = String::from_utf8(value.canonical()).map_err(io::Error::other)?;
    Ok(json!({"encoding":"canonical-json-v1","json":canonical,"masked":masked}))
}

#[cfg(test)]
mod tests;
