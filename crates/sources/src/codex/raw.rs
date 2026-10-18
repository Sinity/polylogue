//! Codex raw field access and item overlays. Decoded values normalize some
//! numeric spellings, so identity and version bytes come from here.
use crate::raw::{encode, object};

/// The raw value at `path`; a missing member is an error.
pub(super) fn field<'a>(raw: &'a [u8], path: &[&str]) -> Result<&'a [u8], String> {
    crate::raw::field(raw, path)?
        .ok_or_else(|| format!("missing raw Codex field {}", path.join(".")))
}
pub(super) fn canonical(raw: &[u8]) -> Result<Vec<u8>, String> {
    crate::raw::canonical(raw, &["timestamp", "ordinal"])
}
pub(super) fn canonical_field(raw: &[u8], path: &[&str]) -> Result<Vec<u8>, String> {
    crate::raw::canonical(field(raw, path)?, &[])
}
pub(super) fn session_argument(raw: &[u8], legacy: bool) -> Result<Vec<u8>, String> {
    let args = field(
        raw,
        if legacy {
            &["arguments"]
        } else {
            &["payload", "arguments"]
        },
    )?;
    if args.first() == Some(&b'"') {
        let inner: String = serde_json::from_slice(args).map_err(|e| e.to_string())?;
        canonical_field(inner.as_bytes(), &["session_id"])
    } else {
        canonical_field(args, &["session_id"])
    }
}
/// Field overlay at the provider item boundary. New field values replace whole
/// fields (including arrays); fields absent from the newer snapshot survive.
/// Structural rollout envelopes are rebuilt without decoding numeric values.
pub(super) fn overlay(
    old: &[u8],
    new: &[u8],
    path: &[&str],
) -> Result<(Vec<u8>, ValueFields), String> {
    fn replace(raw: &[u8], path: &[&str], value: &[u8]) -> Result<Vec<u8>, String> {
        if path.is_empty() {
            return Ok(value.to_vec());
        }
        let mut fields = object(raw)?;
        let child = fields
            .get(path[0])
            .ok_or("missing structural Codex envelope")?;
        let replaced = replace(child.get().as_bytes(), &path[1..], value)?;
        fields.insert(
            path[0].into(),
            serde_json::from_slice(&replaced).map_err(|e| e.to_string())?,
        );
        encode(&fields)
    }
    let mut previous = object(field(old, path)?)?;
    let newer = object(field(new, path)?)?;
    let provided: Vec<String> = newer.keys().cloned().collect();
    let missing: Vec<String> = previous
        .keys()
        .filter(|key| !newer.contains_key(*key))
        .cloned()
        .collect();
    let retained = missing.clone();
    let mut conflicts = missing;
    for name in ["role", "type"] {
        if let (Some(a), Some(b)) = (previous.get(name), newer.get(name)) {
            if a.get() != b.get() {
                conflicts.push(name.into());
            }
        }
    }
    previous.extend(newer);
    let item = encode(&previous)?;
    let canonical = replace(new, path, &item)?;
    let value = serde_json::from_slice(&item).map_err(|e| e.to_string())?;
    Ok((
        canonical,
        ValueFields {
            value,
            conflicts,
            provided,
            retained,
        },
    ))
}
pub(super) struct ValueFields {
    pub value: polylogue_model::RawJson,
    pub conflicts: Vec<String>,
    pub provided: Vec<String>,
    pub retained: Vec<String>,
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_raw_numbers_survive_without_value_reserialization() {
        let raw = br#"{"timestamp":"old","ordinal":9,"payload":{"n":1E+2,"z":-0}}"#;
        assert_eq!(field(raw, &["payload"]).unwrap(), br#"{"n":1E+2,"z":-0}"#);
        assert_eq!(canonical(raw).unwrap(), br#"{"payload":{"n":1E+2,"z":-0}}"#);
        assert!(field(raw, &["missing"]).is_err());
    }
}
