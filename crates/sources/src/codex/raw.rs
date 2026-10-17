//! Exact raw JSON access. Decoded Value is interpretation data, never the source
//! of identity or version bytes: it normalizes some numeric spellings.
use serde_json::value::RawValue;
use std::collections::BTreeMap;

pub(super) fn field<'a>(raw: &'a [u8], path: &[&str]) -> Result<&'a [u8], String> {
    let mut value: &'a RawValue = serde_json::from_slice(raw).map_err(|e| e.to_string())?;
    for name in path {
        let fields: BTreeMap<String, &'a RawValue> =
            serde_json::from_str(value.get()).map_err(|e| e.to_string())?;
        value = fields
            .get(*name)
            .copied()
            .ok_or_else(|| format!("missing raw Codex field {name}"))?;
    }
    Ok(value.get().as_bytes())
}
pub(super) fn canonical(raw: &[u8]) -> Result<Vec<u8>, String> {
    polylogue_model::identity::canonical_record(raw, &["timestamp", "ordinal"])
        .map_err(|e| e.to_string())
}
pub(super) fn canonical_field(raw: &[u8], path: &[&str]) -> Result<Vec<u8>, String> {
    polylogue_model::identity::canonical_record(field(raw, path)?, &[]).map_err(|e| e.to_string())
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
    fn object(raw: &[u8]) -> Result<BTreeMap<String, Box<RawValue>>, String> {
        serde_json::from_slice(raw).map_err(|e| e.to_string())
    }
    fn encode(fields: BTreeMap<String, Box<RawValue>>) -> Result<Vec<u8>, String> {
        serde_json::to_vec(&fields).map_err(|e| e.to_string())
    }
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
        encode(fields)
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
    let item = encode(previous)?;
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
    pub value: polylogue_model::RetainedJson,
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
