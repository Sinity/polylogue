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
