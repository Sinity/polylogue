//! Source JSON access, never normalized Message serialization. Opaque nested
//! payloads replace atomically; only the typed message object's members overlay.
use serde_json::value::RawValue;
use std::collections::BTreeMap;

pub(super) fn canonical(bytes: &[u8]) -> Result<Vec<u8>, String> {
    polylogue_model::identity::canonical_record(bytes, &[]).map_err(|e| e.to_string())
}
pub(super) fn optional_field<'a>(
    bytes: &'a [u8],
    path: &[&str],
) -> Result<Option<&'a [u8]>, String> {
    let mut raw: &'a RawValue = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    for field in path {
        if !raw.get().starts_with('{') {
            return Ok(None);
        }
        let values: BTreeMap<String, &'a RawValue> =
            serde_json::from_str(raw.get()).map_err(|e| e.to_string())?;
        let Some(next) = values.get(*field).copied() else {
            return Ok(None);
        };
        raw = next;
    }
    Ok(Some(raw.get().as_bytes()))
}
pub(super) fn overlay(previous: Option<&[u8]>, incoming: &[u8]) -> Result<Vec<u8>, String> {
    let mut fields: BTreeMap<String, Box<RawValue>> = match previous {
        Some(b) => serde_json::from_slice(b).map_err(|e| e.to_string())?,
        None => BTreeMap::new(),
    };
    let updates: BTreeMap<String, Box<RawValue>> =
        serde_json::from_slice(incoming).map_err(|e| e.to_string())?;
    for (key, value) in updates {
        if key == "message" && value.get().starts_with('{') {
            let mut message: BTreeMap<String, Box<RawValue>> = fields
                .get("message")
                .filter(|v| v.get().starts_with('{'))
                .map(|v| serde_json::from_str(v.get()))
                .transpose()
                .map_err(|e| e.to_string())?
                .unwrap_or_default();
            let supplied: BTreeMap<String, Box<RawValue>> =
                serde_json::from_str(value.get()).map_err(|e| e.to_string())?;
            message.extend(supplied);
            for ignored in ["diagnostics", "container", "context_management"] {
                message.remove(ignored);
            }
            let encoded = serde_json::to_string(&message).map_err(|e| e.to_string())?;
            fields.insert(
                key,
                RawValue::from_string(encoded).map_err(|e| e.to_string())?,
            );
        } else {
            fields.insert(key, value);
        }
    }
    fields.retain(|key, _| {
        !matches!(
            key.as_str(),
            "slug" | "session_id" | "wireToolInputs" | "wireIngestContext"
        ) && !key.starts_with("serverClassifier")
    });
    canonical(&serde_json::to_vec(&fields).map_err(|e| e.to_string())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overlays_keep_exact_number_tokens_and_replace_opaque_usage_without_filling() {
        let a=br#"{"type":"assistant","uuid":"a","message":{"id":"r","content":[{"type":"tool_use","input":{"n":1E+2,"z":-0}}],"usage":{"input_tokens":1,"output_tokens":2}}}"#;
        let b = br#"{"type":"assistant","uuid":"a","message":{"usage":{"output_tokens":3}}}"#;
        let reduced = overlay(Some(a), b).unwrap();
        assert_eq!(
            optional_field(&reduced, &["message", "usage"]).unwrap(),
            Some(br#"{"output_tokens":3}"#.as_slice())
        );
        assert_eq!(
            optional_field(&reduced, &["message", "content"]).unwrap(),
            Some(br#"[{"input":{"n":1E+2,"z":-0},"type":"tool_use"}]"#.as_slice())
        );
        let null = overlay(Some(&reduced), br#"{"message":{"content":null}}"#).unwrap();
        assert_eq!(
            optional_field(&null, &["message", "content"]).unwrap(),
            Some(b"null".as_slice())
        );
        assert_eq!(overlay(Some(&reduced), a).unwrap(), canonical(a).unwrap());
    }
}
