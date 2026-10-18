//! Claude record overlays. A later record with the same UUID replaces whole
//! top-level members; only the `message` object merges member by member.
use crate::raw::{Object, encode, object};
use serde_json::value::RawValue;

pub(super) fn canonical(bytes: &[u8]) -> Result<Vec<u8>, String> {
    crate::raw::canonical(bytes, &[])
}
pub(super) fn overlay(previous: Option<&[u8]>, incoming: &[u8]) -> Result<Vec<u8>, String> {
    let mut fields = previous.map(object).transpose()?.unwrap_or_default();
    for (key, value) in object(incoming)? {
        if key == "message" && value.get().starts_with('{') {
            let mut message: Object = fields
                .get("message")
                .filter(|v| v.get().starts_with('{'))
                .map(|v| object(v.get().as_bytes()))
                .transpose()?
                .unwrap_or_default();
            message.extend(object(value.get().as_bytes())?);
            for ignored in ["diagnostics", "container", "context_management"] {
                message.remove(ignored);
            }
            let encoded = String::from_utf8(encode(&message)?).map_err(|e| e.to_string())?;
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
    canonical(&encode(&fields)?)
}

/// Decode the overlaid record.
pub(super) fn decode_effective(
    raw: &[u8],
    incoming: &polylogue_model::RawJson,
) -> Result<polylogue_model::RawJson, String> {
    let mut v: polylogue_model::RawJson = serde_json::from_slice(raw).map_err(|e| e.to_string())?;
    // Take the members this record supplied from its own decoded value, so
    // text repairs keep their original byte offsets.
    if let (Some(target), Some(supplied)) = (v.as_object_mut(), incoming.as_object()) {
        for (key, value) in supplied {
            if key == &polylogue_model::NativeId::unicode("message") && value.is_object() {
                if let Some(message) = target
                    .get_mut(key)
                    .and_then(polylogue_model::RawJson::as_object_mut)
                {
                    for (field, value) in value.as_object().into_iter().flatten() {
                        if message.contains_key(field) {
                            message.insert(field.clone(), value.clone());
                        }
                    }
                }
            } else if target.contains_key(key) {
                target.insert(key.clone(), value.clone());
            }
        }
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overlays_keep_number_spelling_and_replace_usage_whole() {
        let a=br#"{"type":"assistant","uuid":"a","message":{"id":"r","content":[{"type":"tool_use","input":{"n":1E+2,"z":-0}}],"usage":{"input_tokens":1,"output_tokens":2}}}"#;
        let b = br#"{"type":"assistant","uuid":"a","message":{"usage":{"output_tokens":3}}}"#;
        let reduced = overlay(Some(a), b).unwrap();
        assert_eq!(
            crate::raw::field(&reduced, &["message", "usage"]).unwrap(),
            Some(br#"{"output_tokens":3}"#.as_slice())
        );
        assert_eq!(
            crate::raw::field(&reduced, &["message", "content"]).unwrap(),
            Some(br#"[{"input":{"n":1E+2,"z":-0},"type":"tool_use"}]"#.as_slice())
        );
        let null = overlay(Some(&reduced), br#"{"message":{"content":null}}"#).unwrap();
        assert_eq!(
            crate::raw::field(&null, &["message", "content"]).unwrap(),
            Some(b"null".as_slice())
        );
        assert_eq!(overlay(Some(&reduced), a).unwrap(), canonical(a).unwrap());
    }
}
