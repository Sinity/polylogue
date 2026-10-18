//! Byte-exact access to source JSON. Values are read through `RawValue`, so
//! numbers keep their original spelling in identity and version bytes.
use serde_json::value::RawValue;
use std::collections::BTreeMap;

/// Raw object members, each kept as its original bytes.
pub(crate) type Object = BTreeMap<String, Box<RawValue>>;

/// The raw value at `path`, or `None` when a member is missing or a step is
/// not an object. Malformed JSON is an error.
pub(crate) fn field<'a>(bytes: &'a [u8], path: &[&str]) -> Result<Option<&'a [u8]>, String> {
    let mut raw: &'a RawValue = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    for name in path {
        if !raw.get().starts_with('{') {
            return Ok(None);
        }
        let members: BTreeMap<String, &'a RawValue> =
            serde_json::from_str(raw.get()).map_err(|e| e.to_string())?;
        let Some(next) = members.get(*name).copied() else {
            return Ok(None);
        };
        raw = next;
    }
    Ok(Some(raw.get().as_bytes()))
}

/// Canonical record bytes with the named top-level members removed.
pub(crate) fn canonical(bytes: &[u8], volatile: &[&str]) -> Result<Vec<u8>, String> {
    polylogue_model::identity::canonical_record(bytes, volatile).map_err(|e| e.to_string())
}

pub(crate) fn object(bytes: &[u8]) -> Result<Object, String> {
    serde_json::from_slice(bytes).map_err(|e| e.to_string())
}

pub(crate) fn encode(members: &Object) -> Result<Vec<u8>, String> {
    serde_json::to_vec(members).map_err(|e| e.to_string())
}
