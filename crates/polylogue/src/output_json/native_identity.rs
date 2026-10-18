//! Privacy examines declared logical identities, never their reversible wire
//! spelling. Provider payloads and literal marker strings are not decoded here.
use super::{retained, unchanged_key};
use polylogue_model::{NativeId, RawJson};
use std::{io, time::Instant};

pub(super) fn check_native(value: &NativeId, until: Instant) -> io::Result<()> {
    // Reuse the retained UTF-16 assessment. Replacement scalars are assessment
    // delimiters only: neither a repaired value nor a transformed ID is emitted.
    retained::present(&RawJson::from(value), until, true).map(|_| ())
}

pub(super) fn check_public(value: &str, until: Instant) -> io::Result<()> {
    let reference = ["session:", "message:", "block:"]
        .iter()
        .find_map(|prefix| value.strip_prefix(prefix))
        .unwrap_or(value);
    if reference.starts_with("~i1:") {
        let components = polylogue_model::identity::framed_identity_components(reference)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "unsupported encoded identity")
            })?;
        for component in components {
            check_native(&component, until)?;
        }
    }
    unchanged_key(value, until)
}

/// Only this source-owned descriptor position contains a NativeId wire value.
/// Never infer encoding from an arbitrary provider object with similar keys.
pub(super) fn check_origin(value: &RawJson, until: Instant) -> io::Result<()> {
    if value.get("format").and_then(RawJson::as_str) != Some("claude-code-field-origin") {
        return Ok(());
    }
    if let Some(parent) = value.pointer("/selection/transform/ParentReferenceV1") {
        let native = parent.get("native_uuid").ok_or_else(invalid_origin)?;
        let native: NativeId =
            serde_json::from_slice(&native.canonical()).map_err(|_| invalid_origin())?;
        check_native(&native, until)?;
        let resolved = parent.get("resolved").ok_or_else(invalid_origin)?;
        for key in ["id", "session"] {
            let id = resolved
                .get(key)
                .and_then(RawJson::as_str)
                .ok_or_else(invalid_origin)?;
            check_public(id, until)?;
        }
    }
    Ok(())
}
fn invalid_origin() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "invalid declared native origin witness",
    )
}
