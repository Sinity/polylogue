//! Export member identities and naming sidecars. The caller supplies retained
//! bytes; this module never looks up a pathname or fetches a pointer.
use crate::export::ScanError;
use polylogue_model::Json;
use serde::Deserializer as _;
use serde::de::{Error as _, MapAccess, Visitor};
use std::io::Read;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AssetMetadata {
    pub source_id: String,
    pub name: Option<String>,
    pub media_type: Option<String>,
    pub size: Option<u64>,
}

/// IDs are anchored at a member basename or rendition separator. An arbitrary
/// embedded ID or an original filename is not identity evidence.
pub fn member_id(member: &str) -> Option<&str> {
    if member
        .split('/')
        .any(|part| matches!(part, "" | "." | ".."))
        || member.contains('\\')
    {
        return None;
    }
    let basename = member.rsplit('/').next()?;
    std::iter::once(basename)
        .chain(basename.split('#').skip(1))
        .find_map(|part| {
            let rest = part
                .strip_prefix("file-")
                .or_else(|| part.strip_prefix("file_"))?;
            let len = rest.bytes().take_while(u8::is_ascii_alphanumeric).count();
            if len == 0 || !rest[len..].starts_with(['.', '-', '#']) && len != rest.len() {
                return None;
            }
            Some(&part[..5 + len])
        })
}

pub fn sidecar_kind(member: &str) -> Option<&str> {
    if member
        .split('/')
        .any(|part| matches!(part, "" | "." | ".."))
        || member.contains('\\')
    {
        return None;
    }
    match member.rsplit('/').next()? {
        kind @ ("conversation_asset_file_names.json" | "library_files.json") => Some(kind),
        _ => None,
    }
}

/// Emit one metadata entry at a time. Callers must stage these facts until the
/// complete sidecar, including trailing bytes, has passed validation.
pub fn metadata(
    member: &str,
    reader: impl Read,
    mut emit: impl FnMut(AssetMetadata) -> Result<(), String>,
) -> Result<(), ScanError> {
    let kind = sidecar_kind(member)
        .ok_or_else(|| ScanError::Callback("unsupported asset sidecar name".into()))?;
    if kind == "library_files.json" {
        return crate::export::scan(reader, |_, raw| {
            let value: Json = serde_json::from_str(raw.get()).map_err(|e| e.to_string())?;
            let source_id = value
                .get("file_id")
                .and_then(Json::as_str)
                .filter(|id| !id.is_empty())
                .ok_or("library entry has no string file_id")?;
            emit(AssetMetadata {
                source_id: source_id.to_owned(),
                name: value
                    .get("file_name")
                    .and_then(Json::as_str)
                    .map(str::to_owned),
                media_type: value
                    .get("mime_type")
                    .and_then(Json::as_str)
                    .map(str::to_owned),
                size: value.get("file_size_bytes").and_then(Json::as_u64),
            })
        })
        .map(|_| ());
    }
    let mut de = serde_json::Deserializer::from_reader(std::io::BufReader::new(reader));
    de.deserialize_map(Names(&mut emit))
        .map_err(ScanError::Malformed)?;
    de.end().map_err(ScanError::Malformed)?;
    Ok(())
}

struct Names<'a, F>(&'a mut F);
impl<'de, F: FnMut(AssetMetadata) -> Result<(), String>> Visitor<'de> for Names<'_, F> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("an object mapping provider file IDs to filenames")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        while let Some(key) = map.next_key::<String>()? {
            let name = map.next_value::<String>()?;
            let source_id = key.strip_suffix(".dat").unwrap_or(&key).to_owned();
            if source_id.is_empty() {
                return Err(A::Error::custom("asset name has an empty provider file ID"));
            }
            self.0(AssetMetadata {
                source_id,
                name: Some(name),
                media_type: None,
                size: None,
            })
            .map_err(A::Error::custom)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
