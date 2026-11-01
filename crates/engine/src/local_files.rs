//! Files explicitly attached to a Codex turn. Relative paths without a
//! recorded directory remain in the transcript; they are never resolved
//! against the daemon's working directory.

use polylogue_model::Json;
use std::path::PathBuf;

pub(crate) fn references(bytes: &[u8], initial_directory: Option<&str>) -> Vec<PathBuf> {
    let mut directory = initial_directory.map(PathBuf::from);
    let mut paths = std::collections::BTreeSet::new();
    for line in bytes.split(|b| *b == b'\n') {
        if line.windows(5).any(|w| w == b"\"cwd\"")
            && let Ok(value) = serde_json::from_slice::<Json>(line)
            && matches!(value.get("type").and_then(Json::as_str), Some("session_meta" | "turn_context"))
            && let Some(cwd) = value.get("payload").and_then(|v| v.get("cwd")).and_then(Json::as_str) {
            directory = Some(PathBuf::from(cwd));
        }
        if ![b"local_images".as_slice(), b"local_image", b"file://"]
            .iter()
            .any(|key| line.windows(key.len()).any(|w| w == *key))
        {
            continue;
        }
        if let Ok(value) = serde_json::from_slice::<Json>(line) {
            let mut record = std::collections::BTreeSet::new();
            visit(&value, &mut record);
            paths.extend(record.into_iter().map(|path| {
                if path.is_relative() { directory.as_ref().map_or_else(|| path.clone(), |cwd| cwd.join(&path)) }
                else { path }
            }));
        }
    }
    paths.into_iter().collect()
}

fn add(path: &str, paths: &mut std::collections::BTreeSet<PathBuf>) {
    let path = PathBuf::from(path.strip_prefix("file://").unwrap_or(path));
    paths.insert(path);
}

fn visit(value: &Json, paths: &mut std::collections::BTreeSet<PathBuf>) {
    match value {
        Json::Object(object) => {
            if let Some(images) = object.get("local_images").and_then(Json::as_array) {
                for image in images {
                    if let Some(path) = image
                        .as_str()
                        .or_else(|| image.get("path").and_then(Json::as_str))
                    {
                        add(path, paths);
                    }
                }
            }
            if object.get("type").and_then(Json::as_str) == Some("local_image") {
                if let Some(path) = object.get("path").and_then(Json::as_str) {
                    add(path, paths);
                }
            }
            if matches!(
                object.get("type").and_then(Json::as_str),
                Some("input_image" | "image")
            ) {
                if let Some(path) = object
                    .get("image_url")
                    .or_else(|| object.get("url"))
                    .and_then(Json::as_str)
                    .filter(|p| p.starts_with("file://"))
                {
                    add(path, paths);
                }
            }
            for value in object.values() {
                visit(value, paths);
            }
        }
        Json::Array(items) => {
            for item in items {
                visit(item, paths);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests;
