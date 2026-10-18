//! `<session> export --to DIR`: writes the `session-read` document and a
//! manifest into a directory. The document and its completion status come
//! from that one read; nothing here re-queries or re-judges completeness.
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{self, BufReader, Write},
    os::unix::fs::MetadataExt,
    path::Path,
    time::{Duration, Instant},
};

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}
fn filename(id: &str) -> String {
    let mut name = String::new();
    for byte in id.bytes() {
        if byte.is_ascii_alphanumeric() || b"._~-".contains(&byte) {
            name.push(char::from(byte));
        } else {
            use std::fmt::Write;
            write!(&mut name, "%{byte:02X}").expect("String write");
        }
    }
    name.push_str(".json");
    name
}
#[derive(serde::Deserialize)]
struct ExportMarker {
    format: String,
    version: u64,
}
fn existing_export(destination: &Path) -> io::Result<Option<(u64, u64)>> {
    match fs::symlink_metadata(destination) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
        Ok(meta) => {
            if !meta.is_dir() || meta.file_type().is_symlink() {
                return Err(invalid(
                    "export destination must be a directory, never a symlink",
                ));
            }
            let path = destination.join("manifest.json");
            let manifest_meta = fs::symlink_metadata(&path)?;
            if !manifest_meta.is_file() || manifest_meta.file_type().is_symlink() {
                return Err(invalid(
                    "existing destination is not a supported export directory",
                ));
            }
            let manifest: ExportMarker =
                serde_json::from_reader(BufReader::new(File::open(path)?))?;
            if manifest.format != "polylogue-export-manifest" || manifest.version != 1 {
                return Err(invalid(
                    "existing destination is not a version 1 export directory",
                ));
            }
            Ok(Some((meta.dev(), meta.ino())))
        }
    }
}

/// The document fields the manifest copies; serde skips the rest while
/// reading, so messages are never buffered.
#[derive(serde::Deserialize)]
struct DocumentStatus {
    result: Value,
    recovery: Option<polylogue_model::RecoveryRecord>,
}

/// Exports one session into `destination`, whose parent must exist. An
/// existing export directory is swapped out atomically; anything else there
/// is refused. Interrupting the process leaves only a hidden staging
/// directory next to the destination.
pub fn request(socket: &Path, reference: &str, destination: &Path) -> io::Result<Value> {
    let name = destination
        .file_name()
        .filter(|n| !n.is_empty())
        .ok_or_else(|| invalid("export destination needs a directory name"))?;
    if destination
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(invalid("export destination must not contain '..'"));
    }
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = fs::canonicalize(parent)?;
    let destination = parent.join(name);
    let previous = existing_export(&destination)?;
    let parent_file = File::open(&parent)?;
    let stage = tempfile::Builder::new()
        .prefix(".polylogue-export-")
        .tempdir_in(&parent)?;
    let mut document = File::create(stage.path().join("session.json"))?;
    let end = crate::session_read::request(socket, reference, &mut document)?;
    if end["outcome"] == "error" {
        return Ok(end);
    }
    let origin = end["origin"]
        .as_str()
        .ok_or_else(|| invalid("session read did not return an origin"))?;
    polylogue_model::Origin::new(origin).map_err(|e| invalid(e.to_string()))?;
    let native = end["native_id"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| invalid("session read did not return a native ID"))?;
    let status: DocumentStatus = serde_json::from_reader(BufReader::new(File::open(
        stage.path().join("session.json"),
    )?))?;
    if !end["view"].is_string()
        || !end["up_to_date"].is_boolean()
        || status.result["outcome"] != end["outcome"]
        || !status.result.is_object()
        || serde_json::to_value(&status.recovery)?
            != end.get("recovery").cloned().unwrap_or(Value::Null)
    {
        return Err(invalid(
            "session read's end frame does not match its document",
        ));
    }
    document.sync_all()?;
    drop(document);
    let origin_path = stage.path().join(origin);
    fs::create_dir(&origin_path)?;
    fs::rename(
        stage.path().join("session.json"),
        origin_path.join(filename(native)),
    )?;
    File::open(&origin_path)?.sync_all()?;
    let mut manifest = json!({"format":"polylogue-export-manifest","version":1,
        "query":reference,"ran_at":format!("{:.3}",jiff::Timestamp::now()),
        "view":end["view"],"up_to_date":end["up_to_date"],"sessions":1,
        "with_notes":true,"with_inherited":true,"with_blobs":false,
        "exported_by":concat!("polylogue ",env!("CARGO_PKG_VERSION")),"result":status.result});
    if let Some(recovery) = status.recovery {
        manifest["recovery"] = json!(recovery);
    }
    let mut file = File::create(stage.path().join("manifest.json"))?;
    let mut encoder =
        crate::output_json::Encoder::new(&mut file, Instant::now() + Duration::from_secs(30));
    encoder.value(&manifest)?;
    encoder.finish()?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    drop(file);
    File::open(stage.path())?.sync_all()?;
    parent_file.sync_all()?;
    if existing_export(&destination)? != previous {
        return Err(io::Error::other(
            "export destination changed during the read; it was left as is",
        ));
    }
    // NOREPLACE keeps a directory created meanwhile. EXCHANGE moves the old
    // export to the staging name, so a failed sync below still leaves both.
    use rustix::fs::{CWD, RenameFlags, renameat_with};
    renameat_with(
        CWD,
        stage.path(),
        CWD,
        &destination,
        if previous.is_some() {
            RenameFlags::EXCHANGE
        } else {
            RenameFlags::NOREPLACE
        },
    )?;
    if let Some(expected) = previous {
        let old = fs::symlink_metadata(stage.path());
        if old.as_ref().map(|meta| (meta.dev(), meta.ino())).ok() != Some(expected) {
            let kept = stage.keep();
            return Err(io::Error::other(format!(
                "export destination changed during the swap ({old:?}); the replaced directory is at {}; check the destination before retrying",
                kept.display()
            )));
        }
    }
    if let Err(error) = parent_file.sync_all() {
        let kept = stage.keep();
        return Err(io::Error::other(format!(
            "export written but syncing its parent directory failed: {error}; the previous export is at {}; check the destination before retrying",
            kept.display()
        )));
    }
    // Dropping the TempDir removes the old export (or nothing, on a first
    // export); removal does not follow symlinks inside it.
    Ok(end)
}

#[cfg(test)]
mod tests;
