//! Exact-Session export is a filesystem consumer of the ordinary pinned read.
//! No second query, document formatter or completeness decision happens here.
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io::{self, BufReader, Write},
    os::unix::fs::MetadataExt,
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}
fn check(stop: &AtomicBool) -> io::Result<()> {
    if stop.load(Ordering::Acquire) {
        Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "export cancelled before publication",
        ))
    } else {
        Ok(())
    }
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

// Serde skips unselected fields directly from the reader. No Session payload
// buffer or independently reconstructed completeness result is needed here.
#[derive(serde::Deserialize)]
struct DocumentQualification {
    result: Value,
    recovery: Option<polylogue_model::RecoveryQualification>,
}

/// Publish one detached document and its same-read manifest. Existing recognized
/// export directories are exchanged atomically on Linux; other occupants refuse.
/// The parent must already exist. This does not implement query/empty exports.
pub fn request(
    socket: &Path,
    reference: &str,
    destination: &Path,
    stop: &AtomicBool,
) -> io::Result<Value> {
    check(stop)?;
    let name = destination
        .file_name()
        .filter(|n| !n.is_empty())
        .ok_or_else(|| invalid("export destination needs a directory name"))?;
    if destination
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err(invalid(
            "export destination must not contain parent traversal",
        ));
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
    check(stop)?;
    let origin = end["origin"]
        .as_str()
        .ok_or_else(|| invalid("read lacks export origin"))?;
    polylogue_model::Origin::new(origin).map_err(|e| invalid(e.to_string()))?;
    let native = end["native_id"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| invalid("read lacks export native identity"))?;
    let qualification: DocumentQualification = serde_json::from_reader(BufReader::new(
        File::open(stage.path().join("session.json"))?,
    ))?;
    if !end["view"].is_string()
        || !end["up_to_date"].is_boolean()
        || qualification.result["outcome"] != end["outcome"]
        || !qualification.result.is_object()
        || serde_json::to_value(&qualification.recovery)?
            != end.get("recovery").cloned().unwrap_or(Value::Null)
    {
        return Err(invalid("read lacks pinned export qualification"));
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
        "exported_by":concat!("polylogue ",env!("CARGO_PKG_VERSION")),"result":qualification.result});
    if let Some(recovery) = qualification.recovery {
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
    check(stop)?;
    if existing_export(&destination)? != previous {
        return Err(io::Error::other(
            "export destination changed during the read; previous directory was not replaced",
        ));
    }
    check(stop)?;
    // NOREPLACE also prevents a competing creation from being overwritten.
    // EXCHANGE leaves the previous directory at our private staged name until
    // the public rename is synced, so any sync failure retains both versions.
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
            let retained = stage.keep();
            return Err(io::Error::other(format!(
                "export rename raced with a destination change or its check failed: {old:?}; replaced directory retained at {}; inspect destination before retry",
                retained.display()
            )));
        }
    }
    if let Err(error) = parent_file.sync_all() {
        let retained = stage.keep();
        return Err(io::Error::other(format!(
            "export rename occurred but parent sync failed: {error}; staging name retained at {}; inspect destination before retry",
            retained.display()
        )));
    }
    // TempDir now owns the old export after exchange, or an absent name after
    // first publication. Cleanup cannot follow symlinks inside the old tree.
    Ok(end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{os::unix::net::UnixListener, sync::Arc};
    #[test]
    fn native_filename_is_byte_exact_and_cannot_make_path_components() {
        assert_eq!(filename("../a:b/λ%"), "..%2Fa%3Ab%2F%CE%BB%25.json");
        assert_eq!(filename(".~_-AZ09"), ".~_-AZ09.json");
    }
    fn exchange(
        document: Value,
        mut end: Value,
        cancel: bool,
        mutate: bool,
    ) -> (tempfile::TempDir, Value) {
        let work = tempfile::tempdir().unwrap();
        let destination = work.path().join("out");
        fs::create_dir(&destination).unwrap();
        fs::write(
            destination.join("manifest.json"),
            json!({"format":"polylogue-export-manifest","version":1}).to_string(),
        )
        .unwrap();
        fs::write(destination.join("old"), b"prior complete export").unwrap();
        let socket_owner = tempfile::tempdir_in("/tmp").unwrap();
        let socket = socket_owner.path().join("socket");
        let listener = UnixListener::bind(&socket).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let signal = Arc::clone(&stop);
        let changed_destination = destination.clone();
        let sender = std::thread::spawn(move || {
            use std::io::BufRead;
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = String::new();
            BufReader::new(stream.try_clone().unwrap())
                .read_line(&mut request)
                .unwrap();
            let request: Value = serde_json::from_str(&request).unwrap();
            assert_eq!(request["reference"], "test:../a:b/λ%");
            crate::reference_read::frame(&mut stream,json!({"type":"session","id":"test:../a:b/λ%","origin":"test","native_id":"../a:b/λ%","format":"polylogue-session","version":1})).unwrap();
            // The document's result can exceed one transport frame. Its payload
            // and qualification still travel through the ordinary text frames.
            crate::reference_read::text_frames(&mut stream, |out| {
                serde_json::to_writer(out, &document).map_err(io::Error::other)
            })
            .unwrap();
            if cancel {
                signal.store(true, Ordering::Release);
            }
            if mutate {
                fs::rename(
                    &changed_destination,
                    changed_destination.with_file_name("saved-previous"),
                )
                .unwrap();
                fs::create_dir(&changed_destination).unwrap();
                fs::write(
                    changed_destination.join("foreign"),
                    b"concurrently installed tree",
                )
                .unwrap();
            }
            end["type"] = json!("end");
            crate::reference_read::frame(&mut stream, end).unwrap();
        });
        let result = request(&socket, "test:../a:b/λ%", &destination, &stop);
        sender.join().unwrap();
        let summary = match result {
            Ok(value) => value,
            Err(error) => json!({"io_error":error.to_string()}),
        };
        (work, summary)
    }
    fn end() -> Value {
        json!({"outcome":"degraded","canonical_complete":false,"missing":["qualified scope"],"notes":["qualified scope"],"unknown":{},"view":"generation:17","up_to_date":false,"origin":"test","native_id":"../a:b/λ%"})
    }
    #[test]
    fn failed_or_cancelled_stream_cannot_replace_previous_directory() {
        for cancel in [false, true] {
            let end = if cancel {
                end()
            } else {
                json!({"outcome":"error","error":{"kind":"io","message":"injected failure"}})
            };
            let (work, reply) = exchange(
                json!({"result":{"outcome":"degraded","notes":["qualified scope"],"unknown":{}}}),
                end,
                cancel,
                false,
            );
            assert!(
                reply["outcome"] == "error"
                    || reply["io_error"]
                        .as_str()
                        .is_some_and(|s| s.contains("cancelled")),
                "{reply}"
            );
            assert_eq!(
                fs::read(work.path().join("out/old")).unwrap(),
                b"prior complete export"
            );
            assert_eq!(fs::read_dir(work.path()).unwrap().count(), 1);
        }
    }
    #[test]
    fn same_document_qualification_is_preserved_without_single_frame_limit() {
        let result = json!({"outcome":"degraded","notes":["qualified scope"],"unknown":{},"settlement":{"basis":"x".repeat(131073)}});
        let document = json!({"format":"polylogue-session","version":1,"result":result,"session":{"id":"test:../a:b/λ%"},"messages":[]});
        let (work, reply) = exchange(document.clone(), end(), false, false);
        assert_eq!(reply["outcome"], "degraded", "{reply}");
        let manifest: Value =
            serde_json::from_slice(&fs::read(work.path().join("out/manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest["result"], result);
        assert_eq!(manifest["up_to_date"], false);
        let detached: Value = serde_json::from_slice(
            &fs::read(work.path().join("out/test/..%2Fa%3Ab%2F%CE%BB%25.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(detached, document);
        assert!(!work.path().join("out/old").exists());
    }
    #[test]
    fn changed_destination_during_read_is_preserved_and_refused() {
        let document =
            json!({"result":{"outcome":"degraded","notes":["qualified scope"],"unknown":{}}});
        let (work, reply) = exchange(document, end(), false, true);
        assert!(reply["io_error"].is_string(), "{reply}");
        assert_eq!(
            fs::read(work.path().join("saved-previous/old")).unwrap(),
            b"prior complete export"
        );
        assert_eq!(
            fs::read(work.path().join("out/foreign")).unwrap(),
            b"concurrently installed tree"
        );
        assert!(!work.path().join("out/manifest.json").exists());
    }
}
