use super::*;
use std::os::unix::net::UnixListener;
#[test]
fn native_id_filename_is_percent_encoded_and_has_no_path_separators() {
    assert_eq!(filename("../a:b/λ%"), "..%2Fa%3Ab%2F%CE%BB%25.json");
    assert_eq!(filename(".~_-AZ09"), ".~_-AZ09.json");
}
/// Runs an export against a fake daemon that sends `document` and `end`.
/// With `mutate`, the destination is replaced before the end frame.
fn exchange(document: Value, mut end: Value, mutate: bool) -> (tempfile::TempDir, Value) {
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
        crate::reference_read::text_frames(&mut stream, |out| {
            serde_json::to_writer(out, &document).map_err(io::Error::other)
        })
        .unwrap();
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
    let result = request(&socket, "test:../a:b/λ%", &destination);
    sender.join().unwrap();
    let summary = match result {
        Ok(value) => value,
        Err(error) => json!({"io_error":error.to_string()}),
    };
    (work, summary)
}
fn end() -> Value {
    json!({"outcome":"degraded","canonical_complete":false,"missing":["partial scope"],"notes":["partial scope"],"unknown":{},"view":"generation:17","up_to_date":false,"origin":"test","native_id":"../a:b/λ%"})
}
#[test]
fn failed_read_does_not_replace_previous_export() {
    let (work, reply) = exchange(
        json!({"result":{"outcome":"degraded","notes":["partial scope"],"unknown":{}}}),
        json!({"outcome":"error","error":{"kind":"io","message":"injected failure"}}),
        false,
    );
    assert_eq!(reply["outcome"], "error", "{reply}");
    assert_eq!(
        fs::read(work.path().join("out/old")).unwrap(),
        b"prior complete export"
    );
    assert_eq!(fs::read_dir(work.path()).unwrap().count(), 1);
}
#[test]
fn manifest_copies_a_document_result_larger_than_one_frame() {
    let result = json!({"outcome":"degraded","notes":["partial scope"],"unknown":{},"completion":{"computed_from":"x".repeat(131073)}});
    let document = json!({"format":"polylogue-session","version":1,"result":result,"session":{"id":"test:../a:b/λ%"},"messages":[]});
    let (work, reply) = exchange(document.clone(), end(), false);
    assert_eq!(reply["outcome"], "degraded", "{reply}");
    let manifest: Value =
        serde_json::from_slice(&fs::read(work.path().join("out/manifest.json")).unwrap()).unwrap();
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
fn destination_changed_during_read_is_left_alone() {
    let document = json!({"result":{"outcome":"degraded","notes":["partial scope"],"unknown":{}}});
    let (work, reply) = exchange(document, end(), true);
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
