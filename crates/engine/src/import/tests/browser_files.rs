use super::*;
use polylogue_model::{BlobHash, SessionId};
use polylogue_testkit::ClaudeHome;
use serde_json::json;

fn capture(file: serde_json::Value, at: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({"polylogue_capture_kind":"browser_llm_session", "schema_version":1,
        "provenance":{"captured_at":at}, "session":{"provider":"gemini", "provider_session_id":"files",
        "turns":[{"provider_turn_id":"m1", "role":"user", "text":"an attachment"}],
        "attachments":[file]}})).unwrap()
}

fn append(path: &Path, bytes: &[u8]) {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    if file.metadata().unwrap().len() == 0 {
        file.write_all(polylogue_model::capture_log::MAGIC).unwrap();
    }
    file.write_all(&polylogue_model::capture_log::Header::of(bytes).encode())
        .unwrap();
    file.write_all(bytes).unwrap();
}

#[test]
fn pending_files_live_in_the_daemon_and_known_hashes_can_be_reused() {
    let w = ClaudeHome::new();
    let path = w
        .config(64 << 20)
        .spool_dir("browser-capture")
        .join("gemini/files.json");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let session = SessionId::parse("gemini-app:files").unwrap();
    let file = json!({"provider_attachment_id":"file1", "message_provider_id":"m1", "name":"file.txt", "not_fetched":"downloading"});
    append(&path, &capture(file, "2026-10-27T00:00:00Z"));
    let (mut engine, lock) = engine(&w, 64 << 20);
    let report = engine.run_pass().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    assert!(
        engine
            .reads
            .reader()
            .unwrap()
            .downloading_files(&session)
            .unwrap()
    );
    drop(engine);
    drop(lock);
    let (mut engine, _lock) = super::engine(&w, 64 << 20);
    assert!(
        engine
            .reads
            .reader()
            .unwrap()
            .downloading_files(&session)
            .unwrap()
    );
    let file = json!({"provider_attachment_id":"file1", "message_provider_id":"m1", "content_base64":"aGVsbG8="});
    append(&path, &capture(file, "2026-10-27T00:00:00Z"));
    let report = engine.run_pass().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    let reader = engine.reads.reader().unwrap();
    assert!(!reader.downloading_files(&session).unwrap());
    let files = reader.stored_files(&session).unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].blob, BlobHash::of(b"hello"));
    drop(reader);
    let file = json!({"provider_attachment_id":"file1", "message_provider_id":"m1", "blob":BlobHash::of(b"hello").to_string()});
    append(&path, &capture(file, "2026-10-27T00:01:00Z"));
    let report = engine.run_pass().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    assert_eq!(
        engine
            .reads
            .reader()
            .unwrap()
            .stored_files(&session)
            .unwrap(),
        files
    );
}

#[test]
fn an_unknown_hash_is_refused_before_any_attachment_names_it() {
    let w = ClaudeHome::new();
    let (mut engine, _lock) = engine(&w, 64 << 20);
    let path = w.dir.path().join("capture.json");
    let file = json!({"provider_attachment_id":"file1", "message_provider_id":"m1", "blob":BlobHash::of(b"absent").to_string()});
    std::fs::write(&path, capture(file, "2026-10-27T00:00:00Z")).unwrap();
    let result = engine.import(std::slice::from_ref(&path));
    let error = match result {
        Ok(report) => format!("{:?}", report.pass.errors),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("unknown blob"), "{error}");
    assert!(
        engine
            .reads
            .reader()
            .unwrap()
            .stored_files(&SessionId::parse("gemini-app:files").unwrap())
            .unwrap()
            .is_empty()
    );
}
