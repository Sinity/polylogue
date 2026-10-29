use super::*;
use crate::import::tests::engine;
use polylogue_testkit::ClaudeHome;
#[test]
fn recorded_directory_captures_original_assets_and_reports_missing_and_unresolved_references() {
    let w = ClaudeHome::new();
    let cwd = w.dir.path().join("recorded");
    std::fs::create_dir_all(&cwd).unwrap();
    let image = cwd.join("image.png");
    std::fs::write(&image, b"original image").unwrap();
    let dir = w.home.join(".codex/sessions");
    std::fs::create_dir_all(&dir).unwrap();
    let row = serde_json::json!({"type":"event_msg","payload":{"type":"user_message","message":"attachment","local_images":["image.png","missing.png"]}});
    let header =
        serde_json::json!({"type":"session_meta","payload":{"id":"local-native","cwd":cwd}});
    std::fs::write(
        dir.join("rollout-images.jsonl"),
        format!("{header}\n{row}\n"),
    )
    .unwrap();
    let unresolved = serde_json::json!({"type":"event_msg","payload":{"local_images":["relative-without-cwd.png"]}});
    std::fs::write(
        dir.join("rollout-unknown.jsonl"),
        format!("{{\"type\":\"session_meta\",\"payload\":{{\"id\":\"no-cwd\"}}}}\n{unresolved}\n"),
    )
    .unwrap();
    let (mut e, _lock) = engine(&w, 64);
    let p = e.run_pass().unwrap();
    assert!(p.errors.is_empty(), "{p:?}");
    let id = blake3::hash(b"original image").to_hex().to_string();
    let r = e.reads.reader().unwrap();
    let progress = r.input_progress(&id).unwrap().unwrap();
    assert_eq!(
        r.stored_range(&id, progress.chain, 0, progress.length, &e.blobs)
            .unwrap(),
        b"original image"
    );
    let rows=r.query("SELECT DISTINCT o.input FROM a.observation o JOIN a.input i ON i.id=o.input WHERE i.format='tool-context' AND o.path LIKE 'codex-local-references/%'",&[]).unwrap();
    let mut statuses = Vec::new();
    for row in rows {
        let id = row["input"].as_str().unwrap();
        let p = r.input_progress(id).unwrap().unwrap();
        let bytes = r.stored_range(id, p.chain, 0, p.length, &e.blobs).unwrap();
        for line in bytes.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
            let value: Json = serde_json::from_slice(line).unwrap();
            statuses.push(value["status"].as_str().unwrap().to_owned());
        }
    }
    statuses.sort();
    assert_eq!(statuses, ["collected", "missing", "unresolved"]);
}
