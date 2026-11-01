use super::*;
use crate::import::tests::engine;
use polylogue_testkit::ClaudeHome;

#[test]
fn only_explicit_local_images_are_collected() {
    let bytes = br#"{"payload":{"local_images":["/synthetic/image.png","relative.png"],"other":"/not-an-attachment","content":[{"type":"input_image","image_url":"file:///synthetic/two.png"}]}}"#;
    assert_eq!(
        references(bytes, Some("/synthetic")),
        [
            PathBuf::from("/synthetic/image.png"),
            PathBuf::from("/synthetic/relative.png"),
            PathBuf::from("/synthetic/two.png")
        ]
    );
}

#[test]
fn a_referenced_image_outside_the_tool_root_is_stored_and_rechecked() {
    let w = ClaudeHome::new();
    let image = w.dir.path().join("attached.png");
    let original = b"synthetic image bytes";
    std::fs::write(&image, original).unwrap();
    let dir = w.home.join(".codex/sessions");
    std::fs::create_dir_all(&dir).unwrap();
    let row = serde_json::json!({"type":"event_msg","payload":{"type":"user_message","message":"See attached","local_images":[image]}});
    std::fs::write(
        dir.join("rollout-images.jsonl"),
        format!("{{\"type\":\"session_meta\",\"payload\":{{\"id\":\"s1\"}}}}\n{row}\n"),
    )
    .unwrap();
    let (mut e, _lock) = engine(&w, 4096);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let id = e.location_id(&image);
    let reader = e.reads.reader().unwrap();
    let progress = reader.input_progress(&id).unwrap().unwrap();
    assert_eq!(
        reader
            .stored_range(&id, progress.chain, 0, progress.length, &e.blobs)
            .unwrap(),
        original
    );
    drop(reader);
    assert!(e.run_pass().unwrap().errors.is_empty());
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.input WHERE format='codex-local-file'"),
        1
    );
}
