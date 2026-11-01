use super::*;
use crate::import::tests::engine;
use polylogue_testkit::ClaudeHome;

fn rollout() -> Vec<u8> {
    concat!(
        "{\"type\":\"session_meta\",\"payload\":{\"id\":\"thread-one\",\"cwd\":\"/synthetic\"}}\n",
        "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"id\":\"m1\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"A stored prompt\"}]}}\n"
    ).as_bytes().to_vec()
}

fn paths(w: &ClaudeHome) -> (PathBuf, PathBuf) {
    let dir = w.home.join(".codex/sessions/2026/10/27");
    std::fs::create_dir_all(&dir).unwrap();
    (
        dir.join("rollout-one.jsonl"),
        dir.join("rollout-one.jsonl.zst"),
    )
}

fn verify_messages(e: &Engine) {
    let id = polylogue_model::SessionId::parse("codex:thread-one").unwrap();
    let messages = e.reads.reader().unwrap().messages(&id, false).unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].text(), "A stored prompt");
}

#[test]
fn compressed_rollouts_keep_original_bytes_and_rebuild_without_provider_files() {
    let w = ClaudeHome::new();
    let (_, compressed) = paths(&w);
    let original = zstd::stream::encode_all(rollout().as_slice(), 1).unwrap();
    std::fs::write(&compressed, &original).unwrap();
    for rebuild in [false, true] {
        if rebuild {
            std::fs::remove_file(&compressed).unwrap();
            polylogue_testkit::delete_index(&w.root);
        }
        let (mut e, _lock) = engine(&w, 64);
        let pass = e.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        verify_messages(&e);
        let reader = e.reads.reader().unwrap();
        let id = e.location_id(&compressed);
        let stored = reader.input_progress(&id).unwrap().unwrap();
        assert_eq!(
            reader
                .stored_range(&id, stored.chain, 0, stored.length, &e.blobs)
                .unwrap(),
            original
        );
    }
}

#[test]
fn compressing_a_plain_rollout_keeps_its_input_revision_and_message_version() {
    let w = ClaudeHome::new();
    let (plain, compressed) = paths(&w);
    std::fs::write(&plain, rollout()).unwrap();
    {
        let (mut e, _lock) = engine(&w, 64);
        assert!(e.run_pass().unwrap().errors.is_empty());
        verify_messages(&e);
    }
    let encoded = zstd::stream::encode_all(rollout().as_slice(), 3).unwrap();
    std::fs::write(&compressed, &encoded).unwrap();
    std::fs::remove_file(&plain).unwrap();
    for restart in [false, true] {
        let (mut e, _lock) = engine(&w, 64);
        let pass = e.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        verify_messages(&e);
        let again = e.run_pass().unwrap();
        assert!(again.errors.is_empty(), "{:?}", again.errors);
        assert_eq!(again.inputs_changed, 0);
        assert_eq!(
            w.count("SELECT count(*) AS n FROM a.input WHERE format='codex'"),
            1
        );
        assert_eq!(w.count("SELECT count(*) AS n FROM a.revision r JOIN a.input i ON i.id=r.input WHERE i.format='codex'"), 1);
        assert_eq!(w.count("SELECT count(*) AS n FROM message_source"), 1);
        if !restart {
            std::fs::write(
                &compressed,
                zstd::stream::encode_all(rollout().as_slice(), 9).unwrap(),
            )
            .unwrap();
        }
    }
}

#[test]
fn an_invalid_compressed_rollout_is_visible_in_capture_health() {
    let w = ClaudeHome::new();
    let (_, compressed) = paths(&w);
    std::fs::write(&compressed, b"not zstd").unwrap();
    let (mut e, _lock) = engine(&w, 64);
    let pass = e.run_pass().unwrap();
    assert!(
        pass.capture_health
            .iter()
            .any(|r| r.rejected.contains(&compressed))
    );
}
