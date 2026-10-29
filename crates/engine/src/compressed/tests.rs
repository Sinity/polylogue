use super::*;
use crate::import::tests::engine;
use polylogue_testkit::ClaudeHome;
use std::io::Write;

pub(super) fn rollout(native: &str) -> String {
    format!(
        "{{\"type\":\"session_meta\",\"payload\":{{\"id\":\"{native}\"}}}}\n{{\"type\":\"response_item\",\"payload\":{{\"id\":\"m1\",\"type\":\"message\",\"role\":\"user\",\"content\":[{{\"type\":\"input_text\",\"text\":\"independent payload\"}}]}}}}\n"
    )
}
fn captured(e: &Engine, id: &str) -> Vec<u8> {
    let r = e.reads.reader().unwrap();
    let p = r.input_progress(id).unwrap().unwrap();
    r.stored_range(id, p.chain, 0, p.length, &e.blobs).unwrap()
}

#[test]
fn plain_moved_compressed_and_reencoded_rollouts_share_native_input_without_extra_revisions() {
    let w = ClaudeHome::new();
    let dir = w.home.join(".codex/sessions");
    std::fs::create_dir_all(&dir).unwrap();
    let bytes = rollout("same-native");
    let plain = dir.join("rollout-first.jsonl");
    std::fs::write(&plain, &bytes).unwrap();
    let (mut e, lock) = engine(&w, 64);
    let p = e.run_pass().unwrap();
    assert!(p.errors.is_empty(), "{p:?}");
    assert_eq!(captured(&e, "codex:rollout:same-native"), bytes.as_bytes());
    let moved = w.home.join(".codex/archived_sessions");
    std::fs::create_dir_all(&moved).unwrap();
    std::fs::remove_file(&plain).unwrap();
    let archive = moved.join("rollout-moved.jsonl.zst");
    let encoded = zstd::stream::encode_all(bytes.as_bytes(), 1).unwrap();
    std::fs::write(&archive, &encoded).unwrap();
    let p = e.run_pass().unwrap();
    assert!(p.errors.is_empty(), "{p:?}");
    assert_eq!(
        captured(&e, &blake3::hash(&encoded).to_hex().to_string()),
        encoded
    );
    let encoded2 = zstd::stream::encode_all(bytes.as_bytes(), 9).unwrap();
    std::fs::write(&archive, &encoded2).unwrap();
    assert!(e.run_pass().unwrap().errors.is_empty());
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.revision WHERE input='codex:rollout:same-native'"),
        1
    );
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.observation_container WHERE container_input<>''"),
        2
    );
    drop(e);
    drop(lock);
    let (mut e, _lock) = engine(&w, 64);
    assert!(e.run_pass().unwrap().errors.is_empty());
    assert_eq!(captured(&e, "codex:rollout:same-native"), bytes.as_bytes());
}

#[test]
fn appended_plain_rollout_extends_the_native_chain_and_corrupt_encoding_keeps_original_uncommitted()
{
    let w = ClaudeHome::new();
    let dir = w.home.join(".codex/sessions");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("rollout-native.jsonl");
    let bytes = rollout("append-native");
    std::fs::write(&path, &bytes).unwrap();
    let (mut e, _lock) = engine(&w, 64);
    assert!(e.run_pass().unwrap().errors.is_empty());
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    file.write_all(b"{\"type\":\"event_msg\",\"payload\":{\"type\":\"turn_aborted\"}}\n")
        .unwrap();
    assert!(e.run_pass().unwrap().errors.is_empty());
    assert_eq!(w.count("SELECT count(DISTINCT chain) AS n FROM a.revision WHERE input='codex:rollout:append-native'"),1);
    std::fs::write(dir.join("rollout-corrupt.jsonl.zst"), b"not zstd").unwrap();
    let before = w.count("SELECT count(*) AS n FROM a.input");
    let p = e.run_pass().unwrap();
    assert_eq!(p.errors.len(), 1);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.input"), before);
}
