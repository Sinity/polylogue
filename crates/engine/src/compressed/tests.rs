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
    let sessions = polylogue_model::SessionId::parse("codex:same-native").unwrap();
    assert_eq!(
        e.reads
            .reader()
            .unwrap()
            .messages(&sessions, false)
            .unwrap()
            .iter()
            .map(polylogue_model::Message::text)
            .collect::<Vec<_>>(),
        ["independent payload"]
    );
    let moved = w.home.join(".codex/archived_sessions");
    std::fs::create_dir_all(&moved).unwrap();
    std::fs::remove_file(&plain).unwrap();
    let archive = moved.join("rollout-moved.jsonl.zst");
    let encoded = zstd::stream::encode_all(bytes.as_bytes(), 1).unwrap();
    std::fs::write(&archive, &encoded).unwrap();
    let p = e.run_pass().unwrap();
    assert!(p.errors.is_empty(), "{p:?}");
    assert_eq!(
        captured(&e, blake3::hash(&encoded).to_hex().as_ref()),
        encoded
    );
    let encoded2 = zstd::stream::encode_all(bytes.as_bytes(), 9).unwrap();
    std::fs::write(&archive, &encoded2).unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
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
    std::fs::remove_file(w.root.join("index.db")).unwrap();
    let (mut e, _lock) = engine(&w, 64);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
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
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    file.write_all(b"{\"type\":\"event_msg\",\"payload\":{\"type\":\"turn_aborted\"}}\n")
        .unwrap();
    let appended = e.run_pass().unwrap();
    assert!(appended.errors.is_empty(), "{appended:?}");
    assert_eq!(
        appended.bytes_stored,
        b"{\"type\":\"event_msg\",\"payload\":{\"type\":\"turn_aborted\"}}\n".len() as u64
    );
    assert_eq!(w.count("SELECT count(DISTINCT chain) AS n FROM a.revision WHERE input='codex:rollout:append-native'"),1);
    std::fs::write(dir.join("rollout-corrupt.jsonl.zst"), b"not zstd").unwrap();
    let before = w.count("SELECT count(*) AS n FROM a.input");
    let p = e.run_pass().unwrap();
    assert_eq!(p.errors.len(), 1);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.input"), before);
}

#[test]
fn a_large_compressed_record_has_no_decoded_size_cap() {
    let w = ClaudeHome::new();
    let dir = w.home.join(".codex/sessions");
    std::fs::create_dir_all(&dir).unwrap();
    let header=serde_json::json!({"type":"session_meta","payload":{"id":"large-native","instructions":"x".repeat(17*1024*1024)}}).to_string()+"\n";
    let encoded = zstd::stream::encode_all(header.as_bytes(), 1).unwrap();
    std::fs::write(dir.join("rollout-large.jsonl.zst"), encoded).unwrap();
    let (mut e, _lock) = engine(&w, 1 << 20);
    let p = e.run_pass().unwrap();
    assert!(p.errors.is_empty(), "{p:?}");
    assert_eq!(
        captured(&e, "codex:rollout:large-native"),
        header.as_bytes()
    );
}

#[test]
fn a_proven_stale_alias_observes_its_old_revision_without_regressing_the_native_input() {
    let w = ClaudeHome::new();
    let sessions = w.home.join(".codex/sessions");
    let archived = w.home.join(".codex/archived_sessions");
    std::fs::create_dir_all(&sessions).unwrap();
    std::fs::create_dir_all(&archived).unwrap();
    let path = sessions.join("rollout-live.jsonl");
    let original = rollout("alias-native");
    std::fs::write(&path, &original).unwrap();
    let (mut e, lock) = engine(&w, 64);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let extra=b"{\"type\":\"response_item\",\"payload\":{\"id\":\"m2\",\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"newer native tail\"}]}}\n";
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(extra)
        .unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let copy = archived.join("rollout-stale.jsonl.zst");
    std::fs::write(
        &copy,
        zstd::stream::encode_all(original.as_bytes(), 1).unwrap(),
    )
    .unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.revision WHERE input='codex:rollout:alias-native'"),
        2
    );
    let rows=e.reads.reader().unwrap().query("SELECT revision,length FROM a.observation WHERE input='codex:rollout:alias-native' ORDER BY rowid DESC LIMIT 1",&[]).unwrap();
    assert_eq!(rows[0]["revision"], 1);
    assert_eq!(rows[0]["length"], original.len() as u64);
    std::fs::write(
        &copy,
        zstd::stream::encode_all(original.as_bytes(), 9).unwrap(),
    )
    .unwrap();
    drop(e);
    drop(lock);
    let (mut e, _lock) = engine(&w, 64);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.revision WHERE input='codex:rollout:alias-native'"),
        2
    );
    let id = polylogue_model::SessionId::parse("codex:alias-native").unwrap();
    assert_eq!(
        e.reads
            .reader()
            .unwrap()
            .messages(&id, false)
            .unwrap()
            .iter()
            .map(polylogue_model::Message::text)
            .collect::<Vec<_>>(),
        ["independent payload", "newer native tail"]
    );
    std::fs::write(
        &path,
        original.replace("independent payload", "genuine divergent rewrite"),
    )
    .unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(w.count("SELECT count(DISTINCT chain) AS n FROM a.revision WHERE input='codex:rollout:alias-native'"),2);
}

#[test]
fn equal_decoded_acquisition_keeps_its_actual_body_without_an_extra_revision() {
    let w = ClaudeHome::new();
    let dir = w.home.join(".codex/sessions");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("rollout-body.jsonl");
    let original = rollout("body-native");
    std::fs::write(&path, &original).unwrap();
    let (mut e, _lock) = engine(&w, 64);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let tail = b"{\"type\":\"event_msg\",\"payload\":{\"type\":\"turn_aborted\"}}\n";
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(tail)
        .unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let whole: Vec<u8> = original.bytes().chain(tail.iter().copied()).collect();
    let hash = hashed_blobs::Hash(*blake3::hash(&whole).as_bytes());
    // This chain comprises two blobs; its content hash is not a Blob address.
    assert!(e.blobs.extent(&hash).is_err());
    let compressed = dir.join("rollout-body-copy.jsonl.zst");
    std::fs::write(
        &compressed,
        zstd::stream::encode_all(whole.as_slice(), 1).unwrap(),
    )
    .unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let mut body = Vec::new();
    e.blobs
        .read_plain(&hash, Some(whole.len() as u64), || Ok(()))
        .unwrap()
        .read_to_end(&mut body)
        .unwrap();
    assert_eq!(body, whole);
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.revision WHERE input='codex:rollout:body-native'"),
        2
    );
}

fn frames(parts: &[&[u8]]) -> Vec<u8> {
    parts
        .iter()
        .flat_map(|part| zstd::stream::encode_all(*part, 3).unwrap())
        .collect()
}

#[test]
fn codex_compressing_a_plain_rollout_in_place_keeps_one_input_and_its_newer_tail() {
    let w = ClaudeHome::new();
    let dir = w.home.join(".codex/sessions/2026/10/17");
    std::fs::create_dir_all(&dir).unwrap();
    let plain = dir.join("rollout-2026-10-17T07-27-03-in-place.jsonl");
    let first = rollout("in-place-native");
    std::fs::write(&plain, &first).unwrap();
    let (mut e, _lock) = engine(&w, 64);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let tail = b"{\"type\":\"response_item\",\"payload\":{\"id\":\"m2\",\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"written before compression\"}]}}\n";
    let whole: Vec<u8> = first.bytes().chain(tail.iter().copied()).collect();
    let encoded = zstd::stream::encode_all(whole.as_slice(), 3).unwrap();
    std::fs::write(plain.with_extension("jsonl.zst"), &encoded).unwrap();
    std::fs::remove_file(&plain).unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(captured(&e, "codex:rollout:in-place-native"), whole);
    assert_eq!(
        captured(&e, blake3::hash(&encoded).to_hex().as_ref()),
        encoded
    );
    assert_eq!(
        w.count("SELECT count(DISTINCT chain) AS n FROM a.revision WHERE input='codex:rollout:in-place-native'"),
        1
    );
    let id = polylogue_model::SessionId::parse("codex:in-place-native").unwrap();
    assert_eq!(
        e.reads
            .reader()
            .unwrap()
            .messages(&id, false)
            .unwrap()
            .iter()
            .map(polylogue_model::Message::text)
            .collect::<Vec<_>>(),
        ["independent payload", "written before compression"]
    );
}

#[test]
fn a_rollout_written_as_several_zstd_frames_decodes_across_every_frame() {
    let w = ClaudeHome::new();
    let dir = w.home.join(".codex/sessions/2026/10/18");
    std::fs::create_dir_all(&dir).unwrap();
    let head = rollout("multi-frame-native");
    let tail = b"{\"type\":\"response_item\",\"payload\":{\"id\":\"m2\",\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"second frame\"}]}}\n";
    let encoded = frames(&[head.as_bytes(), tail]);
    std::fs::write(dir.join("rollout-multi-frame.jsonl.zst"), &encoded).unwrap();
    let (mut e, _lock) = engine(&w, 64);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let whole: Vec<u8> = head.bytes().chain(tail.iter().copied()).collect();
    assert_eq!(captured(&e, "codex:rollout:multi-frame-native"), whole);
    assert_eq!(
        captured(&e, blake3::hash(&encoded).to_hex().as_ref()),
        encoded
    );
}

#[test]
fn a_truncated_compressed_rollout_stores_nothing_until_the_whole_file_arrives() {
    let w = ClaudeHome::new();
    let dir = w.home.join(".codex/sessions/2026/10/19");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("rollout-truncated.jsonl.zst");
    let bytes = rollout("truncated-native");
    let encoded = zstd::stream::encode_all(bytes.as_bytes(), 3).unwrap();
    std::fs::write(&path, &encoded[..encoded.len() - 4]).unwrap();
    let (mut e, _lock) = engine(&w, 64);
    let before = w.count("SELECT count(*) AS n FROM a.input");
    let pass = e.run_pass().unwrap();
    assert_eq!(pass.errors.len(), 1, "{pass:?}");
    assert_eq!(w.count("SELECT count(*) AS n FROM a.input"), before);
    std::fs::write(&path, &encoded).unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(
        captured(&e, "codex:rollout:truncated-native"),
        bytes.as_bytes()
    );
}

#[test]
fn rollout_without_session_meta_grows_one_input() {
    let w = ClaudeHome::new();
    let dir = w.home.join(".codex/sessions");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("rollout-growing.jsonl");
    let complete = rollout("grow-native");
    // Codex has created the file and not finished its first record.
    std::fs::write(&path, &complete.as_bytes()[..20]).unwrap();
    let (mut e, _lock) = engine(&w, 64);
    let inputs = |w: &ClaudeHome| w.count("SELECT count(*) AS n FROM a.input");
    let before = inputs(&w);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(inputs(&w), before, "a rollout without its session ID waits");
    std::fs::write(&path, &complete).unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(
        captured(&e, "codex:rollout:grow-native"),
        complete.as_bytes()
    );
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    file.write_all(b"{\"type\":\"event_msg\",\"payload\":{\"type\":\"turn_aborted\"}}\n")
        .unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(inputs(&w), before + 1);
    assert_eq!(
        w.count("SELECT count(DISTINCT chain) AS n FROM a.revision WHERE input='codex:rollout:grow-native'"),
        1
    );
}

#[test]
fn location_input_id_matches_documented_bytes() {
    // archive.md and archive.sql: the first 32 hex digits of
    // BLAKE3(machine, a zero byte, the path's bytes).
    let path = std::path::Path::new("/home/op/.claude/projects/p/s.jsonl");
    let expected = blake3::hash(b"synthetic-machine\0/home/op/.claude/projects/p/s.jsonl").to_hex();
    assert_eq!(
        crate::location_id("synthetic-machine", path),
        &expected.as_str()[..32]
    );
}

#[test]
fn a_file_reached_through_a_linked_directory_is_one_location() {
    let dir = tempfile::tempdir().unwrap();
    let real = dir.path().join("config/claude/projects/p");
    std::fs::create_dir_all(&real).unwrap();
    std::fs::write(real.join("s.jsonl"), b"{}\n").unwrap();
    std::os::unix::fs::symlink(dir.path().join("config/claude"), dir.path().join("linked"))
        .unwrap();
    let through_link = dir.path().join("linked/projects/p/s.jsonl");
    let direct = real.join("s.jsonl");
    assert_eq!(
        crate::location_id("synthetic-machine", &through_link),
        crate::location_id("synthetic-machine", &direct)
    );
    // Once deleted, the file keeps its ID: only its directory is resolved.
    std::fs::remove_file(&direct).unwrap();
    assert_eq!(
        crate::location_id("synthetic-machine", &through_link),
        crate::location_id("synthetic-machine", &direct)
    );
}
