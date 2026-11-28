use super::*;
use crate::import::tests::engine;
use polylogue_testkit::ClaudeHome;
use std::io::Write;

pub(super) fn rollout(session_id: &str) -> String {
    format!(
        "{{\"type\":\"session_meta\",\"payload\":{{\"id\":\"{session_id}\"}}}}\n{{\"type\":\"response_item\",\"payload\":{{\"id\":\"m1\",\"type\":\"message\",\"role\":\"user\",\"content\":[{{\"type\":\"input_text\",\"text\":\"independent payload\"}}]}}}}\n"
    )
}
fn captured(e: &Engine, id: &str) -> Vec<u8> {
    let r = e.reads.reader().unwrap();
    let p = r.latest_chain(id).unwrap().unwrap();
    r.stored_range(id, p.chain, 0, p.stored, &e.blobs).unwrap()
}

#[test]
fn moved_and_reencoded_rollouts_share_one_input() {
    let w = ClaudeHome::new();
    let dir = w.home.join(".codex/sessions");
    std::fs::create_dir_all(&dir).unwrap();
    let bytes = rollout("same-native");
    let plain = dir.join("rollout-first.jsonl");
    std::fs::write(&plain, &bytes).unwrap();
    let (mut e, lock) = engine(&w, 64);
    let p = e.run_pass().unwrap();
    assert!(p.errors.is_empty(), "{p:?}");
    assert_eq!(
        captured(&e, "codex:rollout:same-native@test-machine"),
        bytes.as_bytes()
    );
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
        w.count("SELECT count(*) AS n FROM a.revision WHERE input='codex:rollout:same-native@test-machine'"),
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
    assert_eq!(
        captured(&e, "codex:rollout:same-native@test-machine"),
        bytes.as_bytes()
    );
}

#[test]
fn appended_rollout_extends_chain_but_corrupt_does_not() {
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
    assert_eq!(w.count("SELECT count(DISTINCT chain) AS n FROM a.revision WHERE input='codex:rollout:append-native@test-machine'"),1);
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
        captured(&e, "codex:rollout:large-native@test-machine"),
        header.as_bytes()
    );
}

#[test]
fn stale_alias_keeps_old_revision_and_the_input() {
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
        w.count("SELECT count(*) AS n FROM a.revision WHERE input='codex:rollout:alias-native@test-machine'"),
        2
    );
    let rows=e.reads.reader().unwrap().query("SELECT revision,length FROM a.observation WHERE input='codex:rollout:alias-native@test-machine' ORDER BY id DESC LIMIT 1",&[]).unwrap();
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
        w.count("SELECT count(*) AS n FROM a.revision WHERE input='codex:rollout:alias-native@test-machine'"),
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
    assert_eq!(w.count("SELECT count(DISTINCT chain) AS n FROM a.revision WHERE input='codex:rollout:alias-native@test-machine'"),2);
}

#[test]
fn equal_decoded_read_keeps_its_body_without_a_revision() {
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
    assert!(matches!(
        e.blobs.extent(&hash),
        Err(polylogue_store::StoreError::NotFound(_))
    ));
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
        .read_plain(&hash, whole.len() as u64, || Ok(()))
        .unwrap()
        .read_to_end(&mut body)
        .unwrap();
    assert_eq!(body, whole);
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.revision WHERE input='codex:rollout:body-native@test-machine'"),
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
    assert_eq!(
        captured(&e, "codex:rollout:in-place-native@test-machine"),
        whole
    );
    assert_eq!(
        captured(&e, blake3::hash(&encoded).to_hex().as_ref()),
        encoded
    );
    assert_eq!(
        w.count("SELECT count(DISTINCT chain) AS n FROM a.revision WHERE input='codex:rollout:in-place-native@test-machine'"),
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
    assert_eq!(
        captured(&e, "codex:rollout:multi-frame-native@test-machine"),
        whole
    );
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
        captured(&e, "codex:rollout:truncated-native@test-machine"),
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
        captured(&e, "codex:rollout:grow-native@test-machine"),
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
        w.count("SELECT count(DISTINCT chain) AS n FROM a.revision WHERE input='codex:rollout:grow-native@test-machine'"),
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

/// A rollout of `answers` assistant answers after its session header.
fn long_rollout(session_id: &str, answers: usize) -> String {
    let mut bytes = rollout(session_id);
    for i in 0..answers {
        bytes.push_str(&answer(&format!("a{i}"), &format!("answer {i}")));
    }
    bytes
}

fn answer(id: &str, text: &str) -> String {
    format!(
        "{{\"type\":\"response_item\",\"payload\":{{\"id\":\"{id}\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{{\"type\":\"output_text\",\"text\":\"{text}\"}}]}}}}\n"
    )
}

fn texts(e: &Engine, session: &str) -> Vec<String> {
    e.reads
        .reader()
        .unwrap()
        .messages(&polylogue_model::SessionId::parse(session).unwrap(), false)
        .unwrap()
        .iter()
        .map(polylogue_model::Message::text)
        .collect()
}

/// Compare the stored prefix once, then store and parse only the new
/// records. The resulting index agrees with reading the whole file
/// (docs/architecture.md, the index is a function of the record).
#[test]
fn an_append_to_a_stored_rollout_checks_its_prefix_and_parses_only_new_bytes() {
    let w = ClaudeHome::new();
    let dir = w.home.join(".codex/sessions");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("rollout-2026-03-01T10-00-00-019a0000-0000-7000-8000-00000000000a.jsonl");
    let first = long_rollout("append-cost", 400);
    std::fs::write(&path, &first).unwrap();
    let (mut e, _lock) = engine(&w, 1 << 20);
    let stored = e.run_pass().unwrap();
    assert!(stored.errors.is_empty(), "{stored:?}");
    let appended = answer("late", "written later") + &answer("later", "written last");
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(appended.as_bytes())
        .unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(pass.bytes_stored, appended.len() as u64);
    assert_eq!(pass.messages_written, 2);
    assert!(
        (first.len() as u64..=(first.len() + 2 * appended.len()) as u64).contains(&pass.bytes_read),
        "read {} bytes for an append of {} to a file of {}",
        pass.bytes_read,
        appended.len(),
        first.len() + appended.len()
    );
    assert_eq!(
        w.count("SELECT count(DISTINCT chain) AS n FROM a.revision WHERE input='codex:rollout:append-cost@test-machine'"),
        1
    );
    let whole = ClaudeHome::new();
    let whole_dir = whole.home.join(".codex/sessions");
    std::fs::create_dir_all(&whole_dir).unwrap();
    std::fs::copy(&path, whole_dir.join(path.file_name().unwrap())).unwrap();
    let (mut once, _whole_lock) = engine(&whole, 1 << 20);
    let read_once = once.run_pass().unwrap();
    assert!(read_once.errors.is_empty(), "{read_once:?}");
    let session = "codex:append-cost";
    assert_eq!(texts(&e, session).len(), 403);
    assert_eq!(texts(&e, session), texts(&once, session));
    assert_eq!(w.export(session), whole.export(session));
}

/// A stored rollout changed before its old end and then grown is not taken
/// as an append: it starts a chain of its own.
#[test]
fn a_rollout_rewritten_in_place_and_grown_is_not_an_append() {
    rewritten_rollout(false);
}

#[test]
fn a_longer_rollout_copy_with_an_early_edit_is_not_an_append() {
    rewritten_rollout(true);
}

fn rewritten_rollout(moved: bool) {
    let w = ClaudeHome::new();
    let dir = w.home.join(".codex/sessions");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("rollout-2026-03-01T10-00-00-019a0000-0000-7000-8000-00000000000b.jsonl");
    let first = long_rollout("rewrite-prefix", 80);
    std::fs::write(&path, &first).unwrap();
    let (mut e, _lock) = engine(&w, 1 << 20);
    let stored = e.run_pass().unwrap();
    assert!(stored.errors.is_empty(), "{stored:?}");
    let rewritten = first.replacen("answer 0", "ANSWER 0", 1) + &answer("late", "written later");
    assert_eq!(
        &first[first.len() - 4096..],
        &rewritten[first.len() - 4096..first.len()]
    );
    let input = "codex:rollout:rewrite-prefix@test-machine";
    assert_eq!(captured(&e, input), first.as_bytes());
    if moved {
        std::fs::remove_file(&path).unwrap();
        std::fs::write(dir.join("rollout-copy.jsonl"), &rewritten).unwrap();
    } else {
        std::fs::write(&path, &rewritten).unwrap();
    }
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(
        w.count("SELECT count(DISTINCT chain) AS n FROM a.revision WHERE input='codex:rollout:rewrite-prefix@test-machine'"),
        2
    );
    let reader = e.reads.reader().unwrap();
    assert_eq!(
        reader
            .stored_range(input, 1, 0, first.len() as u64, &e.blobs)
            .unwrap(),
        first.as_bytes()
    );
    drop(reader);
    assert_eq!(captured(&e, input), rewritten.as_bytes());
    let texts = texts(&e, "codex:rewrite-prefix");
    assert_eq!(texts.len(), 82);
    assert_eq!(texts[1], "ANSWER 0");
    assert_eq!(texts[81], "written later");
}

/// No two machines write chains of one input: every origin names its inputs
/// by the machine and the location, by its own ID with the machine, or by
/// the bytes alone (docs/archive.md, stored inputs).
#[test]
fn every_input_identity_contains_its_machine_or_is_its_content_hash() {
    use polylogue_sources::origin::{InputIdentity, ORIGINS};
    let path = std::path::Path::new("/home/op/.codex/sessions/rollout.jsonl");
    for origin in ORIGINS {
        match origin.input_identity {
            InputIdentity::Location => assert_ne!(
                crate::location_id("machine-one", path),
                crate::location_id("machine-two", path),
                "{}",
                origin.name
            ),
            InputIdentity::ProviderId(_) => {
                let one = InputIdentity::provider_named("codex:rollout:session", "machine-one");
                let two = InputIdentity::provider_named("codex:rollout:session", "machine-two");
                assert_ne!(one, two, "{}", origin.name);
                assert!(one.ends_with("@machine-one"), "{one}");
            }
            InputIdentity::Content => {}
        }
    }
}

/// A stored rollout read at its first record, then grown by two.
fn stored_then_grown(
    w: &ClaudeHome,
    native: &str,
) -> (Engine, polylogue_store::ArchiveLock, Vec<String>) {
    let sessions = w.home.join(".codex/sessions");
    std::fs::create_dir_all(&sessions).unwrap();
    std::fs::create_dir_all(w.home.join(".codex/archived_sessions")).unwrap();
    let path = sessions.join("rollout-live.jsonl");
    let first = rollout(native);
    std::fs::write(&path, &first).unwrap();
    let (mut e, lock) = engine(w, 64);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let turns: Vec<String> = ["second turn", "third turn"]
        .iter()
        .enumerate()
        .map(|(i, text)| format!("{{\"type\":\"response_item\",\"payload\":{{\"id\":\"m{}\",\"type\":\"message\",\"role\":\"user\",\"content\":[{{\"type\":\"input_text\",\"text\":\"{text}\"}}]}}}}\n", i + 2))
        .collect();
    let mut live = std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap();
    for turn in &turns {
        live.write_all(turn.as_bytes()).unwrap();
    }
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    (e, lock, [vec![first], turns].concat())
}

fn session_texts(e: &Engine, native: &str) -> Vec<String> {
    let id = polylogue_model::SessionId::parse(&format!("codex:{native}")).unwrap();
    e.reads
        .reader()
        .unwrap()
        .messages(&id, false)
        .unwrap()
        .iter()
        .map(polylogue_model::Message::text)
        .collect()
}

/// A compressed copy of a rollout holding only the first bytes of the chain
/// already stored, ending at no revision's length, is the stored chain seen
/// back with what it held: it starts no chain and no revision, and the
/// session keeps every message.
#[test]
fn a_shorter_compressed_copy_is_seen_back_and_starts_no_chain() {
    let w = ClaudeHome::new();
    let (mut e, _lock, parts) = stored_then_grown(&w, "shorter-native");
    let stale = format!("{}{}", parts[0], parts[1]);
    std::fs::write(
        w.home
            .join(".codex/archived_sessions/rollout-stale.jsonl.zst"),
        zstd::stream::encode_all(stale.as_bytes(), 1).unwrap(),
    )
    .unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let input = "codex:rollout:shorter-native@test-machine";
    assert_eq!(
        w.count(&format!(
            "SELECT count(DISTINCT chain) AS n FROM a.revision WHERE input='{input}'"
        )),
        1
    );
    assert_eq!(
        w.count(&format!(
            "SELECT count(*) AS n FROM a.revision WHERE input='{input}'"
        )),
        2
    );
    assert_eq!(
        w.count(&format!(
            "SELECT count(*) AS n FROM a.observation WHERE input='{input}' AND class='back'
               AND revision IS NULL AND chain=1 AND length={}",
            stale.len()
        )),
        1
    );
    assert_eq!(captured(&e, input), parts.concat().into_bytes());
    assert_eq!(
        session_texts(&e, "shorter-native"),
        ["independent payload", "second turn", "third turn"]
    );
    // The copy written again whole is the stored chain at its length: still
    // one chain, the same current revision, nothing rewritten.
    std::fs::write(
        w.home
            .join(".codex/archived_sessions/rollout-stale.jsonl.zst"),
        zstd::stream::encode_all(parts.concat().as_bytes(), 1).unwrap(),
    )
    .unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(
        w.count(&format!(
            "SELECT count(*) AS n FROM a.revision WHERE input='{input}'"
        )),
        2
    );
    assert_eq!(
        w.count(&format!(
            "SELECT count(*) AS n FROM a.revision WHERE input='{input}' AND produced='rewritten'"
        )),
        0
    );
    assert_eq!(
        session_texts(&e, "shorter-native"),
        ["independent payload", "second turn", "third turn"]
    );
}

/// A shorter copy that differs from the stored chain in one byte inside the
/// stored length is another reading of the rollout: it starts a chain.
#[test]
fn a_shorter_copy_differing_inside_the_stored_length_starts_a_chain() {
    let w = ClaudeHome::new();
    let (mut e, _lock, parts) = stored_then_grown(&w, "differing-native");
    let stale = format!(
        "{}{}",
        parts[0],
        parts[1].replace("second turn", "SECOND turn")
    );
    std::fs::write(
        w.home
            .join(".codex/archived_sessions/rollout-stale.jsonl.zst"),
        zstd::stream::encode_all(stale.as_bytes(), 1).unwrap(),
    )
    .unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(
        w.count("SELECT count(DISTINCT chain) AS n FROM a.revision WHERE input='codex:rollout:differing-native@test-machine'"),
        2
    );
}

/// A compressed copy that decodes to nothing names no session (a rollout
/// is named by its first record), so nothing is stored for it and the
/// stored chain stays current.
#[test]
fn an_empty_compressed_copy_stores_nothing() {
    let w = ClaudeHome::new();
    let (mut e, _lock, parts) = stored_then_grown(&w, "empty-native");
    std::fs::write(
        w.home
            .join(".codex/archived_sessions/rollout-empty.jsonl.zst"),
        zstd::stream::encode_all(&b""[..], 1).unwrap(),
    )
    .unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let input = "codex:rollout:empty-native@test-machine";
    assert_eq!(
        w.count(&format!(
            "SELECT count(DISTINCT chain) AS n FROM a.revision WHERE input='{input}'"
        )),
        1
    );
    assert_eq!(captured(&e, input), parts.concat().into_bytes());
    assert_eq!(
        session_texts(&e, "empty-native"),
        ["independent payload", "second turn", "third turn"]
    );
}
