use super::*;
use polylogue_store::fault::{self, Action};

/// While a build prepares, its progress counts the chains and bytes it
/// found due a reading and those it has read whole.
#[test]
fn build_progress_counts_chains_read_whole() {
    let home = ClaudeHome::new();
    let first = claude_user("first", "progress", "first chunk");
    let second = claude_user("second", "progress", "second chunk");
    home.append(
        "progress.jsonl",
        &[first.as_str(), second.as_str()].concat(),
    );
    {
        let (mut engine, _lock) = home.engine(1 << 20);
        assert!(engine.run_pass().unwrap().errors.is_empty());
    }
    polylogue_testkit::delete_index(&home.root);
    let (mut engine, lock) = home.engine(first.len() as u64);
    engine.config.workers = 1;
    let archive = engine.archive.clone();
    fault::arm("parse", "progress.jsonl", Action::Hold, 1, 1);
    let running = std::thread::spawn(move || {
        let result = engine.run_pass();
        drop(engine);
        drop(lock);
        result
    });
    while !fault::held() && !running.is_finished() {
        std::thread::yield_now();
    }
    let progress = archive.index_build();
    let held = fault::held();
    fault::release();
    let pass = running.join().unwrap().unwrap();
    fault::clear();
    assert!(held, "the chain's second range waits");
    let progress = progress.expect("the index stays private while the build prepares");
    assert_eq!(
        (progress.chains, progress.chains_read),
        (1, 0),
        "{progress:?}"
    );
    assert_eq!(progress.bytes, (first.len() + second.len()) as u64);
    assert_eq!(progress.bytes_read, 0);
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(home.texts("progress"), ["first chunk", "second chunk"]);
}

/// A build reads the record as it stood when it started; the pass that
/// installs it then reads what arrived since, continuing each chain from
/// the build's position (docs/architecture.md, one path).
#[test]
fn a_build_installed_behind_its_record_is_brought_forward_by_upkeep() {
    let home = ClaudeHome::new();
    home.append(
        "forward.jsonl",
        &claude_user("one", "forward", "before the build"),
    );
    {
        let (mut engine, _lock) = home.engine(1 << 20);
        assert!(engine.run_pass().unwrap().errors.is_empty());
    }
    polylogue_testkit::delete_index(&home.root);
    home.append(
        "forward.jsonl",
        &claude_user("two", "forward", "after the position"),
    );
    home.append("later.jsonl", &claude_user("three", "later", "a new input"));
    let (mut engine, _lock) = home.engine(1 << 20);
    assert!(engine.archive.index_build().is_some());
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert!(engine.archive.index_build().is_none());
    assert_eq!(
        home.texts("forward"),
        ["before the build", "after the position"]
    );
    assert_eq!(home.texts("later"), ["a new input"]);
}

#[test]
fn content_named_messages_keep_their_occurrence_after_ranges_and_restart() {
    use std::io::Write;
    let home = ClaudeHome::new();
    let path = home.home.join(".codex/sessions/rollout-counter.jsonl");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let message = |text: &str| {
        serde_json::to_string(&serde_json::json!({
        "type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":text}]}
    })).unwrap()+"\n"
    };
    let mut bytes = "{\"type\":\"session_meta\",\"payload\":{\"id\":\"counter\"}}\n".to_owned();
    bytes.push_str(&message("same"));
    for n in 0..5000 {
        bytes.push_str(&message(&format!("distinct-{n}")));
    }
    bytes.push_str(&message("same"));
    std::fs::write(&path, bytes).unwrap();
    {
        let (mut engine, _lock) = home.engine(64 << 10);
        engine.config.workers = 1;
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    }
    std::fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(message("same").as_bytes())
        .unwrap();
    {
        let (mut engine, _lock) = home.engine(64 << 10);
        engine.config.workers = 1;
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    }
    let session = polylogue_model::SessionId::parse("codex:counter").unwrap();
    let messages = home.read(|reader| reader.messages(&session, false));
    assert_eq!(messages.len(), 5003);
    let hash = blake3::hash(
        b"polylogue-message-content\0[\"user\",[{\"text\":\"same\",\"type\":\"text\"}]]",
    )
    .to_hex();
    let prefix = format!("counter~user-{}-", &hash.as_str()[..16]);
    let repeated: Vec<_> = messages
        .iter()
        .filter(|row| row.id.native().as_str().starts_with(&prefix))
        .map(|row| row.id.native().as_str().to_owned())
        .collect();
    assert_eq!(
        repeated,
        [
            format!("{prefix}0"),
            format!("{prefix}1"),
            format!("{prefix}2")
        ]
    );
}
