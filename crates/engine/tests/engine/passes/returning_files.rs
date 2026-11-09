use super::*;
use polylogue_model::capture_log;

/// `(class, revision chain)` of the file's observations, oldest first.
fn observed(w: &ClaudeHome) -> Vec<(String, Option<i64>)> {
    w.reads()
        .reader()
        .unwrap()
        .query(
            "SELECT o.class, r.chain FROM a.observation o
             LEFT JOIN a.revision r ON r.input = o.input AND r.number = o.revision
             ORDER BY o.rowid",
            &[],
        )
        .unwrap()
        .into_iter()
        .map(|row| {
            (
                row["class"].as_str().unwrap().to_owned(),
                row["chain"].as_i64(),
            )
        })
        .collect()
}

fn chains(w: &ClaudeHome) -> i64 {
    w.count("SELECT count(DISTINCT chain) AS n FROM a.revision")
}

/// A file that disappears and comes back, or is replaced by another inode
/// with the same leading bytes, continues its chain: the same bytes are an
/// observation (`back`), more bytes are an append. Only changed stored bytes
/// start a new chain (docs/archive.md, stored inputs).
#[test]
fn a_returning_file_continues_its_chain_while_its_stored_bytes_remain() {
    let first = user("u1", "back", "first body");
    let more = assistant("a1", "back", "appended body");
    for case in ["gone and back", "gone and grown", "new inode", "changed"] {
        let w = ClaudeHome::new();
        w.append("back.jsonl", &first);
        let (mut engine, _lock) = w.engine(64 << 20);
        assert!(engine.run_pass().unwrap().errors.is_empty());
        let path = w.file("back.jsonl");
        match case {
            "gone and back" | "gone and grown" => {
                fs::remove_file(&path).unwrap();
                assert!(engine.run_pass().unwrap().errors.is_empty());
                let bytes = if case == "gone and back" {
                    first.clone()
                } else {
                    first.clone() + &more
                };
                fs::write(&path, bytes).unwrap();
            }
            "new inode" => {
                let copy = w.file("back.copy");
                fs::write(&copy, &first).unwrap();
                fs::rename(&copy, &path).unwrap();
            }
            _ => {
                fs::write(&path, user("u1", "back", "first bodY")).unwrap();
            }
        }
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{case}: {:?}", pass.errors);
        let expected: Vec<(String, Option<i64>)> = match case {
            "gone and back" => vec![
                ("first".into(), Some(1)),
                ("deleted".into(), None),
                ("back".into(), Some(1)),
            ],
            "gone and grown" => vec![
                ("first".into(), Some(1)),
                ("deleted".into(), None),
                ("appended".into(), Some(1)),
            ],
            "new inode" => vec![("first".into(), Some(1)), ("back".into(), Some(1))],
            _ => vec![("first".into(), Some(1)), ("rewritten".into(), Some(2))],
        };
        assert_eq!(observed(&w), expected, "{case}");
        let texts = w.texts("back");
        match case {
            "gone and grown" => assert_eq!(texts, ["first body", "appended body"], "{case}"),
            "changed" => assert_eq!(texts, ["first bodY"], "{case}"),
            _ => assert_eq!(texts, ["first body"], "{case}"),
        }
        // Nothing further happens while the file stays as it is.
        let quiet = engine.run_pass().unwrap();
        assert_eq!(quiet.bytes_stored, 0, "{case}");
        assert_eq!(observed(&w).len(), expected.len(), "{case}");
        assert_eq!(chains(&w), if case == "changed" { 2 } else { 1 });
    }
}

fn frame(payload: &serde_json::Value) -> Vec<u8> {
    let payload = serde_json::to_vec(payload).unwrap();
    let mut bytes = capture_log::Header::of(&payload).encode();
    bytes.extend(payload);
    bytes
}

fn capture(turns: &[&str]) -> serde_json::Value {
    serde_json::json!({
        "polylogue_capture_kind": "browser_llm_session", "schema_version": 1,
        "provenance": {"captured_at": "2026-10-01T00:00:00Z"},
        "session": {"provider": "gemini", "provider_session_id": "growing",
            "turns": turns.iter().enumerate().map(|(n, text)| serde_json::json!({
                "provider_turn_id": format!("t{n}"), "role": "user", "text": text,
            })).collect::<Vec<_>>()}
    })
}

/// The browser host appends each capture of a conversation to its log, so
/// the archive keeps one chain that grows by appends, read in several
/// ranges, across passes and across a restart (docs/extension.md).
#[test]
fn a_growing_capture_log_appends_to_one_chain() {
    let w = ClaudeHome::new();
    let path = w
        .config(256)
        .spool_dir("browser-capture")
        .join("gemini/growing.json");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let filler = "a capture of a growing conversation ".repeat(20);
    let mut log = capture_log::MAGIC.to_vec();
    let mut turns = Vec::new();
    for round in 0..4 {
        turns.push(format!("{filler}{round}"));
        let texts: Vec<&str> = turns.iter().map(String::as_str).collect();
        log.extend(frame(&capture(&texts)));
        fs::write(&path, &log).unwrap();
        // A restart between captures reloads the input's state.
        let (mut engine, _lock) = w.engine(256);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{round}: {:?}", pass.errors);
    }
    let produced: Vec<String> = w
        .reads()
        .reader()
        .unwrap()
        .query("SELECT produced FROM a.revision ORDER BY number", &[])
        .unwrap()
        .into_iter()
        .map(|row| row["produced"].as_str().unwrap().to_owned())
        .collect();
    assert_eq!(produced[0], "first");
    assert!(produced.len() >= 4, "{produced:?}");
    assert!(
        produced[1..].iter().all(|p| p == "appended"),
        "{produced:?}"
    );
    assert_eq!(chains(&w), 1);
    assert_eq!(
        w.count("SELECT sum(length) AS n FROM a.extent"),
        log.len() as i64
    );
    let reads = w.reads();
    let messages = reads
        .reader()
        .unwrap()
        .messages(&SessionId::parse("gemini-app:growing").unwrap(), false)
        .unwrap();
    assert_eq!(
        messages.iter().map(Message::text).collect::<Vec<_>>(),
        turns
    );
}

/// Chains of one input that hold the same bytes share their blobs: the
/// segment log holds each content once, whatever number of chains cite it
/// (docs/archive.md, stored inputs).
#[test]
fn identical_chains_store_their_bytes_once() {
    let w = ClaudeHome::new();
    let one = user("u1", "same", "the same body every time");
    let other = user("u1", "same", "a different body");
    let path = w.file("same.jsonl");
    let (mut engine, _lock) = w.engine(64 << 20);
    for bytes in [&one, &other, &one, &other, &one] {
        fs::write(&path, bytes).unwrap();
        assert!(engine.run_pass().unwrap().errors.is_empty());
    }
    assert_eq!(chains(&w), 5);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.extent"), 5);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.blob"), 2);
    assert_eq!(
        w.count("SELECT sum(size) AS n FROM a.blob"),
        (one.len() + other.len()) as i64
    );
}
