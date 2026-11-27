use super::*;
use crate::tables::{differences, dump};
use polylogue_testkit::{
    ClaudeHome, claude_assistant as assistant, claude_user as user, delete_index,
};

/// A Claude Code record without a `uuid`, so its message is named by its
/// content.
pub(super) fn unnamed(session: &str, text: &str) -> String {
    format!(
        "{}\n",
        serde_json::json!({
            "type": "user", "sessionId": session, "timestamp": "2026-03-01T10:00:05Z",
            "message": {"role": "user", "content": text}
        })
    )
}

/// A capture log holding one capture per entry, each with these turns.
pub(super) fn capture_log(captures: &[&[&str]]) -> Vec<u8> {
    let mut log = polylogue_model::capture_log::MAGIC.to_vec();
    for turns in captures {
        let payload = serde_json::to_vec(&serde_json::json!({
            "polylogue_capture_kind": "browser_llm_session", "schema_version": 1,
            "provenance": {"captured_at": "2026-10-01T00:00:00Z"},
            "session": {"provider": "gemini", "provider_session_id": "same",
                "turns": turns.iter().enumerate().map(|(n, text)| serde_json::json!({
                    "provider_turn_id": format!("t{n}"),
                    "role": if n % 2 == 0 { "user" } else { "assistant" },
                    "text": text,
                })).collect::<Vec<_>>()}
        }))
        .unwrap();
        log.extend(polylogue_model::capture_log::Header::of(&payload).encode());
        log.extend(payload);
    }
    log
}

/// A rebuild parses only what no newer chain repeats, and gives the index
/// tables incremental updates gave (docs/archive.md, readings;
/// docs/architecture.md, one path). A file written as `a`, `b`, `a`, `b`:
/// chains 1 and 2 are repeated whole by chains 3 and 4 and have no parse,
/// and chain 3 keeps only the records after the first line it shares with
/// chain 4.
#[test]
fn a_rebuild_parses_only_what_no_newer_chain_repeats() {
    fault::clear();
    let w = ClaudeHome::new();
    // `a` has a message named by its content that `b`, the latest chain,
    // no longer holds.
    let a = user("u1", "same", "first prompt")
        + &assistant("a1", "same", "first answer")
        + &unnamed("same", "only in a")
        + &user("u2", "other", "another session");
    let b = user("u1", "same", "first prompt")
        + &assistant("a1", "same", "edited answer")
        + &unnamed("same", "only in b");
    let path = w.file("same.jsonl");
    // A browser capture log whose captures hold different turns.
    let log = w
        .config(64 << 20)
        .spool_dir("browser-capture")
        .join("gemini/same.json");
    fs::create_dir_all(log.parent().unwrap()).unwrap();
    let captured_a = capture_log(&[&["a question", "an answer"]]);
    let captured_b = capture_log(&[&["a question"], &["a question", "another answer"]]);
    {
        let (mut engine, _lock) = w.engine(64 << 20);
        for (bytes, captured) in [
            (&a, &captured_a),
            (&b, &captured_b),
            (&a, &captured_a),
            (&b, &captured_b),
        ] {
            fs::write(&path, bytes).unwrap();
            fs::write(&log, captured).unwrap();
            assert!(engine.run_pass().unwrap().errors.is_empty());
        }
    }
    assert_eq!(
        w.count("SELECT count(*) AS n FROM (SELECT DISTINCT input, chain FROM a.revision)"),
        8
    );
    // Each input keeps a parse of its two newest chains only.
    assert_eq!(w.count("SELECT count(*) AS n FROM parse_state"), 4);
    let updated = dump(&w);
    delete_index(&w.root);
    fault::clear();
    // One worker, so the rebuild writes in one order.
    let (archive, _lock) = polylogue_store::Archive::open_or_create(&w.root).unwrap();
    let mut config = w.config(64 << 20);
    config.workers = 1;
    let mut engine = polylogue_engine::Engine::new(config, archive).unwrap();
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    drop(engine);
    let both = (a.len() + b.len() + captured_a.len() + captured_b.len()) as u64;
    assert_eq!(pass.bytes_read, both);
    // The first line both versions begin with is held by the newest chain
    // only.
    let chains = w.count(
        "SELECT count(DISTINCT chain) AS n FROM message_parse s WHERE s.message_id LIKE '%u1%'",
    );
    assert_eq!(chains, 1);
    let differ = differences(&updated, &dump(&w));
    assert!(differ.is_empty(), "{differ}");
}

/// Bytes alone cannot cover an older reading. The rebuild keeps the older
/// records if the newer parser fails, and omits them when it succeeds.
#[test]
fn a_failed_newer_reading_keeps_the_readable_older_copy() {
    use std::io::Read as _;

    fault::clear();
    let w = ClaudeHome::new();
    let path = w.file("same.jsonl");
    let older = user("shared", "same", "Shared first message");
    let unrelated = user("other", "different", "A different chain");
    let newest = older.clone() + &assistant("later", "same", "Only in the newest chain");
    {
        let (mut engine, _lock) = w.engine(64 << 20);
        for bytes in [&older, &unrelated, &newest] {
            fs::write(&path, bytes).unwrap();
            let pass = engine.run_pass().unwrap();
            assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        }
    }
    let versions = w.read(|reader| {
        reader.query(
            "SELECT input,chain,length FROM a.revision ORDER BY number",
            &[],
        )
    });
    assert_eq!(versions.len(), 3);
    let input = versions[0]["input"].as_str().unwrap();
    let first = versions[0]["chain"].as_i64().unwrap();
    let last = versions[2]["chain"].as_i64().unwrap();
    let reading_chain = || {
        w.read(|reader| {
            reader.query(
                "SELECT chain FROM message_parse WHERE message_id='claude-code:shared'",
                &[],
            )
        })
    };

    fs::remove_file(&path).unwrap();
    delete_index(&w.root);
    fault::arm("parse", "same.jsonl", Action::Panic, 0, 1);
    {
        let (mut engine, _lock) = w.engine(64 << 20);
        let pass = engine.run_pass().unwrap();
        assert_eq!(pass.errors.len(), 1, "{:?}", pass.errors);
        assert!(pass.errors[0].to_string().contains("panicked"));
    }
    assert_eq!(reading_chain(), [serde_json::json!({"chain":first})]);
    assert_eq!(w.texts("same"), ["Shared first message"]);
    assert_eq!(
        w.count(&format!(
            "SELECT count(*) AS n FROM parse_state WHERE chain={last} AND state='failed'"
        )),
        1
    );

    let blobs = std::sync::Arc::new(polylogue_store::Archive::at(&w.root).blobs().unwrap());
    for (version, bytes) in versions.iter().zip([&older, &unrelated, &newest]) {
        let chain = w.read(|reader| {
            reader.stored_chain(
                input,
                version["chain"].as_i64().unwrap(),
                version["length"].as_u64().unwrap(),
            )
        });
        let mut raw = Vec::new();
        chain.reader(blobs.clone()).read_to_end(&mut raw).unwrap();
        assert_eq!(raw, bytes.as_bytes());
    }

    fault::clear();
    delete_index(&w.root);
    {
        let (mut engine, _lock) = w.engine(64 << 20);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    }
    assert_eq!(reading_chain(), [serde_json::json!({"chain":last})]);
    assert_eq!(
        w.texts("same"),
        ["Shared first message", "Only in the newest chain"]
    );
}

/// The trace and the first-seen time of a record name every chain that
/// held it, read from the recorded agreements: the chains with no parse of
/// their own included (docs/archive.md, readings).
#[test]
fn trace_and_first_seen_of_a_record_held_by_a_covered_chain() {
    fault::clear();
    let w = ClaudeHome::new();
    let a = user("u1", "same", "first prompt") + &assistant("a1", "same", "first answer");
    let b = user("u1", "same", "first prompt") + &assistant("a1", "same", "edited answer");
    let path = w.file("same.jsonl");
    {
        let (mut engine, _lock) = w.engine(64 << 20);
        // Each version is read a minute after the one before.
        for (minute, bytes) in [&a, &b, &a, &b].into_iter().enumerate() {
            engine.config.clock = polylogue_model::WallClock::fixed(
                format!("2026-10-07T10:0{minute}:00Z").parse().unwrap(),
            );
            fs::write(&path, bytes).unwrap();
            assert!(engine.run_pass().unwrap().errors.is_empty());
        }
    }
    let id = polylogue_model::MessageId::parse("claude-code:u1").unwrap();
    let card = w
        .read(|reader| reader.record_chain(&polylogue_model::Reference::Message(id.clone())))
        .unwrap();
    let held: Vec<(i64, bool)> = card
        .chain
        .iter()
        .filter_map(|step| match step {
            polylogue_model::read::ChainStep::InputRecord { chain, shown, .. } => {
                Some((*chain, *shown))
            }
            _ => None,
        })
        .collect();
    // Only the newest chain has a row; the three before it hold the same
    // first line.
    let numbers: Vec<i64> = w
        .read(|reader| reader.query("SELECT DISTINCT chain FROM a.revision ORDER BY chain", &[]))
        .iter()
        .map(|row| row["chain"].as_i64().unwrap())
        .collect();
    assert_eq!(numbers.len(), 4);
    assert_eq!(
        held,
        [
            (numbers[3], true),
            (numbers[0], false),
            (numbers[1], false),
            (numbers[2], false)
        ]
    );
    // First seen when the first version was read, not when the chain that
    // has the row was.
    let value = |sql: &str| -> serde_json::Value {
        w.read(|reader| reader.query(sql, &[]))[0]["t"].clone()
    };
    let first = value("SELECT seen_at AS t FROM a.revision ORDER BY number LIMIT 1");
    assert_ne!(
        first,
        value("SELECT seen_at AS t FROM a.revision ORDER BY number DESC LIMIT 1")
    );
    assert_eq!(
        value(&format!(
            "SELECT {} AS t FROM message m WHERE m.id = 'claude-code:u1'",
            polylogue_store::query::fields::first_seen_of("m")
        )),
        first
    );
}

/// A transcript line names its message or event by content, not by the row
/// key the order of writes gave it (docs/transcripts.md, transcript lines:
/// an entry is a function of its message or event alone), so two archives
/// that read the same inputs in opposite orders hold the same lines.
#[test]
fn transcript_lines_are_the_same_whatever_order_inputs_arrive_in() {
    fault::clear();
    let transcript = user("u1", "same", "first prompt") + &assistant("a1", "same", "an answer");
    let captured = capture_log(&[&["a question", "another answer"]]);
    let read = |transcript_first: bool| {
        let w = ClaudeHome::new();
        let path = w.file("same.jsonl");
        let log = w
            .config(64 << 20)
            .spool_dir("browser-capture")
            .join("gemini/same.json");
        fs::create_dir_all(log.parent().unwrap()).unwrap();
        let (mut engine, _lock) = w.engine(64 << 20);
        let mut writes = [(&path, transcript.as_bytes()), (&log, &captured[..])];
        if !transcript_first {
            writes.reverse();
        }
        for (at, bytes) in writes {
            fs::write(at, bytes).unwrap();
            assert!(engine.run_pass().unwrap().errors.is_empty());
        }
        drop(engine);
        let conn = rusqlite::Connection::open(w.root.join("index.db")).unwrap();
        let messages: Vec<String> = conn
            .prepare("SELECT id FROM message ORDER BY pk")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        (messages, dump(&w).remove("transcript_line").unwrap())
    };
    let (transcript_keys, transcript_first) = read(true);
    let (log_keys, log_first) = read(false);
    // The input read first holds the lower row keys.
    let origins = |keys: &[String]| -> Vec<String> {
        keys.iter()
            .map(|key| key.split(':').next().unwrap().to_owned())
            .collect()
    };
    assert_eq!(
        origins(&transcript_keys),
        ["claude-code", "claude-code", "gemini-app", "gemini-app"]
    );
    assert_eq!(
        origins(&log_keys),
        ["gemini-app", "gemini-app", "claude-code", "claude-code"]
    );
    // One header entry per message.
    let headers = transcript_first
        .iter()
        .filter(|row| row.contains("message_pk=Text("))
        .count();
    assert_eq!(headers, 4);
    assert_eq!(transcript_first, log_first);
}
