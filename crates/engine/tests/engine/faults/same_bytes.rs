use super::*;
use polylogue_testkit::{
    ClaudeHome, claude_assistant as assistant, claude_user as user, delete_index,
};
use std::collections::BTreeMap;

/// A Claude Code record without a `uuid`, so its message is named by its
/// content.
fn unnamed(session: &str, text: &str) -> String {
    format!(
        "{}\n",
        serde_json::json!({
            "type": "user", "sessionId": session, "timestamp": "2026-03-01T10:00:05Z",
            "message": {"role": "user", "content": text}
        })
    )
}

/// A capture log holding one capture per entry, each with these turns.
fn capture_log(captures: &[&[&str]]) -> Vec<u8> {
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

/// Every row of every table of `index.db`, as text, sorted per table.
fn dump(w: &ClaudeHome) -> BTreeMap<String, Vec<String>> {
    let conn = rusqlite::Connection::open_with_flags(
        w.root.join("index.db"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .unwrap();
    let tables: Vec<String> = conn
        .prepare("SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    let mut out = BTreeMap::new();
    for table in tables {
        let mut statement = conn.prepare(&format!("SELECT * FROM \"{table}\"")).unwrap();
        let names: Vec<String> = statement
            .column_names()
            .into_iter()
            .map(str::to_owned)
            .collect();
        let mut rows: Vec<String> = statement
            .query_map([], |row| {
                let mut text = String::new();
                for (n, name) in names.iter().enumerate() {
                    let column = |(t, c): &(&str, &str)| {
                        (*t == "*" || *t == table.as_str()) && *c == name.as_str()
                    };
                    if WRITE_COUNTER_VALUES.iter().any(&column) || WRITE_ORDER.iter().any(&column) {
                        continue;
                    }
                    let value: rusqlite::types::Value = row.get(n)?;
                    text.push_str(&format!("{name}={value:?};"));
                }
                Ok(text)
            })
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        if table == "meta" {
            rows.retain(|row| {
                !CREATED_WITH_THE_INDEX
                    .iter()
                    .any(|key| row.starts_with(key))
            });
        }
        rows.sort();
        out.insert(table, rows);
    }
    out
}

/// Columns holding values of the write counter rather than facts: which
/// write admitted a row, last changed a session, or last derived or indexed
/// it. A copy and a parse are different writes.
const WRITE_COUNTER_VALUES: &[(&str, &str)] = &[
    ("*", "admitted"),
    ("*", "commit_seq"),
    ("session", "changed"),
    ("derive_state", "from_changed"),
    ("session_item_state", "indexed_changed"),
];
/// Values that follow the order of writes rather than what was written: a
/// block's row key, assigned as blocks are inserted, and the counter that
/// moves each time a session's indexed entries are rewritten, which readers
/// compare only with its own earlier value (docs/reading.md, the session
/// item index). A copy is written after the newer chain's parse, where a
/// parse of the older chain would have come before it.
const WRITE_ORDER: &[(&str, &str)] = &[("block", "pk"), ("session_item_state", "rewrite_version")];
/// Rows of `meta` that are the write counter itself or identities each index
/// is given when it is created.
const CREATED_WITH_THE_INDEX: &[&str] = &[
    "key=Text(\"write_counter\")",
    "key=Text(\"generation\")",
    "key=Text(\"cursor_key\")",
];

/// A rebuild records an older chain that holds exactly a newer chain's
/// bytes from that chain's reading instead of parsing it. The index equals
/// the one a rebuild that parses every chain writes, provenance included
/// (docs/archive.md, versioned rows): every table, row for row.
#[test]
fn an_identical_older_chain_is_recorded_exactly_as_its_parse() {
    fault::clear();
    let w = ClaudeHome::new();
    // Chains 1 and 3 hold `a`, chains 2 and 4 hold `b`; `a` has a message
    // named by its content that `b`, the latest chain, no longer holds.
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

    let rebuild = |copies: bool| {
        delete_index(&w.root);
        fault::clear();
        if !copies {
            fault::arm("write.copy_reading", "", Action::Fail, 0, u32::MAX);
        }
        // One worker, so each rebuild writes in one order.
        let (archive, _lock) = polylogue_store::Archive::open_or_create(&w.root).unwrap();
        let mut config = w.config(64 << 20);
        config.workers = 1;
        let mut engine =
            polylogue_engine::Engine::new(config, archive, polylogue_testkit::stopped_attempt())
                .unwrap();
        let pass = engine.run_pass().unwrap();
        fault::clear();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        drop(engine);
        (pass, dump(&w))
    };
    let (parsed, every_chain_parsed) = rebuild(false);
    assert_eq!(parsed.readings_copied, 0);
    let (copied, with_copies) = rebuild(true);
    // In each input chain 1 is recorded from chain 3, chain 2 from chain 4,
    // and only chains 3 and 4 are read.
    assert_eq!(copied.readings_copied, 4);
    let both = (a.len() + b.len() + captured_a.len() + captured_b.len()) as u64;
    assert_eq!(parsed.bytes_read, 2 * both);
    assert_eq!(copied.bytes_read, both);
    let differing: Vec<String> = every_chain_parsed
        .iter()
        .filter(|(table, rows)| with_copies.get(*table) != Some(rows))
        .map(|(table, rows)| {
            format!(
                "{table}:\n  parsed {rows:?}\n  copied {:?}",
                with_copies.get(table)
            )
        })
        .collect();
    assert!(differing.is_empty(), "{}", differing.join("\n"));
    assert_eq!(
        every_chain_parsed.keys().collect::<Vec<_>>(),
        with_copies.keys().collect::<Vec<_>>()
    );
    // Every chain a message appeared in stays recorded.
    let chains = w.count(
        "SELECT count(DISTINCT chain) AS n FROM message_source s JOIN message m ON m.pk = s.message_pk WHERE m.id LIKE '%u1%'",
    );
    assert_eq!(chains, 4);
}
