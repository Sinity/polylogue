use super::*;
use polylogue_testkit::{
    ClaudeHome, claude_assistant as assistant, claude_user as user, delete_index,
};
use std::collections::BTreeMap;

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

/// Every row of every table, sorted, with local version keys named by their
/// content hashes. The full payload and every reference remain compared.
/// Wire-shape records are compared by what it says and the chain it cites, not
/// by the byte ranges it was read in or the revisions it took them at
/// ([`READ_RANGES`]). A status count no chain has reads as absent.
/// The text a `text` row names: its group's decoded bytes at its start and
/// length.
fn grouped_text(
    row: &rusqlite::Row<'_>,
    names: &[String],
    group: usize,
) -> rusqlite::Result<rusqlite::types::Value> {
    let at = |column: &str| names.iter().position(|name| name == column).unwrap();
    let body: Vec<u8> = row.get(group)?;
    let start = usize::try_from(row.get::<_, i64>(at("start"))?).unwrap();
    let length = usize::try_from(row.get::<_, i64>(at("length"))?).unwrap();
    let decoded = zstd::stream::decode_all(body.as_slice()).unwrap();
    Ok(rusqlite::types::Value::Text(
        String::from_utf8(decoded[start..start + length].to_vec()).unwrap(),
    ))
}

pub(super) fn dump(w: &ClaudeHome) -> BTreeMap<String, Vec<String>> {
    let conn = rusqlite::Connection::open_with_flags(
        w.root.join("index.db"),
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
    )
    .unwrap();
    assert_eq!(
        conn.query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |row| {
            row.get::<_, i64>(0)
        })
        .unwrap(),
        0
    );
    let tables: Vec<String> = conn
        .prepare("SELECT name FROM sqlite_schema WHERE (type = 'table' OR name = 'message_source') AND name NOT LIKE 'sqlite_%' ORDER BY name")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    let mut out = BTreeMap::new();
    for table in tables {
        let statement = conn.prepare(&format!("SELECT * FROM \"{table}\"")).unwrap();
        let names: Vec<String> = statement
            .column_names()
            .into_iter()
            .map(str::to_owned)
            .collect();
        // Version row keys follow insertion order. Translate them, rather than
        // dropping them: a wrong payload/reference still names the wrong hash.
        let columns = names.iter().map(|name| {
            if name == "version_pk" || name == "source_version_pk" || (table == "message_version" && name == "pk") {
                format!("(SELECT v.version_hash FROM message_version v WHERE v.pk=t.\"{name}\") AS \"{name}\"")
            } else if matches!(table.as_str(), "wire_shape" | "wire_type") && matches!(name.as_str(), "first" | "last") {
                // The chain the occurrence record cites, not the read range it took.
                format!("(SELECT json_extract(r.occurrence,'$.chain') FROM wire_occurrence r WHERE r.input_pk=t.input_pk AND r.chain=t.chain AND r.reading_version=t.reading_version AND r.number=t.\"{name}\") AS \"{name}\"")
            } else if table == "wire_occurrence" && name == "number" {
                "NULL".to_owned()
            } else if table == "wire_occurrence" && name == "occurrence" {
                "json_extract(t.occurrence,'$.chain') AS occurrence".to_owned()
            } else if table == "text" && name == "group_id" {
                // Which group holds a text's bytes follows the steps that wrote
                // it (docs/archive.md, evolution); readers see only the text.
                "(SELECT g.body FROM text_group g WHERE g.id=t.group_id) AS group_id".to_owned()
            } else if table == "wire_reading" && name == "state" {
                format!("json_remove(t.state,{}) AS state", READ_RANGES.iter().map(|key| format!("'$.{key}'")).collect::<Vec<_>>().join(","))
            } else { format!("t.\"{name}\"") }
        }).collect::<Vec<_>>().join(",");
        let mut statement = conn
            .prepare(&format!("SELECT {columns} FROM \"{table}\" t"))
            .unwrap();
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
                    if table == "text" && matches!(name.as_str(), "start" | "length") {
                        continue;
                    }
                    let mut value: rusqlite::types::Value = row.get(n)?;
                    if table == "text" && name == "group_id" {
                        value = grouped_text(row, &names, n)?;
                    }
                    if table == "parse_state" && name == "checkpoint" {
                        value = checkpoint_without_read_range(value);
                    }
                    text.push_str(&format!("{name}={value:?};"));
                }
                Ok(text)
            })
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        if table == "text_group" {
            // Compared through the text rows above.
            continue;
        }
        if table == "meta" {
            rows.retain(|row| {
                !CREATED_WITH_THE_INDEX
                    .iter()
                    .any(|key| row.starts_with(key))
            });
        }
        rows.sort();
        if table == "wire_occurrence" {
            // One row per chain cited, however many ranges read it.
            rows.dedup();
        }
        if table == "parse_status_count" {
            // Readers count only statuses some chain has (read.rs).
            rows.retain(|row| !row.ends_with("n=Integer(0);"));
        }
        out.insert(table, rows);
    }
    out
}

/// Columns holding values of the write counter rather than content: which
/// write committed a row, last changed a session, or last derived or indexed
/// it. A copy and a parse are different writes.
const WRITE_COUNTER_VALUES: &[(&str, &str)] = &[
    ("*", "committed"),
    ("*", "commit_seq"),
    ("session", "changed"),
    ("session", "derived_changed"),
    ("session_first_prompt", "changed"),
    ("session_item_state", "indexed_changed"),
];
/// Values that follow the order of writes rather than what was written: a
/// block's row key, assigned as blocks are inserted, and the counter that
/// moves each time a session's indexed entries are rewritten, which readers
/// compare only with its own earlier value (docs/reading.md, the session
/// item index). A copy is written after the newer chain's parse, where a
/// parse of the older chain would have come before it.
const WRITE_ORDER: &[(&str, &str)] = &[("block", "pk"), ("session_item_state", "rewrite_version")];
/// Fields of a reading's wire-shape record that note the ranges it was read
/// in: the occurrences of its first and latest read ranges (each a range and
/// the revision it was taken at) and the latest range's bounds and hash,
/// which let a range read again be recognized. A copied reading carries the
/// newer chain's read ranges over the shared bytes, retaken for its own
/// chain's revisions (docs/archive.md, versioned rows), where a parse of the
/// chain alone reads other ranges and so cites other revisions; what they
/// recorded is compared in full.
const READ_RANGES: &[&str] = &["first", "last", "last_start", "last_end", "last_hash"];

/// A parser checkpoint as JSON text, without the position its last read
/// range started at (the base of the parser context changes it carries),
/// which is where the chain's reading last stopped and went on, not what it
/// read.
fn checkpoint_without_read_range(value: rusqlite::types::Value) -> rusqlite::types::Value {
    let rusqlite::types::Value::Blob(compressed) = &value else {
        return value;
    };
    let mut checkpoint: serde_json::Value =
        serde_json::from_slice(&zstd::decode_all(&compressed[..]).unwrap()).unwrap();
    if let Some(context) = checkpoint
        .pointer_mut("/parser_state/_polylogue_context")
        .and_then(serde_json::Value::as_object_mut)
    {
        context.remove("base");
    }
    rusqlite::types::Value::Text(checkpoint.to_string())
}

/// Rows of `meta` that are the write counter itself or identities each index
/// is given when it is created.
const CREATED_WITH_THE_INDEX: &[&str] = &[
    "key=Text(\"write_counter\")",
    "key=Text(\"generation\")",
    "key=Text(\"cursor_key\")",
];

/// A rebuild records an older chain that holds exactly a newer chain's
/// bytes from that chain's reading instead of parsing it. The index equals
/// the one a rebuild that parses every chain writes, sources included
/// (docs/archive.md, versioned rows): every table, row for row.
#[test]
fn identical_older_chain_is_recorded_as_its_parse() {
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
    assert_eq!((parsed.readings_copied, parsed.bytes_copied), (0, 0));
    let (copied, with_copies) = rebuild(true);
    // In each input chain 1 is recorded from chain 3 and chain 2 from chain
    // 4. Chains `a` and `b` of the transcript also share their first record,
    // so chain 3 is recorded from chain 4 through it and parses only the
    // rest (docs/archive.md, versioned rows); the capture logs share no
    // complete capture, and chain 3 of the log is parsed whole.
    assert_eq!(copied.readings_copied, 5);
    let first = user("u1", "same", "first prompt").len() as u64;
    let both = (a.len() + b.len() + captured_a.len() + captured_b.len()) as u64;
    // Chain 4's parse stops at the end of the record the chains share and
    // reads on from there, so no byte is read twice.
    assert_eq!(parsed.bytes_read, 2 * both);
    assert_eq!(copied.bytes_read, both - first);
    // Chains 1 and 2 of each input are read whole by their copies, chain 3
    // of the transcript through its first record.
    assert_eq!(copied.bytes_copied, both + first);
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
