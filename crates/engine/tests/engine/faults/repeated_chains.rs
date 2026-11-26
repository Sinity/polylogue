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

/// The content a row names: its group's decoded bytes at its start, for its
/// length and its data's.
fn grouped_piece(
    row: &rusqlite::Row<'_>,
    names: &[String],
    group: usize,
) -> rusqlite::Result<rusqlite::types::Value> {
    let at = |column: &str| names.iter().position(|name| name == column);
    let body: Vec<u8> = row.get(group)?;
    let start = usize::try_from(row.get::<_, i64>(at("content_start").unwrap())?).unwrap();
    let mut length = usize::try_from(row.get::<_, i64>(at("content_length").unwrap())?).unwrap();
    if let Some(data) = at("data_length") {
        length += row.get::<_, Option<i64>>(data)?.unwrap_or(0) as usize;
    }
    let decoded = zstd::stream::decode_all(body.as_slice()).unwrap();
    Ok(rusqlite::types::Value::Blob(
        decoded[start..start + length].to_vec(),
    ))
}

/// Every row of every table, sorted, with local version keys named by their
/// content hashes. The full payload and every reference remain compared.
/// Wire-shape records are compared by what it says and the chain it cites, not
/// by the byte ranges it was read in or the revisions it took them at
/// ([`READ_RANGES`]). A status count no chain has reads as absent.
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
        .prepare("SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
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
        let columns = names.iter().map(|name| {
            if matches!(table.as_str(), "wire_shape" | "wire_type") && matches!(name.as_str(), "first" | "last") {
                // The chain the occurrence record cites, not the read range it took.
                format!("(SELECT json_extract(r.occurrence,'$.chain') FROM wire_occurrence r WHERE r.input_pk=t.input_pk AND r.chain=t.chain AND r.parse_version=t.parse_version AND r.number=t.\"{name}\") AS \"{name}\"")
            } else if let Some(natural) = natural_key(&table, name) {
                // Row keys are local to an index; what they name is compared
                // (docs/architecture.md, the record and the index).
                format!("{natural} AS \"{name}\"")
            } else if table == "wire_occurrence" && name == "number" {
                "NULL".to_owned()
            } else if table == "wire_occurrence" && name == "occurrence" {
                "json_extract(t.occurrence,'$.chain') AS occurrence".to_owned()
            } else if table == "text" && name == "group_id" {
                // Which group holds a text's bytes follows the steps that wrote
                // it (docs/archive.md, evolution); readers see only the text.
                "(SELECT g.body FROM content_group g WHERE g.id=t.group_id) AS group_id".to_owned()
            } else if name == "content_group" {
                // Likewise where a reading's content lies: its bytes are
                // compared.
                "(SELECT g.body FROM content_group g WHERE g.id=t.content_group) AS content_group"
                    .to_owned()
            } else if table == "wire_parse" && name == "state" {
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
                    if matches!(
                        name.as_str(),
                        "content_start" | "content_length" | "data_length"
                    ) {
                        continue;
                    }
                    let mut value: rusqlite::types::Value = row.get(n)?;
                    if table == "text" && name == "group_id" {
                        value = grouped_text(row, &names, n)?;
                    }
                    if name == "content_group" {
                        value = grouped_piece(row, &names, n)?;
                    }
                    if table == "parse_state" && name == "resume" {
                        value = resume_without_read_range(value);
                    }
                    text.push_str(&format!("{name}={value:?};"));
                }
                Ok(text)
            })
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        if table == "content_group" {
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
];
/// Values that follow the order of writes rather than what was written: a
/// block's row key, assigned as blocks are inserted, and the state of a
/// session's transcript lines, whose rewrite counter moves each time its
/// lines are rewritten and which readers compare only with its own earlier
/// value (docs/transcripts.md); the lines themselves are compared. A line's
/// row key is its row's place in the session, which names the row's input
/// by its ID, as `message.session_order` does. A copy is written after the newer chain's parse, where a
/// parse of the older chain would have come before it.
const WRITE_ORDER: &[(&str, &str)] = &[
    ("block", "pk"),
    ("session", "pk"),
    ("message", "pk"),
    ("tool_call", "pk"),
    ("event", "pk"),
    ("link", "pk"),
    ("document", "pk"),
    ("text", "id"),
    ("input_ref", "pk"),
    ("summary_state", "state"),
    ("transcript_line", "row_key"),
];

/// The natural key a column holding a local row key names.
fn natural_key(table: &str, column: &str) -> Option<String> {
    Some(match column {
        "session_pk" if table != "session" => {
            "(SELECT x.id FROM session x WHERE x.pk=t.session_pk)".to_owned()
        }
        "message_pk" => "(SELECT x.id FROM message x WHERE x.pk=t.message_pk)".to_owned(),
        "text_id" => "(SELECT hex(x.hash) FROM text x WHERE x.id=t.text_id)".to_owned(),
        "input_pk" => "(SELECT x.id FROM input_ref x WHERE x.pk=t.input_pk)".to_owned(),
        "call_block_pk" => "(SELECT m.id||'/'||b.idx FROM block b JOIN message m ON m.pk=b.message_pk WHERE b.pk=t.call_block_pk)".to_owned(),
        "event_pk" => "(SELECT x.id FROM event x WHERE x.pk=t.event_pk)".to_owned(),
        "document_pk" => "(SELECT x.id FROM document x WHERE x.pk=t.document_pk)".to_owned(),
        _ => return None,
    })
}
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
fn resume_without_read_range(value: rusqlite::types::Value) -> rusqlite::types::Value {
    let rusqlite::types::Value::Blob(compressed) = &value else {
        return value;
    };
    let mut checkpoint: serde_json::Value =
        serde_json::from_slice(&zstd::decode_all(&compressed[..]).unwrap()).unwrap();
    if let Some(context) = checkpoint
        .pointer_mut("/_polylogue_context")
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
    "key=Text(\"search_files\")",
];

/// A rebuild reads an older chain that holds exactly a newer chain's bytes
/// whole, like every chain, and keeps every chain a message appeared in
/// recorded (docs/archive.md, readings; docs/architecture.md, one path).
#[test]
fn a_rebuild_reads_an_identical_older_chain_whole() {
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
    assert_eq!(pass.bytes_read, 2 * both);
    // Every chain a message appeared in stays recorded.
    let chains = w.count(
        "SELECT count(DISTINCT chain) AS n FROM message_parse s WHERE s.message_id LIKE '%u1%'",
    );
    assert_eq!(chains, 4);
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
