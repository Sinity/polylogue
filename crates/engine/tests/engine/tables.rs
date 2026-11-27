//! Every table of an index, compared by what its rows say: local row keys
//! are named by the natural keys they stand for, and values that follow the
//! order of writes rather than what was written are left out.
use polylogue_testkit::ClaudeHome;
use std::collections::BTreeMap;

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
pub(crate) fn dump(w: &ClaudeHome) -> BTreeMap<String, Vec<String>> {
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
            if matches!(table.as_str(), "wire_shape" | "wire_type") && matches!(name.as_str(), "first_position" | "last_position") {
                // The chain the occurrence record cites, not the read range it took.
                format!("(SELECT json_extract(r.occurrence,'$.chain') FROM wire_occurrence r WHERE r.input_pk=t.input_pk AND r.chain=t.chain AND r.parser_version=t.parser_version AND r.position=t.\"{name}\") AS \"{name}\"")
            } else if let Some(natural) = natural_key(&table, name) {
                // Row keys are local to an index; what they name is compared
                // (docs/architecture.md, the record and the index).
                format!("{natural} AS \"{name}\"")
            } else if table == "wire_occurrence" && name == "position" {
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
                // Completion is evidence from earlier reads. A rebuild of
                // an incomplete tail never made the earlier complete read.
                // Wire coverage tests compare those observations separately.
                format!(
                    "json_remove(t.state,{},'$.completed','$.completed_before_reset') AS state",
                    READ_RANGES.iter().map(|key| format!("'$.{key}'")).collect::<Vec<_>>().join(",")
                )
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
        if table == "summary_resume" {
            // Fold checkpoints follow the boundaries of previous runs.
            // The repeated-version property compares their resulting
            // summaries, including after an append uses these checkpoints.
            continue;
        }
        if table == "meta" {
            rows.retain(|row| !INDEX_LOCAL_META.iter().any(|key| row.starts_with(key)));
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
    ("session", "summary_changed"),
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
        "call_block_pk" => "(SELECT m.id||'/'||b.position FROM block b JOIN message m ON m.pk=b.message_pk WHERE b.pk=t.call_block_pk)".to_owned(),
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
const INDEX_LOCAL_META: &[&str] = &[
    "key=Text(\"write_counter\")",
    "key=Text(\"generation\")",
    "key=Text(\"index_instance\")",
    "key=Text(\"import_observation_position\")",
    "key=Text(\"cursor_key\")",
    "key=Text(\"search_files\")",
];

pub(crate) fn differences(
    left: &BTreeMap<String, Vec<String>>,
    right: &BTreeMap<String, Vec<String>>,
) -> String {
    let mut out = String::new();
    let tables: std::collections::BTreeSet<&String> = left.keys().chain(right.keys()).collect();
    for table in tables {
        let (l, r) = (
            left.get(table).cloned().unwrap_or_default(),
            right.get(table).cloned().unwrap_or_default(),
        );
        if l == r {
            continue;
        }
        out.push_str(&format!("table {table}\n"));
        for row in l.iter().filter(|row| !r.contains(row)) {
            out.push_str(&format!("  only left:  {row}\n"));
        }
        for row in r.iter().filter(|row| !l.contains(row)) {
            out.push_str(&format!("  only right: {row}\n"));
        }
        if l.len() != r.len() {
            out.push_str(&format!("  rows: {} left, {} right\n", l.len(), r.len()));
        }
    }
    out
}
