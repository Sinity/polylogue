//! Excision in `index.db`, `archive.db` and `embeddings.db`: a session's
//! rows from every table, a text from the search indexes, and the notes,
//! deliveries and inputs that held it.

use super::*;

/// Text scope in `index.db`: every value holding the text gets spaces in
/// its place: texts (with their search entries), block bodies, tool
/// inputs and outputs, tool-call commands and paths, event payloads,
/// titles and provider attributes. Texts that changed are recorded so
/// their embeddings go.
pub(super) fn redact_index(
    i: &Transaction<'_>,
    text: &str,
    needles: &[Vec<u8>],
    dry_run: bool,
    removed_texts: &mut BTreeSet<Vec<u8>>,
) -> Result<()> {
    let texts: Vec<(i64, Vec<u8>, Vec<u8>, i64)> = i
        .prepare("SELECT id, hash, body, indexed FROM text")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .collect::<rusqlite::Result<_>>()?;
    for (id, hash, body, indexed) in texts {
        let Ok(body) = unpack_text(&body) else {
            continue;
        };
        if !body.contains(text) {
            continue;
        }
        removed_texts.insert(hash);
        if dry_run {
            continue;
        }
        let new_body = body.replace(text, &blank(text.as_bytes()));
        let new_hash = blake3::hash(new_body.as_bytes());
        let existing: Option<i64> = i
            .query_row(
                "SELECT id FROM text WHERE hash = ?1",
                [&new_hash.as_bytes()[..]],
                |r| r.get(0),
            )
            .optional()?;
        let new_id = match existing {
            Some(n) => n,
            None => {
                i.execute(
                    "INSERT INTO text(hash, body, indexed) VALUES (?1, ?2, 0)",
                    params![
                        &new_hash.as_bytes()[..],
                        zstd::bulk::compress(new_body.as_bytes(), 1)?
                    ],
                )?;
                i.last_insert_rowid()
            }
        };
        let folded = polylogue_model::text::fold(&new_body);
        for (bit, table) in [(1, "text_fts"), (2, "output_fts")] {
            if indexed & bit != 0 {
                i.execute(
                    &format!("INSERT INTO {table}(rowid, body) VALUES (?1, ?2)"),
                    params![new_id, folded],
                )?;
            }
        }
        i.execute(
            "UPDATE text SET indexed = indexed | ?2 WHERE id = ?1",
            params![new_id, indexed],
        )?;
        i.execute(
            "UPDATE block SET text_id = ?2 WHERE text_id = ?1",
            params![id, new_id],
        )?;
        remove_text(i, id, indexed)?;
    }
    if dry_run {
        return Ok(());
    }
    // Compressed JSON columns.
    for (table, key, col) in [("block", "pk", "body"), ("event", "pk", "data")] {
        let rows: Vec<(i64, Vec<u8>)> = i
            .prepare(&format!("SELECT {key}, {col} FROM {table}"))?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        for (pk, packed) in rows {
            let Ok(json) = zstd::stream::decode_all(&packed[..]) else {
                continue;
            };
            let Ok(s) = String::from_utf8(json) else {
                continue;
            };
            let new = replace_all(&s, needles);
            if new != s {
                i.execute(
                    &format!("UPDATE {table} SET {col} = ?2 WHERE {key} = ?1"),
                    params![pk, zstd::bulk::compress(new.as_bytes(), 1)?],
                )?;
            }
        }
    }
    // Plain text columns.
    let calls: Vec<(i64, Option<String>, Option<String>)> = i
        .prepare(
            "SELECT pk, command, path FROM tool_call WHERE instr(command, ?1) OR instr(path, ?1)",
        )?
        .query_map([text], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    for (pk, command, path) in calls {
        i.execute(
            "INSERT INTO tool_call_trigram(tool_call_trigram, rowid, command, path)
             VALUES ('delete', ?1, ?2, ?3)",
            params![pk, command, path],
        )?;
        let fix = |v: Option<String>| v.map(|s| replace_all(&s, needles));
        let (command, path) = (fix(command), fix(path));
        i.execute(
            "UPDATE tool_call SET command = ?2, path = ?3, command_norm = NULL WHERE pk = ?1",
            params![pk, command, path],
        )?;
        i.execute(
            "INSERT INTO tool_call_trigram(rowid, command, path) VALUES (?1, ?2, ?3)",
            params![pk, command, path],
        )?;
    }
    for (table, col) in [
        ("block", "input_json"),
        ("block", "data_json"),
        ("message", "attrs_json"),
        ("session", "title"),
        ("session", "derived_title"),
        ("session", "attrs_json"),
    ] {
        for n in needles {
            let Ok(n) = std::str::from_utf8(n) else {
                continue;
            };
            i.execute(
                &format!(
                    "UPDATE {table} SET {col} = replace({col}, ?1, ?2) WHERE instr({col}, ?1)"
                ),
                params![n, blank(n.as_bytes())],
            )?;
        }
    }
    // Derived rows may quote the text (work-graph titles, derived titles):
    // every session is derived again from the blanked records.
    i.execute("DELETE FROM derive_state", [])?;
    Ok(())
}

/// Notes about the removed sessions: targets naming a session or one of its
/// messages go, and a note left with no target goes with its relations.
pub(super) fn remove_notes(
    a: &Transaction<'_>,
    sessions: &[String],
    messages: &[String],
    dry_run: bool,
) -> Result<u64> {
    let refs: HashSet<String> = sessions
        .iter()
        .map(|s| format!("session:{s}"))
        .chain(messages.iter().map(|m| format!("message:{m}")))
        .collect();
    // Tool calls are named `tool:<session>#<call>`.
    let tools: Vec<String> = sessions.iter().map(|s| format!("tool:{s}#")).collect();
    let about = |r: &str| refs.contains(r) || tools.iter().any(|p| r.starts_with(p.as_str()));
    let mut by_note: BTreeMap<String, (u64, Vec<String>)> = BTreeMap::new();
    let mut stmt = a.prepare("SELECT note_id, ref FROM note_target")?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
    for row in rows {
        let (note, r) = row?;
        let e = by_note.entry(note).or_default();
        e.0 += 1;
        if about(&r) {
            e.1.push(r);
        }
    }
    drop(stmt);
    let mut removed = 0;
    for (note, (all, hit)) in by_note {
        if hit.is_empty() {
            continue;
        }
        let whole = hit.len() as u64 == all;
        if whole {
            removed += 1;
        }
        if dry_run {
            continue;
        }
        for r in &hit {
            a.execute(
                "DELETE FROM note_target WHERE note_id = ?1 AND ref = ?2",
                params![note, r],
            )?;
        }
        if whole {
            a.execute("DELETE FROM note_relation WHERE note_id = ?1", [&note])?;
            a.execute("DELETE FROM note WHERE id = ?1", [&note])?;
        }
    }
    Ok(removed)
}

/// Delivered briefs that named a removed session: the session's own texts
/// in them are blanked, and the content hash follows the new text.
pub(super) fn redact_deliveries(
    a: &Transaction<'_>,
    sessions: &[String],
    texts: &BTreeSet<String>,
    dry_run: bool,
) -> Result<u64> {
    let rows: Vec<(String, Vec<u8>, String)> = a
        .prepare("SELECT id, body, refs_json FROM delivery")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    let mut n = 0;
    for (id, body, refs) in rows {
        if !sessions.iter().any(|s| refs.contains(s.as_str())) {
            continue;
        }
        let Ok(text) = unpack_text(&body) else {
            continue;
        };
        let mut new = text.clone();
        for s in texts.iter().filter(|s| s.len() >= MIN_BLANKED) {
            new = new.replace(s.as_str(), &blank(s.as_bytes()));
        }
        if new == text {
            continue;
        }
        n += 1;
        if dry_run {
            continue;
        }
        a.execute(
            "UPDATE delivery SET body = ?2, content_hash = ?3 WHERE id = ?1",
            params![
                id,
                zstd::bulk::compress(new.as_bytes(), 1)?,
                &blake3::hash(new.as_bytes()).as_bytes()[..]
            ],
        )?;
    }
    Ok(n)
}

/// Removes an input whole: its revisions and extents, and records its
/// locations so the loop never collects them again.
pub(super) fn remove_input(
    t: &Transactions<'_, '_>,
    input: &str,
    req: &ExciseRequest,
    refused: &mut RefusedJson,
    done: &mut Excised,
) -> Result<()> {
    let locations: Vec<Location> = t
        .archive
        .prepare(
            "SELECT DISTINCT machine, path, device, inode FROM observation
             WHERE input = ?1 AND path NOT LIKE 'zip:%' AND path NOT LIKE 'copy:%'",
        )?
        .query_map([input], |r| {
            Ok(Location {
                machine: r.get(0)?,
                path: r.get(1)?,
                device: r.get(2)?,
                inode: r.get(3)?,
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    refused.locations.extend(locations);
    let revisions: Vec<Option<Vec<u8>>> = t
        .archive
        .prepare("SELECT blake3 FROM revision WHERE input = ?1")?
        .query_map([input], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    refused
        .revisions
        .extend(revisions.into_iter().flatten().map(|h| hex(&h)));
    let blobs: Vec<Vec<u8>> = t
        .archive
        .prepare("SELECT DISTINCT blob FROM extent WHERE input = ?1")?
        .query_map([input], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    refused.blobs.extend(blobs.iter().map(|b| hex(b)));
    if req.dry_run {
        return Ok(());
    }
    insert_excised_observations(t, input, req, None)?;
    t.archive
        .execute("DELETE FROM extent WHERE input = ?1", [input])?;
    t.archive
        .execute("DELETE FROM revision WHERE input = ?1", [input])?;
    if let Some(pk) = input_pk(t.index, input)? {
        // Rows of other sessions from this input (an input scope's
        // shared sessions keep their other inputs' rows).
        for table in ["parse_state", "unrecognized", "parsed", "message_source"] {
            t.index
                .execute(&format!("DELETE FROM {table} WHERE input_pk = ?1"), [pk])?;
        }
        t.index
            .execute("UPDATE input_ref SET excised = 1 WHERE pk = ?1", [pk])?;
    }
    let _ = done;
    Ok(())
}

/// One `excised` observation per location of the input, naming the
/// excision.
pub(super) fn insert_excised_observations(
    t: &Transactions<'_, '_>,
    input: &str,
    req: &ExciseRequest,
    spans: Option<&serde_json::Value>,
) -> Result<()> {
    let at = now_ms();
    let rows: Vec<(String, String)> = t
        .archive
        .prepare("SELECT DISTINCT machine, path FROM observation WHERE input = ?1")?
        .query_map([input], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    for (n, (machine, path)) in rows.into_iter().enumerate() {
        t.archive.execute(
            "INSERT INTO observation(id, input, machine, path, class, excision, spans_json,
                 polylogue_version, seen_at)
             VALUES (?1, ?2, ?3, ?4, 'excised', ?5, ?6, ?7, ?8)",
            params![
                format!("{at:012x}-{:08x}-{n}-{input}", t.seq),
                input,
                machine,
                path,
                req.id,
                spans.map(ToString::to_string),
                env!("CARGO_PKG_VERSION"),
                at,
            ],
        )?;
    }
    Ok(())
}

/// Removes one session's rows from `index.db`.
pub(super) fn delete_session(
    t: &Transactions<'_, '_>,
    pk: i64,
    id: &str,
    dry_run: bool,
    removed_texts: &mut BTreeSet<Vec<u8>>,
    blobs: &mut BTreeSet<Vec<u8>>,
    done: &mut Excised,
) -> Result<()> {
    let i = t.index;
    let text_ids: Vec<i64> = i
        .prepare(
            "SELECT DISTINCT text_id FROM block WHERE session_pk = ?1 AND text_id IS NOT NULL",
        )?
        .query_map([pk], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let block_blobs: Vec<Vec<u8>> = i
        .prepare(
            "SELECT DISTINCT blob_hash FROM block WHERE session_pk = ?1 AND blob_hash IS NOT NULL",
        )?
        .query_map([pk], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    blobs.extend(block_blobs);
    if dry_run {
        for tid in text_ids {
            let shared: bool = i.query_row(
                "SELECT EXISTS (SELECT 1 FROM block WHERE text_id = ?1 AND session_pk <> ?2)",
                params![tid, pk],
                |r| r.get(0),
            )?;
            if !shared {
                let hash: Vec<u8> =
                    i.query_row("SELECT hash FROM text WHERE id = ?1", [tid], |r| r.get(0))?;
                removed_texts.insert(hash);
            }
        }
        let _ = done;
        return Ok(());
    }
    // Trigram entries before their tool calls.
    let calls: Vec<(i64, Option<String>, Option<String>)> = i
        .prepare("SELECT pk, command, path FROM tool_call WHERE session_pk = ?1")?
        .query_map([pk], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    for (cpk, command, path) in &calls {
        i.execute(
            "INSERT INTO tool_call_trigram(tool_call_trigram, rowid, command, path)
             VALUES ('delete', ?1, ?2, ?3)",
            params![cpk, command, path],
        )?;
    }
    // Every table that names the session, its messages or its tool calls,
    // whichever component added it.
    let tables = columns(i)?;
    for (table, cols) in &tables {
        if table == "session" || table == "message" || table == "tool_call" {
            continue;
        }
        if cols.contains("tool_call_pk") {
            i.execute(
                &format!(
                    "DELETE FROM {table} WHERE tool_call_pk IN
                     (SELECT pk FROM tool_call WHERE session_pk = ?1)"
                ),
                [pk],
            )?;
        }
    }
    for (table, cols) in &tables {
        if table == "session" || table == "message" || table == "tool_call" {
            continue;
        }
        if cols.contains("message_pk") && !cols.contains("session_pk") {
            i.execute(
                &format!(
                    "DELETE FROM {table} WHERE message_pk IN
                     (SELECT pk FROM message WHERE session_pk = ?1)"
                ),
                [pk],
            )?;
        }
    }
    i.execute("DELETE FROM tool_call WHERE session_pk = ?1", [pk])?;
    for (table, cols) in &tables {
        if table == "session" || table == "message" {
            continue;
        }
        if cols.contains("session_pk") {
            i.execute(&format!("DELETE FROM {table} WHERE session_pk = ?1"), [pk])?;
        }
    }
    i.execute(
        "DELETE FROM link WHERE parent_id = ?1 OR child_id = ?1",
        [id],
    )?;
    i.execute("DELETE FROM message WHERE session_pk = ?1", [pk])?;
    i.execute("DELETE FROM session WHERE pk = ?1", [pk])?;
    i.execute(
        "INSERT INTO removed_session(session_id, changed) VALUES (?1, ?2)
         ON CONFLICT(session_id) DO UPDATE SET changed = excluded.changed",
        params![id, t.counter],
    )?;
    // Texts no remaining block holds.
    for tid in text_ids {
        let used: bool = i.query_row(
            "SELECT EXISTS (SELECT 1 FROM block WHERE text_id = ?1)",
            [tid],
            |r| r.get(0),
        )?;
        if used {
            continue;
        }
        let (hash, indexed): (Vec<u8>, i64) =
            i.query_row("SELECT hash, indexed FROM text WHERE id = ?1", [tid], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })?;
        remove_text(i, tid, indexed)?;
        removed_texts.insert(hash);
    }
    Ok(())
}

pub(super) fn remove_text(i: &Transaction<'_>, id: i64, indexed: i64) -> Result<()> {
    if indexed & 1 != 0 {
        i.execute("DELETE FROM text_fts WHERE rowid = ?1", [id])?;
    }
    if indexed & 2 != 0 {
        i.execute("DELETE FROM output_fts WHERE rowid = ?1", [id])?;
    }
    i.execute("DELETE FROM search_pending WHERE text_id = ?1", [id])?;
    i.execute("DELETE FROM text WHERE id = ?1", [id])?;
    Ok(())
}

/// Ordinary tables in `index.db` and their column names.
pub(super) fn columns(i: &Transaction<'_>) -> Result<Vec<(String, HashSet<String>)>> {
    let names: Vec<String> = i
        .prepare(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND sql NOT LIKE 'CREATE VIRTUAL%'
             AND name NOT LIKE 'sqlite_%' AND name NOT LIKE '%_fts%' AND name NOT LIKE 'fts_%'
             AND name NOT LIKE 'tool_call_trigram%'",
        )?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    let mut out = Vec::new();
    for name in names {
        let cols: HashSet<String> = i
            .prepare(&format!("SELECT name FROM pragma_table_info('{name}')"))?
            .query_map([], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        out.push((name, cols));
    }
    Ok(out)
}

pub(super) fn session_pk(i: &Transaction<'_>, id: &str) -> Result<Option<i64>> {
    Ok(
        i.query_row("SELECT pk FROM session WHERE id = ?1", [id], |r| r.get(0))
            .optional()?,
    )
}

pub(super) fn input_pk(i: &Transaction<'_>, id: &str) -> Result<Option<i64>> {
    Ok(
        i.query_row("SELECT pk FROM input_ref WHERE id = ?1", [id], |r| r.get(0))
            .optional()?,
    )
}
