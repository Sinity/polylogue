//! Excision in `index.db`, `archive.db` and `embeddings.db`: a session's
//! rows from every table, a text from the search indexes, and the notes,
//! deliveries and inputs that held it.

use super::*;

/// Text scope in `index.db`: rather than overwriting the columns that may
/// hold the text, every row parsed from records that held it goes, and the
/// inputs are parsed again from their overwritten stored bytes (the caller
/// runs that parse). The affected sessions are those parsed from an input
/// whose bytes held the text, and those whose texts hold it; each loses
/// every row, and each input it was built from is parsed again, so it
/// comes back whole and clean. Returns the inputs to parse again. Texts
/// that held the text are recorded so their embeddings go; the session's
/// other texts come back with the same hash and keep theirs.
#[allow(clippy::too_many_arguments)]
pub(super) fn clear_for_reparse(
    t: &Transactions<'_, '_>,
    text: &str,
    holding: &BTreeSet<String>,
    dry_run: bool,
    removed_texts: &mut BTreeSet<Vec<u8>>,
    blobs: &mut BTreeSet<Vec<u8>>,
) -> Result<Vec<String>> {
    let i = t.index;
    // Texts holding it, and the sessions whose blocks show them.
    let mut stmt = i.prepare("SELECT id, hash, body FROM text")?;
    let mut rows = stmt.query([])?;
    let mut held = Vec::new();
    while let Some(row) = rows.next()? {
        let id: i64 = row.get(0)?;
        let hash: Vec<u8> = row.get(1)?;
        let body: Vec<u8> = row.get(2)?;
        if unpack_text(&body)?.contains(text) {
            removed_texts.insert(hash);
            held.push(id);
        }
    }
    drop(rows);
    drop(stmt);
    let mut sessions: BTreeSet<(i64, String)> = BTreeSet::new();
    let mut by_text = i.prepare(
        "SELECT DISTINCT s.pk, s.id FROM block b JOIN session s ON s.pk = b.session_pk
         WHERE b.text_id = ?1",
    )?;
    for id in &held {
        for row in by_text.query_map([id], |r| Ok((r.get(0)?, r.get(1)?)))? {
            sessions.insert(row?);
        }
    }
    drop(by_text);
    let mut by_input = i.prepare(
        "SELECT DISTINCT s.pk, s.id FROM session s WHERE s.pk IN (
             SELECT p.session_pk FROM parsed p JOIN input_ref r ON r.pk = p.input_pk WHERE r.id = ?1
             UNION SELECT e.session_pk FROM event e JOIN event_source es ON es.event_id=e.id JOIN input_ref r ON r.pk = es.input_pk WHERE r.id = ?1
             UNION SELECT m.session_pk FROM message_source ms JOIN message m ON m.pk = ms.message_pk
                   JOIN input_ref r ON r.pk = ms.input_pk WHERE r.id = ?1)",
    )?;
    for input in holding {
        for row in by_input.query_map([input], |r| Ok((r.get(0)?, r.get(1)?)))? {
            sessions.insert(row?);
        }
    }
    drop(by_input);
    // Every input the affected sessions were built from; the rows these
    // inputs gave other sessions stay, and parsing them again rewrites
    // them unchanged (every write is an upsert).
    let mut inputs: BTreeSet<(i64, String)> = BTreeSet::new();
    let mut of_session = i.prepare(
        "SELECT pk, id FROM input_ref WHERE pk IN (
             SELECT input_pk FROM parsed WHERE session_pk = ?1
             UNION SELECT input_pk FROM session_record WHERE session_pk = ?1
             UNION SELECT es.input_pk FROM event_source es JOIN event e ON e.id=es.event_id WHERE e.session_pk = ?1
             UNION SELECT ms.input_pk FROM message_source ms JOIN message m ON m.pk = ms.message_pk
                   WHERE m.session_pk = ?1
             UNION SELECT p.input_pk FROM process p JOIN process_session ps ON ps.process_pk = p.pk
                   WHERE ps.session_pk = ?1
             UNION SELECT r.input_pk FROM run r JOIN run_session rs ON rs.run_pk = r.pk
                   WHERE rs.session_pk = ?1
             UNION SELECT l.input_pk FROM link l WHERE l.child_pk = ?1
                   OR l.parent_id = (SELECT id FROM session WHERE pk = ?1))",
    )?;
    for (pk, _) in &sessions {
        for row in of_session.query_map([pk], |r| Ok((r.get(0)?, r.get(1)?)))? {
            inputs.insert(row?);
        }
    }
    drop(of_session);
    for input in holding {
        if let Some(pk) = input_pk(i, input)? {
            inputs.insert((pk, input.clone()));
        }
    }
    if dry_run {
        return Ok(inputs.into_iter().map(|(_, id)| id).collect());
    }
    let mut freed = BTreeSet::new();
    for (pk, id) in &sessions {
        let (text_ids, session_blobs) = crate::session_row::remove(i, *pk, id, t.counter)?;
        freed.extend(text_ids);
        blobs.extend(session_blobs);
    }
    // Rows of these inputs that belong to no one session (links,
    // processes, runs, repository lookups, parse positions), with the
    // rows that refer to them; parsing the inputs again writes them anew.
    let tables = crate::schema::index_columns(i)?;
    let pks: Vec<i64> = inputs.iter().map(|(pk, _)| *pk).collect();
    let list = pks
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    for (table, cols) in &tables {
        if table == "input_ref"
            || !cols.contains("input_pk")
            || cols.contains("session_pk")
            || cols.contains("message_pk")
        {
            continue;
        }
        let reference = format!("{table}_pk");
        if cols.contains("pk") {
            for (other, other_cols) in &tables {
                if other_cols.contains(&reference) {
                    i.execute(
                        &format!(
                            "DELETE FROM {other} WHERE {reference} IN
                             (SELECT pk FROM {table} WHERE input_pk IN ({list}))"
                        ),
                        [],
                    )?;
                }
            }
        }
        i.execute(
            &format!("DELETE FROM {table} WHERE input_pk IN ({list})"),
            [],
        )?;
    }
    // Texts no block holds any more, and any text left holding it.
    freed.extend(held);
    for tid in freed {
        let row: Option<i64> = i
            .query_row(
                "SELECT indexed FROM text WHERE id = ?1
                 AND NOT EXISTS (SELECT 1 FROM block WHERE text_id = ?1)
                 AND NOT EXISTS (SELECT 1 FROM version_block WHERE text_id = ?1)",
                [tid],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(indexed) = row {
            crate::text_rows::remove(i, tid, indexed)?;
        }
    }
    Ok(inputs.into_iter().map(|(_, id)| id).collect())
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
/// `messages` is the plan's sorted list of exact IDs.
pub(super) fn redact_deliveries(
    a: &Transaction<'_>,
    sessions: &[String],
    messages: &[String],
    texts: &SessionTexts,
    dry_run: bool,
) -> Result<u64> {
    let mut after: Option<String> = None;
    let mut n = 0;
    loop {
        let row = |r: &rusqlite::Row<'_>| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?));
        let next = match &after {
            Some(after) => a
                .prepare_cached(
                    "SELECT id,refs_json FROM delivery WHERE id>?1 ORDER BY id LIMIT 1",
                )?
                .query_row([after], row)
                .optional()?,
            None => a
                .prepare_cached("SELECT id,refs_json FROM delivery ORDER BY id LIMIT 1")?
                .query_row([], row)
                .optional()?,
        };
        let Some((id, refs)) = next else {
            break;
        };
        after = Some(id.clone());
        let refs: Vec<String> = serde_json::from_str(&refs)?;
        let referenced = refs.iter().any(|reference| {
            sessions
                .iter()
                .any(|s| reference == s || reference.strip_prefix("session:") == Some(s.as_str()))
                || reference
                    .strip_prefix("message:")
                    .is_some_and(|id| messages.binary_search_by(|m| m.as_str().cmp(id)).is_ok())
        });
        if !referenced {
            continue;
        }
        let body: Vec<u8> = a
            .prepare_cached("SELECT body FROM delivery WHERE id=?1")?
            .query_row([&id], |r| r.get(0))?;
        let text = unpack_text(&body)?;
        let mut new = text.clone();
        texts.visit(MIN_BLANKED, |s| {
            new = new.replace(s, &blank(s.as_bytes()));
            Ok(())
        })?;
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
                crate::encoding::pack_text(&new)?,
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
    removed_texts: &mut BTreeSet<Vec<u8>>,
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
    let mut texts = Vec::new();
    if let Some(pk) = input_pk(t.index, input)? {
        let media:Vec<Vec<u8>>=t.index.prepare("SELECT DISTINCT v.blob_hash FROM version_block v JOIN message_source s ON s.version_hash=v.version_hash WHERE s.input_pk=?1 AND v.blob_hash IS NOT NULL")?.query_map([pk],|r|r.get(0))?.collect::<rusqlite::Result<_>>()?;
        refused.blobs.extend(media.iter().map(|hash| hex(hash)));

        texts=t.index.prepare("SELECT DISTINCT v.text_id FROM version_block v JOIN message_source s ON s.version_hash=v.version_hash WHERE s.input_pk=?1 AND v.text_id IS NOT NULL")?.query_map([pk],|r|r.get::<_,i64>(0))?.collect::<rusqlite::Result<_>>()?;
        let messages: Vec<i64> = t
            .index
            .prepare("SELECT DISTINCT message_pk FROM message_source WHERE input_pk=?1")?
            .query_map([pk], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        // Rows of other sessions from this input (an input scope's
        // shared sessions keep their other inputs' rows).
        for table in [
            "parse_state",
            "unrecognized",
            "parsed",
            "message_source",
            "event_source",
            "link_source",
            "snapshot",
        ] {
            t.index
                .execute(&format!("DELETE FROM {table} WHERE input_pk = ?1"), [pk])?;
        }
        t.index
            .execute("UPDATE input_ref SET excised = 1 WHERE pk = ?1", [pk])?;
        crate::writer::promote_current(t.index, &messages, t.counter)?;
    }
    let _ = done;
    crate::versions::prune(t.index)?;
    for id in texts {
        let orphan:Option<(Vec<u8>,i64)>=t.index.query_row("SELECT hash,indexed FROM text WHERE id=?1 AND NOT EXISTS(SELECT 1 FROM block WHERE text_id=?1) AND NOT EXISTS(SELECT 1 FROM version_block WHERE text_id=?1)",[id],|r|Ok((r.get(0)?,r.get(1)?))).optional()?;
        if let Some((hash, indexed)) = orphan {
            crate::text_rows::remove(t.index, id, indexed)?;
            removed_texts.insert(hash);
        }
    }
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
    for (machine, path) in rows {
        t.archive.execute(
            "INSERT INTO observation(id, input, machine, path, class, excision, spans_json,
                 polylogue_version, seen_at, commit_seq)
             VALUES (?1, ?2, ?3, ?4, 'excised', ?5, ?6, ?7, ?8, ?9)",
            params![
                crate::writer::observation_id(t.archive, t.seq, input)?,
                input,
                machine,
                path,
                req.id,
                spans.map(ToString::to_string),
                env!("CARGO_PKG_VERSION"),
                at,
                t.seq,
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
) -> Result<()> {
    let i = t.index;
    if dry_run {
        let hashes: Vec<Vec<u8>> = i
            .prepare(
                "SELECT DISTINCT t.hash FROM block b JOIN text t ON t.id = b.text_id
                 WHERE b.session_pk = ?1 AND NOT EXISTS
                   (SELECT 1 FROM block o WHERE o.text_id = b.text_id AND o.session_pk <> ?1)",
            )?
            .query_map([pk], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        removed_texts.extend(hashes);
        return Ok(());
    }
    let (text_ids, session_blobs) = crate::session_row::remove(i, pk, id, t.counter)?;
    blobs.extend(session_blobs);
    i.execute(
        "INSERT INTO removed_session(session_id, changed) VALUES (?1, ?2)
         ON CONFLICT(session_id) DO UPDATE SET changed = excluded.changed",
        params![id, t.counter],
    )?;
    // Texts no remaining block holds.
    for tid in text_ids {
        let row: Option<(Vec<u8>, i64)> = i
            .query_row(
                "SELECT hash, indexed FROM text WHERE id = ?1
                 AND NOT EXISTS (SELECT 1 FROM block WHERE text_id = ?1)
                 AND NOT EXISTS (SELECT 1 FROM version_block WHERE text_id = ?1)",
                [tid],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        if let Some((hash, indexed)) = row {
            crate::text_rows::remove(i, tid, indexed)?;
            removed_texts.insert(hash);
        }
    }
    Ok(())
}

/// Deletes one session's rows from every `index.db` table that names the
/// session, its messages or its tool calls, whichever component added the
/// table. Returns the texts its blocks held and the blobs they named.
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
