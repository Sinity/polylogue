//! Freeze affected sessions and their inputs before deleting rows for text reparse.
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
pub(in crate::excise) fn clear_for_reparse(
    t: &Transactions<'_, '_>,
    text: &str,
    holding: &BTreeSet<String>,
    dry_run: bool,
    removed_texts: &mut BTreeSet<Vec<u8>>,
    blobs: &mut BTreeSet<Vec<u8>>,
) -> Result<Vec<String>> {
    let i = t.index;
    i.execute_batch(
        "CREATE TEMP TABLE excision_reparse_text(id INTEGER PRIMARY KEY);
         CREATE TEMP TABLE excision_reparse_session(pk INTEGER PRIMARY KEY,id TEXT NOT NULL);
         CREATE TEMP TABLE excision_reparse_input(pk INTEGER PRIMARY KEY,id TEXT NOT NULL);",
    )?;
    // Start with matching current texts. Freed texts join this table only
    // after the complete session and input selections have been frozen.
    let mut stmt = i.prepare("SELECT id, hash, body FROM text")?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let hash: Vec<u8> = row.get(1)?;
        let body: Vec<u8> = row.get(2)?;
        if unpack_text(&body)?.contains(text) {
            removed_texts.insert(hash);
            i.prepare_cached("INSERT INTO temp.excision_reparse_text VALUES (?1)")?
                .execute([row.get::<_, i64>(0)?])?;
        }
    }
    drop(rows);
    drop(stmt);
    i.execute(
        "INSERT OR IGNORE INTO temp.excision_reparse_session
         SELECT s.pk,s.id FROM block b JOIN session s ON s.pk=b.session_pk
         WHERE b.text_id IN (SELECT id FROM temp.excision_reparse_text)",
        [],
    )?;
    let mut by_input = i.prepare(
        "INSERT OR IGNORE INTO temp.excision_reparse_session
         SELECT s.pk,s.id FROM session s WHERE s.pk IN (
             SELECT p.session_pk FROM parsed p JOIN input_ref r ON r.pk=p.input_pk WHERE r.id=?1
             UNION SELECT e.session_pk FROM event e JOIN event_source es ON es.event_id=e.id
                   JOIN input_ref r ON r.pk=es.input_pk WHERE r.id=?1
             UNION SELECT m.session_pk FROM message_source ms JOIN message m ON m.pk=ms.message_pk
                   JOIN input_ref r ON r.pk=ms.input_pk WHERE r.id=?1)",
    )?;
    for input in holding {
        by_input.execute([input])?;
    }
    drop(by_input);
    // Preserve all provenance before session removal prunes sources and
    // promotes copies. TEMP rows deliberately have no live-row foreign keys.
    i.execute(
        "INSERT OR IGNORE INTO temp.excision_reparse_input
         SELECT pk,id FROM input_ref WHERE pk IN (
             SELECT input_pk FROM parsed WHERE session_pk IN (SELECT pk FROM temp.excision_reparse_session)
             UNION SELECT input_pk FROM session_record WHERE session_pk IN (SELECT pk FROM temp.excision_reparse_session)
             UNION SELECT es.input_pk FROM event_source es JOIN event e ON e.id=es.event_id
                   WHERE e.session_pk IN (SELECT pk FROM temp.excision_reparse_session)
             UNION SELECT ms.input_pk FROM message_source ms JOIN message m ON m.pk=ms.message_pk
                   WHERE m.session_pk IN (SELECT pk FROM temp.excision_reparse_session)
             UNION SELECT p.input_pk FROM process p JOIN process_session ps ON ps.process_pk=p.pk
                   WHERE ps.session_pk IN (SELECT pk FROM temp.excision_reparse_session)
             UNION SELECT r.input_pk FROM run r JOIN run_session rs ON rs.run_pk=r.pk
                   WHERE rs.session_pk IN (SELECT pk FROM temp.excision_reparse_session)
             UNION SELECT input_pk FROM link WHERE child_pk IN (SELECT pk FROM temp.excision_reparse_session)
                   OR parent_id IN (SELECT id FROM temp.excision_reparse_session))", [],
    )?;
    for input in holding {
        i.prepare_cached(
            "INSERT OR IGNORE INTO temp.excision_reparse_input SELECT pk,id FROM input_ref WHERE id=?1",
        )?.execute([input])?;
    }
    crate::fault::check("excision_reparse.after_selection", "")?;
    if dry_run {
        return finish(i);
    }
    let mut selected =
        i.prepare("SELECT pk,id FROM temp.excision_reparse_session ORDER BY pk,id")?;
    let mut rows = selected.query([])?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(1)?;
        let (texts, session_blobs) = crate::session_row::remove(i, row.get(0)?, &id, t.counter)?;
        for text in texts {
            i.prepare_cached("INSERT OR IGNORE INTO temp.excision_reparse_text VALUES (?1)")?
                .execute([text])?;
        }
        blobs.extend(session_blobs);
        crate::fault::check("excision_reparse.after_session", "")?;
    }
    drop(rows);
    drop(selected);
    // Clear input-owned rows and their dependents through the same frozen set.
    let tables = crate::schema::index_columns(i)?;
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
                         (SELECT pk FROM {table} WHERE input_pk IN
                          (SELECT pk FROM temp.excision_reparse_input))"
                        ),
                        [],
                    )?;
                }
            }
        }
        i.execute(
            &format!(
                "DELETE FROM {table} WHERE input_pk IN (SELECT pk FROM temp.excision_reparse_input)"
            ),
            [],
        )?;
    }
    let mut texts = i.prepare("SELECT id FROM temp.excision_reparse_text ORDER BY id")?;
    let mut rows = texts.query([])?;
    while let Some(row) = rows.next()? {
        let tid: i64 = row.get(0)?;
        let indexed: Option<i64> = i
            .query_row(
                "SELECT indexed FROM text WHERE id=?1
             AND NOT EXISTS (SELECT 1 FROM block WHERE text_id=?1)
             AND NOT EXISTS (SELECT 1 FROM version_block WHERE text_id=?1)",
                [tid],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(indexed) = indexed {
            crate::text_rows::remove(i, tid, indexed)?;
        }
    }
    drop(rows);
    drop(texts);
    finish(i)
}

fn finish(i: &rusqlite::Connection) -> Result<Vec<String>> {
    let inputs = i
        .prepare("SELECT id FROM temp.excision_reparse_input ORDER BY pk,id")?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    i.execute_batch(
        "DROP TABLE temp.excision_reparse_text;
         DROP TABLE temp.excision_reparse_session;
         DROP TABLE temp.excision_reparse_input;",
    )?;
    Ok(inputs)
}

#[cfg(test)]
mod tests;
