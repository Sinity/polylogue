//! Clear the sessions built from changed inputs so they are parsed again.
use super::*;

/// Rather than patching the columns that may hold removed content, every
/// row parsed from a changed input goes, and the inputs are parsed again
/// from their stored bytes (the caller runs that parse). The affected
/// sessions are those parsed from a `holding` input, and, for a text
/// excision, those whose texts hold the text; each loses every row, and
/// each input it was built from is parsed again, so it comes back whole
/// from the bytes that remain. Returns the inputs to parse again. Texts
/// that held the text are recorded so their embeddings go; the session's
/// other texts come back with the same hash and keep theirs.
pub(in crate::excise) fn clear_for_reparse(
    t: &Transactions<'_, '_>,
    text: Option<&str>,
    holding: &BTreeSet<String>,
    dry_run: bool,
    hashes: &RemovalHashes<'_>,
) -> Result<Vec<String>> {
    let i = t.index;
    if !dry_run {
        for input in holding {
            crate::assets::invalidate_metadata(i, input)?;
        }
    }
    i.execute_batch(
        "CREATE TEMP TABLE excision_reparse_text(id INTEGER PRIMARY KEY);
         CREATE TEMP TABLE excision_reparse_session(pk INTEGER PRIMARY KEY,id TEXT NOT NULL);
         CREATE TEMP TABLE excision_reparse_document(pk INTEGER PRIMARY KEY,id TEXT NOT NULL);
         CREATE TEMP TABLE excision_reparse_input(pk INTEGER PRIMARY KEY,id TEXT NOT NULL);",
    )?;
    // Start with matching current texts. Freed texts join this table only
    // after the complete session and input selections are made.
    if let Some(text) = text {
        let mut stmt = i.prepare("SELECT id, hash, body FROM text")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let body: Vec<u8> = row.get(2)?;
            if unpack_text(&body)?.contains(text) {
                hashes.text(&row.get::<_, Vec<u8>>(1)?)?;
                i.prepare_cached("INSERT INTO temp.excision_reparse_text VALUES (?1)")?
                    .execute([row.get::<_, i64>(0)?])?;
            }
        }
    }
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
             UNION SELECT e.session_pk FROM event e
                   JOIN input_ref r ON r.pk=e.input_pk WHERE r.id=?1
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
             UNION SELECT input_pk FROM event
                   WHERE session_pk IN (SELECT pk FROM temp.excision_reparse_session)
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
    i.execute("INSERT OR IGNORE INTO temp.excision_reparse_document SELECT DISTINCT d.pk,d.id FROM document d JOIN document_block b ON b.document_pk=d.pk WHERE b.text_id IN(SELECT id FROM temp.excision_reparse_text)",[])?;
    for input in holding {
        i.execute("INSERT OR IGNORE INTO temp.excision_reparse_document SELECT d.pk,d.id FROM document d JOIN document_source s ON s.document_pk=d.pk JOIN input_ref r ON r.pk=s.input_pk WHERE r.id=?1",[input])?;
    }
    // Documents from the inputs parsed again are rebuilt with them, from every
    // input that holds them.
    i.execute("INSERT OR IGNORE INTO temp.excision_reparse_document SELECT d.pk,d.id FROM document d JOIN document_source s ON s.document_pk=d.pk WHERE s.input_pk IN(SELECT pk FROM temp.excision_reparse_input)",[])?;
    i.execute("INSERT OR IGNORE INTO temp.excision_reparse_input SELECT r.pk,r.id FROM input_ref r JOIN document_source s ON s.input_pk=r.pk WHERE s.document_pk IN(SELECT pk FROM temp.excision_reparse_document)",[])?;
    crate::fault::check("excision_reparse.after_selection", "")?;
    if dry_run {
        return finish(i);
    }
    let mut selected =
        i.prepare("SELECT pk,id FROM temp.excision_reparse_session ORDER BY pk,id")?;
    let mut rows = selected.query([])?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(1)?;
        crate::session_row::remove_with_references(i, row.get(0)?, &id, |references| {
            references.visit_texts(|text| {
                i.prepare_cached("INSERT OR IGNORE INTO temp.excision_reparse_text VALUES (?1)")?
                    .execute([text])?;
                Ok(())
            })?;
            references.visit_blobs(|hash| hashes.blob_candidate(hash))
        })?;
        crate::fault::check("excision_reparse.after_session", "")?;
    }
    drop(rows);
    drop(selected);
    i.execute("INSERT OR IGNORE INTO temp.excision_reparse_text SELECT text_id FROM document_block WHERE document_pk IN(SELECT pk FROM temp.excision_reparse_document) AND text_id IS NOT NULL",[])?;
    i.execute(
        "DELETE FROM document WHERE pk IN(SELECT pk FROM temp.excision_reparse_document)",
        [],
    )?;
    i.execute(
        "DELETE FROM event WHERE session_pk IS NULL AND input_pk IN (SELECT pk FROM temp.excision_reparse_input)",
        [],
    )?;
    // Clear input-owned rows and their dependents through the same selection.
    let tables = crate::schema::index_columns(i)?;
    for (table, cols) in &tables {
        if table == "document"
            || table == "input_ref"
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
             AND NOT EXISTS (SELECT 1 FROM document_block WHERE text_id=?1)",
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
         DROP TABLE temp.excision_reparse_document;
         DROP TABLE temp.excision_reparse_input;",
    )?;
    Ok(inputs)
}

#[cfg(test)]
mod tests;
