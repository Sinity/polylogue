//! Excision's reads and writes: which inputs hold a session or document, the
//! refusal of anything shared, and the removal of sessions, documents, notes
//! and whole inputs from `index.db` and `archive.db`.

use super::*;

pub(super) fn session_pk(index: &Connection, id: &str) -> Result<Option<i64>> {
    Ok(index
        .query_row("SELECT pk FROM session WHERE id = ?1", [id], |r| r.get(0))
        .optional()?)
}

pub(super) fn input_pk(index: &Connection, id: &str) -> Result<Option<i64>> {
    Ok(index
        .query_row("SELECT pk FROM input_ref WHERE id = ?1", [id], |r| r.get(0))
        .optional()?)
}

fn strings(index: &Connection, sql: &str, key: i64) -> Result<Vec<String>> {
    Ok(index
        .prepare_cached(sql)?
        .query_map([key], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?)
}

fn keyed(index: &Connection, sql: &str, key: i64) -> Result<Vec<(i64, String)>> {
    Ok(index
        .prepare_cached(sql)?
        .query_map([key], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?)
}

/// Every input a session's rows came from.
pub(super) fn inputs_of_session(index: &Connection, pk: i64) -> Result<Vec<String>> {
    strings(
        index,
        "SELECT id FROM input_ref WHERE pk IN (
            SELECT input_pk FROM parsed WHERE session_pk = ?1
            UNION SELECT s.input_pk FROM message_source s JOIN message m ON m.pk = s.message_pk
                  WHERE m.session_pk = ?1
            UNION SELECT input_pk FROM event WHERE session_pk = ?1)",
        pk,
    )
}

pub(super) fn inputs_of_document(index: &Connection, pk: i64) -> Result<Vec<String>> {
    strings(
        index,
        "SELECT r.id FROM document_source s JOIN input_ref r ON r.pk = s.input_pk
         WHERE s.document_pk = ?1",
        pk,
    )
}

/// The sessions an input's rows belong to.
pub(super) fn sessions_of_input(index: &Connection, pk: i64) -> Result<Vec<(i64, String)>> {
    keyed(
        index,
        "SELECT pk, id FROM session WHERE pk IN (
            SELECT session_pk FROM parsed WHERE input_pk = ?1
            UNION SELECT m.session_pk FROM message_source s JOIN message m ON m.pk = s.message_pk
                  WHERE s.input_pk = ?1
            UNION SELECT session_pk FROM event WHERE input_pk = ?1)",
        pk,
    )
}

pub(super) fn documents_of_input(index: &Connection, pk: i64) -> Result<Vec<(i64, String)>> {
    keyed(
        index,
        "SELECT pk, id FROM document WHERE pk IN
            (SELECT document_pk FROM document_source WHERE input_pk = ?1)",
        pk,
    )
}

/// Refuses an excision that would remove anything it does not name: an
/// input inside an export ZIP, an input that also holds another session or
/// document (or, unless the input itself is named, records of no session), a
/// removed session or document that also comes from an input the excision
/// keeps, or a delivered brief that holds a removed session.
pub(super) fn refuse_shared(
    archive: &Connection,
    index: &Connection,
    plan: &Plan,
    input_named: bool,
) -> Result<()> {
    crate::zip_members::refuse_inside_zip(archive, &plan.inputs)?;
    let refuse = |target: &str, reason: String| -> Result<()> {
        Err(StoreError::ExcisionRefused {
            target: target.to_owned(),
            reason,
        })
    };
    let sessions: BTreeSet<&str> = plan.sessions.iter().map(|(_, id)| id.as_str()).collect();
    let documents: BTreeSet<&str> = plan.documents.iter().map(|(_, id)| id.as_str()).collect();
    for input in &plan.inputs {
        let Some(pk) = input_pk(index, input)? else {
            continue;
        };
        for (_, session) in sessions_of_input(index, pk)? {
            if !sessions.contains(session.as_str()) {
                return refuse(
                    input,
                    format!(
                        "it also holds session {session}, and excising part of a file would rewrite it"
                    ),
                );
            }
        }
        let unbound: bool = index.query_row(
            "SELECT EXISTS(SELECT 1 FROM event WHERE input_pk = ?1 AND session_pk IS NULL)",
            [pk],
            |r| r.get(0),
        )?;
        if unbound && !input_named {
            return refuse(
                input,
                "it also holds records of no session, and excising part of a file would rewrite it"
                    .into(),
            );
        }
        for (_, document) in documents_of_input(index, pk)? {
            if !documents.contains(document.as_str()) {
                return refuse(
                    input,
                    format!(
                        "it also holds document {document}, and excising part of a file would rewrite it"
                    ),
                );
            }
        }
    }
    for (pk, id) in &plan.sessions {
        for input in inputs_of_session(index, *pk)? {
            if !plan.inputs.contains(&input) {
                return refuse(
                    &input,
                    format!("session {id} also comes from this input, which the excision keeps"),
                );
            }
        }
    }
    for (pk, id) in &plan.documents {
        for input in inputs_of_document(index, *pk)? {
            if !plan.inputs.contains(&input) {
                return refuse(
                    &input,
                    format!("document {id} also comes from this input, which the excision keeps"),
                );
            }
        }
    }
    if !plan.sessions.is_empty() {
        let mut references = Vec::new();
        for (_, id) in &plan.sessions {
            references.push(format!("session:{id}"));
            references.push(id.clone());
        }
        for id in message_ids(index, &plan.sessions)? {
            references.push(format!("message:{id}"));
        }
        let delivered: Option<String> = archive
            .query_row(
                "SELECT d.id FROM delivery d, json_each(d.refs_json) r
                 WHERE r.value IN (SELECT value FROM json_each(?1)) LIMIT 1",
                [serde_json::to_string(&references)?],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(delivery) = delivered {
            return refuse(
                &format!("delivery:{delivery}"),
                "this delivered brief holds the session's text, and excising it would rewrite the brief"
                    .into(),
            );
        }
    }
    Ok(())
}

/// The message IDs of `sessions`.
pub(super) fn message_ids(
    index: &Connection,
    sessions: &[(i64, String)],
) -> Result<BTreeSet<String>> {
    let mut out = BTreeSet::new();
    for (pk, _) in sessions {
        out.extend(strings(
            index,
            "SELECT id FROM message WHERE session_pk = ?1",
            *pk,
        )?);
    }
    Ok(out)
}

/// Removes a text row no block holds any more, and its search entries;
/// returns its hash.
fn remove_text_if_unused(index: &Connection, id: i64) -> Result<Option<Vec<u8>>> {
    let unused: Option<(Vec<u8>, i64)> = index
        .query_row(
            "SELECT hash, indexed FROM text WHERE id = ?1
             AND NOT EXISTS(SELECT 1 FROM block WHERE text_id = ?1)
             AND NOT EXISTS(SELECT 1 FROM document_block WHERE text_id = ?1)",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    let Some((hash, indexed)) = unused else {
        return Ok(None);
    };
    crate::text_rows::remove(index, id, indexed)?;
    Ok(Some(hash))
}

/// Removes one session's rows from `index.db`.
pub(super) fn remove_session(
    t: &Transactions<'_, '_>,
    pk: i64,
    id: &str,
    removed_texts: &mut BTreeSet<Vec<u8>>,
    blobs: &mut BTreeSet<BlobHash>,
) -> Result<()> {
    let i = t.index;
    crate::session_row::remove_with_references(i, pk, id, |references| {
        references.visit_blobs(|hash| {
            blobs.insert(blob_hash(hash)?);
            Ok(())
        })?;
        i.execute(
            "INSERT INTO removed_session(session_id, changed) VALUES (?1, ?2)
             ON CONFLICT(session_id) DO UPDATE SET changed = excluded.changed",
            params![id, t.counter],
        )?;
        references.visit_texts(|text| {
            removed_texts.extend(remove_text_if_unused(i, text)?);
            Ok(())
        })
    })
}

/// Removes one document's rows from `index.db`.
pub(super) fn remove_document(
    t: &Transactions<'_, '_>,
    pk: i64,
    removed_texts: &mut BTreeSet<Vec<u8>>,
    blobs: &mut BTreeSet<BlobHash>,
) -> Result<()> {
    let rows: Vec<(Option<i64>, Option<Vec<u8>>)> = t
        .index
        .prepare("SELECT text_id, blob_hash FROM document_block WHERE document_pk = ?1")?
        .query_map([pk], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    t.index
        .execute("DELETE FROM document WHERE pk = ?1", [pk])?;
    for (text, media) in rows {
        if let Some(text) = text {
            removed_texts.extend(remove_text_if_unused(t.index, text)?);
        }
        if let Some(media) = media {
            blobs.insert(blob_hash(&media)?);
        }
    }
    Ok(())
}

/// The note targets naming a removed session, one of its messages or one
/// of its tool calls.
pub(crate) fn note_references(
    archive: &Connection,
    sessions: &[String],
    messages: &BTreeSet<String>,
) -> Result<Vec<String>> {
    if sessions.is_empty() {
        return Ok(Vec::new());
    }
    let about = |reference: &str| {
        if let Some(id) = reference.strip_prefix("session:") {
            return sessions.iter().any(|s| s == id);
        }
        if let Some(id) = reference.strip_prefix("message:") {
            return messages.contains(id);
        }
        reference.strip_prefix("tool:").is_some_and(|rest| {
            sessions.iter().any(|session| {
                rest.strip_prefix(session.as_str())
                    .is_some_and(|suffix| suffix.starts_with('#'))
            })
        })
    };
    Ok(archive
        .prepare("SELECT DISTINCT ref FROM note_target")?
        .query_map([], |r| r.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?
        .into_iter()
        .filter(|reference| about(reference))
        .collect())
}

/// Removes the note targets naming `references`; a note left without
/// targets goes whole. Returns the notes that went.
pub(crate) fn remove_notes_about(
    archive: &Connection,
    references: &[String],
) -> Result<Vec<String>> {
    let mut notes = BTreeSet::new();
    for reference in references {
        notes.extend(
            archive
                .prepare_cached("SELECT note_id FROM note_target WHERE ref = ?1")?
                .query_map([reference], |r| r.get::<_, String>(0))?
                .collect::<rusqlite::Result<Vec<_>>>()?,
        );
        archive.execute("DELETE FROM note_target WHERE ref = ?1", [reference])?;
    }
    let mut removed = Vec::new();
    for note in notes {
        let remains: bool = archive.query_row(
            "SELECT EXISTS(SELECT 1 FROM note_target WHERE note_id = ?1)",
            [&note],
            |r| r.get(0),
        )?;
        if !remains {
            archive.execute("DELETE FROM note_relation WHERE note_id = ?1", [&note])?;
            archive.execute("DELETE FROM note WHERE id = ?1", [&note])?;
            removed.push(note);
        }
    }
    Ok(removed)
}

/// What removing an input whole removed, for the refusals of later reads.
pub(super) struct RemovedInput {
    pub blobs: Vec<BlobHash>,
    pub locations: Vec<(String, String)>,
}

/// Removes an input whole: its revisions and extents, and every index row
/// it owns; its locations are never collected again. Its sessions and
/// documents are removed before.
pub(super) fn remove_input(
    t: &Transactions<'_, '_>,
    excision: &str,
    input: &str,
) -> Result<RemovedInput> {
    let locations: Vec<(String, String)> = t
        .archive
        .prepare(
            "SELECT DISTINCT machine, path FROM observation WHERE input = ?1
             AND class <> 'excised' ORDER BY machine, path",
        )?
        .query_map([input], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    let stored: Vec<Vec<u8>> = t
        .archive
        .prepare("SELECT DISTINCT blob FROM extent WHERE input = ?1")?
        .query_map([input], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    for (machine, path) in &locations {
        t.archive.execute(
            "INSERT INTO observation(id, input, machine, path, class, excision,
                 polylogue_version, seen_at, commit_seq)
             VALUES (?1, ?2, ?3, ?4, 'excised', ?5, ?6, ?7, 0)",
            params![
                crate::writer::observation_id(t.archive, t.seq, input)?,
                input,
                machine,
                path,
                excision,
                env!("CARGO_PKG_VERSION"),
                now_ms(),
            ],
        )?;
    }
    t.archive
        .execute("DELETE FROM extent WHERE input = ?1", [input])?;
    t.archive
        .execute("DELETE FROM revision WHERE input = ?1", [input])?;
    crate::assets::invalidate_metadata(t.index, input)?;
    t.index
        .execute("DELETE FROM import_member WHERE input = ?1", [input])?;
    if let Some(pk) = input_pk(t.index, input)? {
        // Rows of other kinds the input owns (events without a session,
        // parse progress), after the rows that refer to them.
        let tables = crate::schema::index_columns(t.index)?;
        for (table, cols) in &tables {
            if table == "input_ref" || !cols.contains("input_pk") {
                continue;
            }
            let reference = format!("{table}_pk");
            if cols.contains("pk") {
                for (other, other_cols) in &tables {
                    if other_cols.contains(&reference) {
                        t.index.execute(
                            &format!(
                                "DELETE FROM {other} WHERE {reference} IN
                                 (SELECT pk FROM {table} WHERE input_pk = ?1)"
                            ),
                            [pk],
                        )?;
                    }
                }
            }
            t.index
                .execute(&format!("DELETE FROM {table} WHERE input_pk = ?1"), [pk])?;
        }
        t.index
            .execute("UPDATE input_ref SET excised = 1 WHERE pk = ?1", [pk])?;
    }
    Ok(RemovedInput {
        blobs: stored.iter().map(|h| blob_hash(h)).collect::<Result<_>>()?,
        locations,
    })
}

/// Whether no stored input and no index row uses the blob.
pub(super) fn blob_unused(t: &Transactions<'_, '_>, hash: &BlobHash) -> Result<bool> {
    let used: bool = t.archive.query_row(
        "SELECT EXISTS(SELECT 1 FROM extent WHERE blob = ?1)",
        [&hash.0[..]],
        |r| r.get(0),
    )? || t.index.query_row(
        "SELECT EXISTS(SELECT 1 FROM block WHERE blob_hash = ?1)
             OR EXISTS(SELECT 1 FROM document_block WHERE blob_hash = ?1)",
        [&hash.0[..]],
        |r| r.get(0),
    )?;
    Ok(!used)
}

fn blob_hash(bytes: &[u8]) -> Result<BlobHash> {
    Ok(BlobHash(bytes.try_into().map_err(|_| {
        StoreError::Invalid("invalid blob hash".into())
    })?))
}
