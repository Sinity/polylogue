//! Excision in `index.db`, `archive.db` and `embeddings.db`: a session's
//! rows from every table, a text from the search indexes, and the
//! deliveries and inputs that held it.

use super::*;

mod reparse;
pub(super) use reparse::clear_for_reparse;

/// Delivered briefs that named a removed session: the session's own texts
/// in them are blanked, and the content hash follows the new text.
/// Exact message selection remains available after the index rows are gone.
pub(super) fn redact_deliveries(
    a: &Transaction<'_>,
    sessions: &[String],
    messages: &SelectedMessages,
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
        let mut referenced = false;
        for reference in &refs {
            if sessions
                .iter()
                .any(|s| reference == s || reference.strip_prefix("session:") == Some(s.as_str()))
                || (match reference.strip_prefix("message:") {
                    Some(id) => messages.contains(id)?,
                    None => false,
                })
            {
                referenced = true;
                break;
            }
        }
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
        let media:Vec<Vec<u8>>=t.index.prepare("SELECT DISTINCT v.blob_hash FROM version_block v JOIN message_version mv ON mv.pk=v.version_pk JOIN message_source s ON s.version_hash=mv.hash WHERE s.input_pk=?1 AND v.blob_hash IS NOT NULL")?.query_map([pk],|r|r.get(0))?.collect::<rusqlite::Result<_>>()?;
        refused.blobs.extend(media.iter().map(|hash| hex(hash)));

        texts=t.index.prepare("SELECT DISTINCT v.text_id FROM version_block v JOIN message_version mv ON mv.pk=v.version_pk JOIN message_source s ON s.version_hash=mv.hash WHERE s.input_pk=?1 AND v.text_id IS NOT NULL")?.query_map([pk],|r|r.get::<_,i64>(0))?.collect::<rusqlite::Result<_>>()?;
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

/// The local index key for an exact session ID.
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
