//! `archive.db` rows for stored bytes: blobs, revisions, extents and
//! observations, numbered by the archive's write counter.

use super::*;

/// Stores the bytes' rows and returns the revision chain the batch's
/// records are in, or `None` when there is none (a disappearance).
pub(super) fn write_bytes(tx: &Transaction<'_>, b: &Batch, seq: i64) -> Result<Option<i64>> {
    tx.prepare_cached("INSERT OR IGNORE INTO input(id, format) VALUES (?1, ?2)")?
        .execute(params![b.input.id, b.input.format])?;
    for piece in &b.blobs {
        insert_blob(tx, piece)?;
    }
    if let Change::Reparse { chain } = b.change {
        if b.piece.is_some()
            || !tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM revision WHERE input=?1 AND chain=?2 AND length>=?3)",
                params![b.input.id, chain, b.length as i64],
                |r| r.get::<_, bool>(0),
            )?
        {
            return Err(StoreError::Invalid(
                "reparse must name retained bytes".into(),
            ));
        }
        return Ok(Some(chain));
    }
    let latest: Option<(i64, i64, i64)> = tx
        .prepare_cached(
            "SELECT number, chain, length FROM revision WHERE input = ?1
             ORDER BY number DESC LIMIT 1",
        )?
        .query_row([&b.input.id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .optional()?;
    let next = latest.map_or(1, |l| l.0 + 1);

    let (number, chain, offset) = match (&b.change, b.continues, latest) {
        (Change::Disappeared, _, l) => {
            insert_observation(tx, b, "deleted", None, seq)?;
            return Ok(l.map(|l| l.1));
        }
        (_, true, Some((number, chain, length))) => {
            tx.prepare_cached("UPDATE revision SET length = ?3 WHERE input = ?1 AND number = ?2")?
                .execute(params![b.input.id, number, b.length as i64])?;
            (number, chain, length as u64)
        }
        (Change::Unchanged, _, Some((number, chain, _))) => (number, chain, b.length),
        (Change::Appended { from }, false, Some((_, chain, _))) => {
            insert_revision(tx, b, next, chain, "appended", seq)?;
            (next, chain, *from)
        }
        (change, _, _) => {
            let produced = match (change, b.input.route_rank) {
                (_, 2) => "imported",
                (Change::First, _) => "first",
                _ => "rewritten",
            };
            insert_revision(tx, b, next, next, produced, seq)?;
            (next, next, 0)
        }
    };

    if let Some(piece) = &b.piece {
        insert_blob(tx, piece)?;
        tx.prepare_cached(
            "INSERT OR IGNORE INTO extent(input, chain, offset, length, blob, blob_offset)
             VALUES (?1, ?2, ?3, ?4, ?5, 0)",
        )?
        .execute(params![
            b.input.id,
            chain,
            offset as i64,
            piece.size as i64,
            &piece.hash.0[..]
        ])?;
    }
    if b.observation.is_some() && !b.continues {
        let class = match &b.change {
            Change::First if b.input.route_rank == 2 => "imported",
            Change::First => "first",
            Change::Appended { .. } => "appended",
            Change::Rewritten => "rewritten",
            Change::Unchanged if b.input.route_rank == 2 => "imported",
            Change::Unchanged => "back",
            Change::Reparse { .. } => unreachable!(),
            Change::Disappeared => "deleted",
        };
        insert_observation(tx, b, class, Some(number), seq)?;
    }
    Ok(Some(chain))
}

pub(super) fn insert_revision(
    tx: &Transaction<'_>,
    b: &Batch,
    number: i64,
    chain: i64,
    produced: &str,
    seq: i64,
) -> Result<()> {
    tx.prepare_cached(
        "INSERT INTO revision(input, number, produced, chain, length, commit_seq)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
    )?
    .execute(params![
        b.input.id,
        number,
        produced,
        chain,
        b.length as i64,
        seq
    ])?;
    Ok(())
}

pub(super) fn insert_blob(tx: &Transaction<'_>, piece: &Piece) -> Result<()> {
    let (place, stored_size, data) = match &piece.stored {
        Stored::Inline(z) => ("inline", z.len() as u64, Some(z.as_slice())),
        Stored::File { stored_size } => ("file", *stored_size, None),
    };
    tx.prepare_cached(
        "INSERT OR IGNORE INTO blob(hash, size, stored_size, place, encoding, state, data)
         VALUES (?1, ?2, ?3, ?4, 'plain', 'present', ?5)",
    )?
    .execute(params![
        &piece.hash.0[..],
        piece.size as i64,
        stored_size as i64,
        place,
        data
    ])?;
    Ok(())
}

/// New IDs sort by insertion within one transaction's shared sequence.
/// A UUID distinguishes independent archives with equal local counters.
/// Restored IDs remain opaque; migrations order those facts by their rowid.
pub(crate) fn observation_id(tx: &Connection, seq: i64, input: &str) -> Result<String> {
    let ordinal: i64 = tx
        .prepare_cached("SELECT COALESCE(MAX(rowid),0)+1 FROM observation")?
        .query_row([], |r| r.get(0))?;
    Ok(format!(
        "{seq:016x}-{ordinal:016x}-{}-{input}",
        uuid::Uuid::now_v7()
    ))
}

pub(super) fn check_import_format(
    tx: &Connection,
    input: &str,
    member: Option<&ImportMember>,
) -> Result<()> {
    let Some(format) = member
        .and_then(|member| member.format.as_deref())
        .filter(|format| *format != "export-file")
    else {
        return Ok(());
    };
    let competing: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM input WHERE id=?1 AND format NOT IN ('export-file',?2))
        OR EXISTS(SELECT 1 FROM observation WHERE input=?1 AND class='imported' AND json_valid(details_json)
            AND json_type(details_json,'$.import_format')='text' AND json_extract(details_json,'$.import_format') NOT IN ('export-file',?2))",
        params![input,format], |r|r.get(0))?;
    if competing {
        return Err(StoreError::Conflict("immutable import has a competing parsed origin; switching interpretations is unsupported".into()));
    }
    Ok(())
}

pub(super) fn insert_observation(
    tx: &Transaction<'_>,
    b: &Batch,
    class: &str,
    revision: Option<i64>,
    seq: i64,
) -> Result<()> {
    let Some(o) = &b.observation else {
        return Ok(());
    };
    check_import_format(tx, &b.input.id, o.import.as_ref())?;
    tx.prepare_cached("INSERT OR IGNORE INTO machine(id, host, first_seen) VALUES (?1, ?2, ?3)")?
        .execute(params![o.machine, o.host, o.seen_at_ms])?;
    let f = o.facts;
    let id = observation_id(tx, seq, &b.input.id)?;
    tx.prepare_cached(
        "INSERT INTO observation(id, input, machine, path, class, revision, length, device, inode,
             size, mtime_ns, ctime_ns, polylogue_version, seen_at, commit_seq, details_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
    )?
    .execute(params![
        id,
        b.input.id,
        o.machine,
        o.path,
        class,
        revision,
        revision.map(|_| b.length as i64),
        f.map(|f| f.device as i64),
        f.map(|f| f.inode as i64),
        f.map(|f| f.size as i64),
        f.map(|f| f.mtime_ns as i64),
        f.map(|f| f.ctime_ns as i64),
        env!("CARGO_PKG_VERSION"),
        o.seen_at_ms,
        seq,
        o.import
            .as_ref()
            .map(|member| serde_json::to_string(
                &serde_json::json!({"import": member, "import_format": member.format})
            ))
            .transpose()?,
    ])?;
    if let Some(member) = &o.import {
        crate::containers::record(tx, &id, member)?;
    }
    Ok(())
}
