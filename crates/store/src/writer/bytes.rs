//! Archive rows for stored bytes: blobs, revisions, extents and
//! observations, numbered by the archive's write counter.

use super::*;
use crate::input_rows::{Extent, InputRows, ObservationRow, Revision};

/// Stores the bytes' rows and returns the revision chain the batch's
/// records are in (`None` when there is none: a disappearance before any
/// revision), with the archive rows it wrote, if any.
pub(super) fn write_bytes(
    tx: &Transaction<'_>,
    b: &Batch,
    seq: i64,
) -> Result<(Option<i64>, Option<InputRows>)> {
    tx.prepare_cached("INSERT OR IGNORE INTO input(id, format) VALUES (?1, ?2)")?
        .execute(params![b.input.id, b.input.format])?;
    for part in b.blobs.iter().flat_map(NewBlob::parts) {
        check_blob_size(tx, part)?;
    }
    if matches!(b.change, Change::Reparse { .. }) {
        return Ok((Some(stored_reading_chain(tx, b)?), None));
    }
    let (chain, rows) = input_rows(tx, b, seq)?;
    for part in b.new_blob.iter().flat_map(NewBlob::parts) {
        check_blob_size(tx, part)?;
    }
    rows.insert(tx, seq)?;
    Ok((chain, Some(rows)))
}

// A chain only grows. Seek the latest revision first so replay of
// a long append history does not revisit that history for every parsed chunk.
// Older chains keep their original existence check, including sparse inputs.
pub(super) fn stored_reading_chain(conn: &Connection, batch: &Batch) -> Result<i64> {
    let Change::Reparse { chain } = batch.change else {
        return Err(StoreError::Invalid(
            "a stored reading must name its chain".into(),
        ));
    };
    let length = crate::to_i64(batch.length, "input exceeds archive address space")?;
    let latest: Option<(i64, i64)> = conn
        .prepare_cached(
            "SELECT chain,length FROM revision WHERE input=?1 ORDER BY number DESC LIMIT 1",
        )?
        .query_row([&batch.input.id], |row| <(_, _)>::try_from(row))
        .optional()?;
    let stored = match latest {
        Some((latest_chain, stored_length)) if latest_chain == chain => stored_length >= length,
        Some(_) => conn
            .prepare_cached(
                "SELECT EXISTS(SELECT 1 FROM revision WHERE input=?1 AND chain=?2 AND length>=?3)",
            )?
            .query_row(params![batch.input.id, chain, length], |row| {
                row.get::<_, bool>(0)
            })?,
        None => false,
    };
    if batch.new_blob.is_some() || !stored {
        return Err(StoreError::Invalid(
            "a reparse must name stored bytes".into(),
        ));
    }
    Ok(chain)
}

/// What one step of an input adds to the archive, from the input's latest
/// revision and the step's change.
/// Returns it with the revision chain the step's records are in.
fn input_rows(tx: &Connection, b: &Batch, seq: i64) -> Result<(Option<i64>, InputRows)> {
    let source = crate::source::Source::of(tx)?;
    let latest: Option<(i64, i64, i64, String)> = tx
        .prepare_cached(&source.sql(
            "SELECT number, chain, length, produced FROM @source.revision WHERE input = ?1
             ORDER BY number DESC LIMIT 1",
        ))?
        .query_row([&b.input.id], |r| <(_, _, _, _)>::try_from(r))
        .optional()?;
    input_rows_at(tx, b, seq, latest, &std::collections::BTreeSet::new())
}

fn input_rows_at(
    tx: &Connection,
    b: &Batch,
    seq: i64,
    latest: Option<(i64, i64, i64, String)>,
    staged: &std::collections::BTreeSet<String>,
) -> Result<(Option<i64>, InputRows)> {
    let next = latest.as_ref().map_or(1, |l| l.0 + 1);
    let mut rows = InputRows {
        capture_read: b.capture_read.clone(),
        input: b.input.id.clone(),
        origin: b.input.format.clone(),
        revision: None,
        extents: Vec::new(),
        observation: None,
    };
    if let Change::Observed {
        revision: number,
        chain,
    } = b.change
    {
        if b.continues
            || b.new_blob.is_some()
            || b.blobs.iter().any(|blob| {
                !b.capture_read.as_ref().is_some_and(|capture_read| {
                    capture_read.hash == blob.hash
                        && capture_read.range.length().ok() == Some(blob.size)
                })
            })
            || !b.parse.is_empty()
            || b.observation.is_none()
        {
            return Err(StoreError::Invalid(
                "a recorded revision observation cannot create source extents or parsed rows"
                    .into(),
            ));
        }
        let source = crate::source::Source::of(tx)?;
        let stored:bool=tx.scalar(&source.sql("SELECT EXISTS(SELECT 1 FROM @source.revision WHERE input=?1 AND number=?2 AND chain=?3 AND length=?4)"),params![b.input.id,number,chain,b.length as i64])?;
        if !stored {
            return Err(StoreError::Invalid(
                "observed revision point is not stored".into(),
            ));
        }
        rows.observation = observation(tx, b, "back", Some(number), seq, staged)?;
        return Ok((Some(chain), rows));
    }
    let revision = |number, produced: &str, chain| Revision {
        number,
        produced: produced.to_owned(),
        chain,
        length: b.length as i64,
    };
    // Where the step's new bytes go: the end of its chain as stored, so they
    // never land on a stored extent. `None` when they are stored already.
    let (number, chain, offset) = match (&b.change, b.continues, latest) {
        (Change::Disappeared, _, latest) => {
            rows.observation = observation(tx, b, "deleted", None, seq, staged)?;
            return Ok((latest.map(|l| l.1), rows));
        }
        (_, true, Some((number, chain, length, produced))) => {
            never_shorter(b, length)?;
            rows.revision = Some(revision(number, &produced, chain));
            (number, chain, Some(length as u64))
        }
        (Change::Unchanged, _, Some((number, chain, _, _))) => {
            if b.new_blob.is_some() {
                return Err(StoreError::Invalid(
                    "an unchanged step adds no bytes".into(),
                ));
            }
            (number, chain, None)
        }
        (Change::Appended { from }, false, Some((_, chain, length, _))) => {
            never_shorter(b, length)?;
            let mut offset = Some(length as u64);
            if let Some(blob) = &b.new_blob
                && *from != length as u64
            {
                // A repeated committed append may name the extent it stored,
                // and then adds nothing; stale source state must not grow a
                // prefix over later bytes.
                let retry = b.length == length as u64
                    && tx.scalar::<bool>(
                        &crate::source::Source::of(tx)?.sql(
                            "SELECT EXISTS(SELECT 1 FROM @source.extent WHERE input=?1 AND chain=?2
                         AND offset=?3 AND length=?4 AND blob=?5 AND blob_offset=0)",
                        ),
                        params![
                            b.input.id,
                            chain,
                            *from as i64,
                            blob.size as i64,
                            &blob.hash.0[..]
                        ],
                    )?;
                if !retry {
                    return Err(StoreError::Invalid(
                        "an append must start at the stored length".into(),
                    ));
                }
                offset = None;
            }
            rows.revision = Some(revision(next, "appended", chain));
            (next, chain, offset)
        }
        (change, _, _) => {
            let produced = match (change, b.input.route.is_imported()) {
                (_, true) => "imported",
                (Change::First, _) => "first",
                _ => "rewritten",
            };
            rows.revision = Some(revision(next, produced, next));
            (next, next, Some(0))
        }
    };
    if let (Some(new_blob), Some(offset)) = (&b.new_blob, offset) {
        rows.extents = extents(new_blob, chain, offset);
    }
    if b.observation.is_some() && !b.continues {
        let class = match &b.change {
            Change::First if b.input.route.is_imported() => "imported",
            Change::First => "first",
            Change::Appended { .. } => "appended",
            Change::Rewritten => "rewritten",
            Change::Unchanged if b.input.route.is_imported() => "imported",
            Change::Unchanged => "back",
            Change::Reparse { .. } | Change::Observed { .. } => unreachable!(),
            Change::Disappeared => "deleted",
        };
        rows.observation = observation(tx, b, class, Some(number), seq, staged)?;
    }
    Ok((Some(chain), rows))
}

/// A chain only grows: a step that continues or appends to it reaches at
/// least the chain's stored length, so every earlier revision keeps the bytes
/// it was stored with.
fn never_shorter(b: &Batch, stored: i64) -> Result<()> {
    if (b.length as i64) < stored {
        return Err(StoreError::Invalid(
            "a step cannot make its revision chain shorter".into(),
        ));
    }
    Ok(())
}

/// Where new bytes sit in their chain from `offset` on: one extent per blob
/// that holds them. The blob is committed even for an empty capture read;
/// extents describe covered source bytes, so empty
/// bytes have none.
pub(super) fn extents(new_blob: &NewBlob, chain: i64, offset: u64) -> Vec<Extent> {
    let mut offset = offset as i64;
    new_blob
        .parts()
        .filter(|part| part.size > 0)
        .map(|part| {
            let extent = Extent {
                chain,
                offset,
                length: part.size as i64,
                blob: part.hash,
                blob_offset: 0,
            };
            offset += part.size as i64;
            extent
        })
        .collect()
}

/// Refuses a chunk whose blob row already exists with another size. A new
/// blob's row is written by the segment log, not here.
fn check_blob_size(tx: &Transaction<'_>, new_blob: crate::Chunk) -> Result<()> {
    let size: Option<i64> = tx.scalar_optional(
        "SELECT size FROM blob WHERE hash=?1",
        [&new_blob.hash.0[..]],
    )?;
    if size.is_some_and(|size| size != new_blob.size as i64) {
        return Err(StoreError::BlobCorrupt(new_blob.hash));
    }
    Ok(())
}

/// New IDs sort by insertion within one transaction's shared sequence.
/// A UUID distinguishes independent archives with equal local counters.
pub(crate) fn observation_id(tx: &Connection, seq: i64, input: &str) -> Result<String> {
    let ordinal: i64 = tx
        .prepare_cached(
            &crate::source::Source::of(tx)?
                .sql("SELECT COALESCE(MAX(rowid),0)+1 FROM @source.observation"),
        )?
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
    let competing: bool = tx.scalar(&crate::source::Source::of(tx)?.sql("SELECT EXISTS(SELECT 1 FROM @source.input WHERE id=?1 AND format NOT IN ('export-file',?2))
        OR EXISTS(SELECT 1 FROM @source.observation WHERE input=?1 AND class='imported' AND json_valid(details_json)
            AND json_type(details_json,'$.import_format')='text' AND json_extract(details_json,'$.import_format') NOT IN ('export-file',?2))"),
        params![input,format])?;
    if competing {
        return Err(StoreError::Conflict(
            "this import was already parsed as another origin; changing its format is unsupported"
                .into(),
        ));
    }
    Ok(())
}

/// The observation a step records, checked against what the archive
/// already holds for the input. `staged` holds the inputs staged earlier in
/// this capture group, which are not in the archive yet.
pub(super) fn observation(
    tx: &Connection,
    b: &Batch,
    class: &str,
    revision: Option<i64>,
    seq: i64,
    staged: &std::collections::BTreeSet<String>,
) -> Result<Option<ObservationRow>> {
    let Some(o) = &b.observation else {
        return Ok(None);
    };
    check_import_format(tx, &b.input.id, o.import.as_ref())?;
    let container = match o.container.as_ref() {
        Some(container) if staged.contains(&container.input) => Some(
            crate::container_members::check_shape(&b.input.id, container)?,
        ),
        Some(container) => Some(crate::container_members::check(tx, &b.input.id, container)?),
        None => None,
    };
    let f = o.facts;
    Ok(Some(ObservationRow {
        id: observation_id(tx, seq, &b.input.id)?,
        machine: o.machine.clone(),
        host: o.host.clone(),
        path: o.path.clone(),
        path_bytes: Some(o.path_bytes.clone()),
        class: class.to_owned(),
        revision,
        chain: None,
        length: revision.map(|_| b.length as i64),
        device: f.map(|f| f.device as i64),
        inode: f.map(|f| f.inode as i64),
        size: f.map(|f| f.size as i64),
        mtime_ns: f.map(|f| f.mtime_ns as i64),
        ctime_ns: f.map(|f| f.ctime_ns as i64),
        polylogue_version: env!("CARGO_PKG_VERSION").to_owned(),
        seen_at: o.seen_at_ms,
        details_json: match (&o.import, &o.database) {
            (Some(member), _) => Some(serde_json::to_string(
                &serde_json::json!({"import": member, "import_format": member.format}),
            )?),
            (None, Some(log)) => Some(serde_json::to_string(
                &serde_json::json!({"database_wal": log.wal.map(crate::tool_databases::facts_json)}),
            )?),
            (None, None) => None,
        },
        container,
    }))
}
