//! Register rows for stored bytes: blobs, revisions, extents and
//! observations, numbered by the archive's write counter.

use super::*;
use crate::input_record::{Extent, InputRecord, ObservationRecord, Revision};

/// Stores the bytes' rows and returns the revision chain the batch's
/// records are in (`None` when there is none: a disappearance before any
/// revision), with the register record it wrote, if any.
pub(super) fn write_bytes(
    tx: &Transaction<'_>,
    b: &Batch,
    seq: i64,
) -> Result<(Option<i64>, Option<InputRecord>)> {
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
                "a reparse must name stored bytes".into(),
            ));
        }
        return Ok((Some(chain), None));
    }
    let (chain, record) = input_record(tx, b, seq)?;
    if let Some(piece) = &b.piece {
        insert_blob(tx, piece)?;
    }
    record.insert(tx, seq)?;
    Ok((chain, Some(record)))
}

/// What one step of an input adds to the register, from the input's latest
/// revision and the step's change.
/// Returns it with the revision chain the step's records are in.
fn input_record(
    tx: &Transaction<'_>,
    b: &Batch,
    seq: i64,
) -> Result<(Option<i64>, InputRecord)> {
    let latest: Option<(i64, i64, i64, String)> = tx
        .prepare_cached(
            "SELECT number, chain, length, produced FROM revision WHERE input = ?1
             ORDER BY number DESC LIMIT 1",
        )?
        .query_row([&b.input.id], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
        })
        .optional()?;
    let next = latest.as_ref().map_or(1, |l| l.0 + 1);
    let mut record = InputRecord {
        input: b.input.id.clone(),
        format: b.input.format.clone(),
        revision: None,
        extent: None,
        observation: None,
    };
    let revision = |number, produced: &str, chain| Revision {
        number,
        produced: produced.to_owned(),
        chain,
        length: b.length as i64,
    };
    let (number, chain, offset) = match (&b.change, b.continues, latest) {
        (Change::Disappeared, _, latest) => {
            record.observation = observation(tx, b, "deleted", None, seq)?;
            return Ok((latest.map(|l| l.1), record));
        }
        (_, true, Some((number, chain, length, produced))) => {
            record.revision = Some(revision(number, &produced, chain));
            (number, chain, length as u64)
        }
        (Change::Unchanged, _, Some((number, chain, _, _))) => (number, chain, b.length),
        (Change::Appended { from }, false, Some((_, chain, _, _))) => {
            record.revision = Some(revision(next, "appended", chain));
            (next, chain, *from)
        }
        (change, _, _) => {
            let produced = match (change, b.input.route_rank) {
                (_, 2) => "imported",
                (Change::First, _) => "first",
                _ => "rewritten",
            };
            record.revision = Some(revision(next, produced, next));
            (next, next, 0)
        }
    };
    if let Some(piece) = &b.piece {
        record.extent = Some(Extent {
            chain,
            offset: offset as i64,
            length: piece.size as i64,
            blob: piece.hash,
        });
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
        record.observation = observation(tx, b, class, Some(number), seq)?;
    }
    Ok((Some(chain), record))
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
        return Err(StoreError::Conflict(
            "this import was already parsed as another origin; changing its format is unsupported"
                .into(),
        ));
    }
    Ok(())
}

/// The observation a step records, checked against what the archive
/// already holds for the input.
pub(super) fn observation(
    tx: &Transaction<'_>,
    b: &Batch,
    class: &str,
    revision: Option<i64>,
    seq: i64,
) -> Result<Option<ObservationRecord>> {
    let Some(o) = &b.observation else {
        return Ok(None);
    };
    check_import_format(tx, &b.input.id, o.import.as_ref())?;
    let container = match o.import.as_ref().and_then(|member| member.container.as_ref()) {
        Some(container) => Some(crate::containers::check(tx, &b.input.id, container)?),
        None => None,
    };
    let f = o.facts;
    Ok(Some(ObservationRecord {
        id: observation_id(tx, seq, &b.input.id)?,
        machine: o.machine.clone(),
        host: o.host.clone(),
        path: o.path.clone(),
        class: class.to_owned(),
        revision,
        length: revision.map(|_| b.length as i64),
        device: f.map(|f| f.device as i64),
        inode: f.map(|f| f.inode as i64),
        size: f.map(|f| f.size as i64),
        mtime_ns: f.map(|f| f.mtime_ns as i64),
        ctime_ns: f.map(|f| f.ctime_ns as i64),
        polylogue_version: env!("CARGO_PKG_VERSION").to_owned(),
        seen_at: o.seen_at_ms,
        details_json: o
            .import
            .as_ref()
            .map(|member| {
                serde_json::to_string(
                    &serde_json::json!({"import": member, "import_format": member.format}),
                )
            })
            .transpose()?,
        excision: None,
        container,
    }))
}
