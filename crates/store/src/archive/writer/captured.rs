//! Archive rows for captured inputs whose bytes are put: inputs,
//! revisions, extents and observations, numbered by the archive's write
//! counter, committed in one transaction per group.

use super::*;
use crate::ConnExt as _;
use crate::blobs::Change;
use crate::input_rows::{Extent, InputRows, ObservationRow, Produced, Revision, Seen};
use crate::{ImportMember, NewBlob};
use polylogue_model::BlobHash;
use rusqlite::{OptionalExtension, Transaction, params};
use std::collections::BTreeSet;

impl Inner {
    /// Commits the rows of `batches`, whose blobs are put, in one
    /// transaction, and answers where each batch's bytes are. A batch that
    /// only parses stored bytes, and an import already recorded the same
    /// way, write nothing.
    pub(super) fn captured(
        &mut self,
        mut batches: Vec<Batch>,
        mut added: BTreeMap<BlobHash, u64>,
        agreements: Vec<Option<super::agreement::Agreement>>,
    ) -> Result<Committed<Vec<StoredInput>>> {
        for batch in &batches {
            check(&self.conn, batch)?;
        }
        if let Some(stored) = repeated_imports(&self.conn, &batches)? {
            return Ok(Committed {
                position: self.position()?,
                outcome: stored,
            });
        }
        if batches.iter().all(|batch| {
            matches!(batch.change, Change::Reparse { .. })
                && batch.new_blob.is_none()
                && batch.observation.is_none()
                && batch.blobs.is_empty()
                && batch.capture_read.is_none()
        }) {
            let stored = batches
                .iter()
                .map(|b| {
                    Ok(StoredInput {
                        input: b.input.id.clone(),
                        chain: stored_reading_chain(&self.conn, b)?,
                        length: b.length,
                        already_stored: true,
                        bytes_written: 0,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(Committed {
                position: self.position()?,
                outcome: stored,
            });
        }
        // A member read from a container needs its container stored, earlier
        // in this group or before.
        let staged: BTreeSet<&str> = batches.iter().map(|b| b.input.id.as_str()).collect();
        for batch in &batches {
            if let Some(zip) = crate::container_members::container_of(batch)
                && !staged.contains(zip.input.as_str())
            {
                crate::container_members::require_original(&self.conn, &zip.input)?;
            }
        }
        let tx = self.begin()?;
        tx.execute_batch("PRAGMA defer_foreign_keys=ON")?;
        let seq = bump_counter(&tx)?;
        let counted = tx.total_changes();
        let mut stored = Vec::new();
        let mut seen = BTreeSet::new();
        for (b, agreement) in batches
            .iter_mut()
            .zip(agreements.into_iter().chain(std::iter::repeat(None)))
        {
            let bytes_written = b
                .new_blob
                .iter()
                .chain(&b.blobs)
                .flat_map(NewBlob::parts)
                .filter_map(|part| added.remove(&part.hash))
                .sum();
            // An immutable import named by its bytes may be stored already, or
            // appear twice in a group (equal container members).
            let known: Option<(i64, i64)> = tx
                .query_row(
                    "SELECT chain, length FROM revision WHERE input = ?1 ORDER BY number DESC LIMIT 1",
                    [&b.input.id],
                    |r| <(_, _)>::try_from(r),
                )
                .optional()?;
            let again = !seen.insert(b.input.id.clone());
            if b.input.route.is_imported() && b.new_blob.is_some() && (known.is_some() || again) {
                let new_blob = b.new_blob.as_ref().expect("checked new_blob");
                // Empty bytes are stored with no extent.
                let same: bool = tx.scalar("SELECT CASE WHEN ?2 <> ?3 THEN 0 WHEN ?3 = 0 THEN count(*) = 0
                            ELSE count(*) = 1 AND min(offset) = 0 AND min(length) = ?3 AND min(blob) = ?4 END
                     FROM extent WHERE input = ?1 AND chain = (SELECT chain FROM revision WHERE input = ?1 ORDER BY number DESC LIMIT 1)", params![b.input.id, known.map_or(b.length as i64, |k| k.1), b.length as i64, &new_blob.hash.0[..]])?;
                if !same {
                    return Err(StoreError::Conflict(
                        "an import named by its bytes holds other bytes".into(),
                    ));
                }
                b.change = Change::Unchanged;
                b.new_blob = None;
            }
            let (chain, _) = write_bytes(&tx, b, seq, agreement)?;
            let chain = chain.or_invalid("stored bytes produced no chain")?;
            crate::fault::check("import.rows", &b.input.id)?;
            let already_stored = again
                || matches!(
                    b.change,
                    Change::Unchanged | Change::Observed { .. } | Change::Shorter { .. }
                );
            stored.push(StoredInput {
                input: b.input.id.clone(),
                chain,
                length: b.length,
                already_stored,
                bytes_written,
            });
        }
        if tx.total_changes() == counted {
            // A step that names the chain its stored bytes are in writes nothing.
            tx.rollback()?;
            return Ok(Committed {
                position: self.position()?,
                outcome: stored,
            });
        }
        crate::fault::check("write.archive_commit", "")?;
        let position = self.commit(tx, seq, crate::completion::Committed::CONTENT)?;
        self.shared.report.locked().batches +=
            batches.iter().filter(|b| b.parse.is_empty()).count() as u64;
        Ok(Committed {
            position,
            outcome: stored,
        })
    }
}

/// What a batch must satisfy before any of its rows are written.
fn check(conn: &Connection, b: &Batch) -> Result<()> {
    if b.input.id.is_empty() {
        return Err(StoreError::Invalid("stored bytes need an input".into()));
    }
    crate::to_i64(b.length, "input exceeds archive address space")?;
    check_import_format(
        conn,
        &b.input.id,
        b.observation.as_ref().and_then(|o| o.import.as_ref()),
    )?;
    if b.new_blob.as_ref().is_some_and(|new_blob| {
        new_blob.parts().map(|part| part.size).sum::<u64>() != new_blob.size
    }) {
        return Err(StoreError::Invalid(
            "the chunks of new bytes do not add up to their length".into(),
        ));
    }
    if b.input.route.is_imported()
        && let Some(new_blob) = &b.new_blob
        && (b.input.id != new_blob.hash.to_string() || new_blob.size != b.length || b.continues)
    {
        return Err(StoreError::Invalid(
            "an import must be named by its complete bytes".into(),
        ));
    }
    Ok(())
}

/// A retry of an unchanged import group has nothing new to write. A new
/// path, receiver or import membership is a new observation even for known
/// bytes.
fn repeated_imports(conn: &Connection, group: &[Batch]) -> Result<Option<Vec<StoredInput>>> {
    let mut stored = Vec::with_capacity(group.len());
    for b in group {
        let (Some(observation), Some(blob)) = (&b.observation, &b.new_blob) else {
            return Ok(None);
        };
        if !b.input.route.is_imported() || !b.parse.is_empty() || !b.blobs.is_empty() {
            return Ok(None);
        }
        let details = observation
            .import
            .as_ref()
            .map(|member| {
                serde_json::to_string(
                    &serde_json::json!({"import":member,"import_format":member.format}),
                )
            })
            .transpose()?;
        let known = conn.query_row(
            "SELECT r.chain,r.length,
               EXISTS(SELECT 1 FROM observation o WHERE o.input=?1 AND o.machine=?4 AND o.path_bytes=?5 AND o.class='imported' AND json_extract(o.details_json,'$.import') IS json_extract(?6,'$.import') AND json_extract(o.details_json,'$.import_format') IS json_extract(?6,'$.import_format'))
             FROM revision r WHERE r.input=?1
               AND r.number=(SELECT max(number) FROM revision WHERE input=?1)
               AND r.length=?2
               AND (SELECT count(*) FROM extent WHERE input=?1 AND chain=r.chain)=1
               AND EXISTS(SELECT 1 FROM extent x JOIN blob b ON b.hash=x.blob
                   WHERE x.input=?1 AND x.chain=r.chain AND x.offset=0
                     AND x.length=?2 AND x.blob_offset=0 AND x.blob=?3 AND b.size=?2)",
            params![b.input.id,b.length as i64,&blob.hash.0[..],observation.machine,observation.path_bytes,details],
            |row| <(i64, i64, bool)>::try_from(row),
        ).optional()?;
        let Some((chain, length, true)) = known else {
            return Ok(None);
        };
        stored.push(StoredInput {
            input: b.input.id.clone(),
            chain,
            length: length as u64,
            already_stored: true,
            bytes_written: 0,
        });
    }
    Ok(Some(stored))
}

/// Stores the bytes' rows and returns the revision chain the batch's
/// records are in (`None` when there is none: a disappearance before any
/// revision), with the archive rows it wrote, if any.
pub(crate) fn write_bytes(
    tx: &Transaction<'_>,
    b: &Batch,
    seq: i64,
    agreement: Option<super::agreement::Agreement>,
) -> Result<(Option<i64>, Option<InputRows>)> {
    tx.prepare_cached("INSERT OR IGNORE INTO input(id, origin) VALUES (?1, ?2)")?
        .execute(params![b.input.id, b.input.format])?;
    for part in b.blobs.iter().flat_map(NewBlob::parts) {
        check_blob_size(tx, part)?;
    }
    if matches!(b.change, Change::Reparse { .. }) {
        return Ok((Some(stored_reading_chain(tx, b)?), None));
    }
    let (chain, mut rows) = input_rows(tx, b, seq)?;
    if rows
        .revision
        .as_ref()
        .is_some_and(|r| r.chain == r.number && r.produced.starts_chain())
    {
        rows.chain = Some(agreement.unwrap_or(super::agreement::Agreement::NONE));
    }
    for part in b.new_blob.iter().flat_map(NewBlob::parts) {
        check_blob_size(tx, part)?;
    }
    rows.insert(tx, seq)?;
    Ok((chain, Some(rows)))
}

// A chain only grows. Seek the latest revision first so replay of
// a long append history does not revisit that history for every parsed chunk.
// Older chains keep their original existence check, including sparse inputs.
pub(crate) fn stored_reading_chain(conn: &Connection, batch: &Batch) -> Result<i64> {
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
    let latest = latest
        .map(|(number, chain, length, produced)| {
            Produced::from_name(&produced)
                .map(|produced| (number, chain, length, produced))
                .ok_or_else(|| StoreError::stored(3, format!("revision produced {produced:?}")))
        })
        .transpose()?;
    input_rows_at(tx, b, seq, latest, &std::collections::BTreeSet::new())
}

fn input_rows_at(
    tx: &Connection,
    b: &Batch,
    seq: i64,
    latest: Option<(i64, i64, i64, Produced)>,
    staged: &std::collections::BTreeSet<String>,
) -> Result<(Option<i64>, InputRows)> {
    let next = latest.as_ref().map_or(1, |l| l.0 + 1);
    let mut rows = InputRows {
        capture_read: b.capture_read.clone(),
        input: b.input.id.clone(),
        origin: b.input.format.clone(),
        revision: None,
        chain: None,
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
        let length = b.length as i64;
        let exact: bool = tx.scalar(&source.sql("SELECT EXISTS(SELECT 1 FROM @source.revision WHERE input=?1 AND number=?2 AND chain=?3 AND length=?4)"),params![b.input.id,number,chain,length])?;
        if !exact {
            return Err(StoreError::Invalid(
                "observed revision point is not stored".into(),
            ));
        }
        let seen = Seen::Back {
            revision: number,
            length,
        };
        rows.observation = observation(tx, b, seen, seq, staged)?;
        return Ok((Some(chain), rows));
    }
    if let Change::Shorter { chain } = b.change {
        // The capture proved every byte the file holds equal to the stored
        // byte at its place; the chain must hold more than that.
        if b.continues
            || b.new_blob.is_some()
            || !b.blobs.is_empty()
            || !b.parse.is_empty()
            || b.observation.is_none()
        {
            return Err(StoreError::Invalid(
                "a file seen holding a prefix of its chain stores no bytes or rows".into(),
            ));
        }
        let source = crate::source::Source::of(tx)?;
        let length = b.length as i64;
        let longer: bool = tx.scalar(&source.sql("SELECT EXISTS(SELECT 1 FROM @source.revision WHERE input=?1 AND chain=?2 AND length>?3)"),params![b.input.id,chain,length])?;
        if !longer {
            return Err(StoreError::Invalid(
                "a file seen holding a prefix names a chain no longer than it".into(),
            ));
        }
        rows.observation = observation(tx, b, Seen::Prefix { chain, length }, seq, staged)?;
        return Ok((Some(chain), rows));
    }
    let revision = |number, produced, chain| Revision {
        number,
        produced,
        chain,
        length: b.length as i64,
        seen_at: b.observation.as_ref().map(|o| o.seen_at_ms),
        mtime_ns: b
            .observation
            .as_ref()
            .and_then(|o| o.facts)
            .map(|f| f.mtime_ns as i64),
        ctime_ns: b
            .observation
            .as_ref()
            .and_then(|o| o.facts)
            .map(|f| f.ctime_ns as i64),
    };
    // Where the step's new bytes go: the end of its chain as stored, so they
    // never land on a stored extent. `None` when they are stored already.
    let (number, chain, offset) = match (&b.change, b.continues, latest) {
        (Change::Disappeared, _, latest) => {
            rows.observation = observation(tx, b, Seen::Deleted, seq, staged)?;
            return Ok((latest.map(|l| l.1), rows));
        }
        (_, true, Some((number, chain, length, produced))) => {
            never_shorter(b, length)?;
            rows.revision = Some(revision(number, produced, chain));
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
            rows.revision = Some(revision(next, Produced::Appended, chain));
            (next, chain, offset)
        }
        (change, _, _) => {
            let produced = match (change, b.input.route.is_imported()) {
                (_, true) => Produced::Imported,
                (Change::First, _) => Produced::First,
                _ => Produced::Rewritten,
            };
            rows.revision = Some(revision(next, produced, next));
            (next, next, Some(0))
        }
    };
    if let (Some(new_blob), Some(offset)) = (&b.new_blob, offset) {
        rows.extents = extents(new_blob, chain, offset);
    }
    if b.observation.is_some() && !b.continues {
        let length = b.length as i64;
        let stored = |produced| Seen::Stored {
            produced,
            revision: number,
            length,
        };
        let seen = match &b.change {
            Change::First | Change::Unchanged if b.input.route.is_imported() => {
                stored(Produced::Imported)
            }
            Change::First => stored(Produced::First),
            Change::Appended { .. } => stored(Produced::Appended),
            Change::Rewritten => stored(Produced::Rewritten),
            Change::Unchanged => Seen::Back {
                revision: number,
                length,
            },
            Change::Reparse { .. }
            | Change::Observed { .. }
            | Change::Shorter { .. }
            | Change::Disappeared => {
                unreachable!()
            }
        };
        // An append to the file the input was last observed as, at the same
        // location, is the revision alone: an observation records a change
        // of location or of file (a return after it was gone, another
        // inode). A database read's observation carries its log's facts, so
        // it is always written.
        let unmoved = matches!(b.change, Change::Appended { .. })
            && b.observation.as_ref().is_some_and(|o| {
                o.database.is_none()
                    && last_location(tx, &b.input.id)
                        .ok()
                        .flatten()
                        .is_some_and(|last| {
                            last.machine == o.machine
                                && last.path_bytes == o.path_bytes
                                && last.class
                                    != crate::input_rows::ObservationClass::Deleted.as_str()
                                && o.facts.is_some_and(|f| {
                                    last.device == Some(f.device as i64)
                                        && last.inode == Some(f.inode as i64)
                                })
                        })
            });
        if !unmoved {
            rows.observation = observation(tx, b, seen, seq, staged)?;
        }
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
pub(crate) fn extents(new_blob: &NewBlob, chain: i64, offset: u64) -> Vec<Extent> {
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

/// New IDs sort by insertion: the write counter, then the observation's
/// place among those of its commit, both in fixed-width hex. A UUID
/// distinguishes independent archives with equal local counters.
pub(crate) fn observation_id(tx: &Connection, seq: i64, input: &str) -> Result<String> {
    let ordinal: i64 = tx
        .prepare_cached(
            &crate::source::Source::of(tx)?
                .sql("SELECT count(*)+1 FROM @source.observation WHERE commit_seq = ?1"),
        )?
        .query_row([seq], |r| r.get(0))?;
    Ok(format!(
        "{seq:016x}-{ordinal:016x}-{}-{input}",
        uuid::Uuid::now_v7()
    ))
}

pub(crate) fn check_import_format(
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
    let competing: bool = tx.scalar(&crate::source::Source::of(tx)?.sql("SELECT EXISTS(SELECT 1 FROM @source.input WHERE id=?1 AND origin NOT IN ('export-file',?2))
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
/// Where and as what file the input was last observed.
struct LastLocation {
    machine: String,
    path_bytes: Vec<u8>,
    class: String,
    device: Option<i64>,
    inode: Option<i64>,
}

fn last_location(tx: &Connection, input: &str) -> Result<Option<LastLocation>> {
    let source = crate::source::Source::of(tx)?;
    Ok(tx
        .prepare_cached(&source.sql(
            "SELECT machine, path_bytes, class, device, inode FROM @source.observation
             WHERE input = ?1 ORDER BY commit_seq DESC, id DESC LIMIT 1",
        ))?
        .query_row([input], |r| {
            Ok(LastLocation {
                machine: r.get(0)?,
                path_bytes: r.get(1)?,
                class: r.get(2)?,
                device: r.get(3)?,
                inode: r.get(4)?,
            })
        })
        .optional()?)
}

pub(crate) fn observation(
    tx: &Connection,
    b: &Batch,
    seen: Seen,
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
        path_bytes: o.path_bytes.clone(),
        seen,
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
