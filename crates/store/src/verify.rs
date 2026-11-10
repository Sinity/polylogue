//! The on-demand check of stored bytes (`polylogue verify`). It reads every
//! committed blob and checks its encoded bytes against the committed digest
//! and its decoded bytes against its name, checks every extent against the
//! blob it cites and every revision against its chain's extents, compares
//! the kept counters with the rows they count, and runs SQLite's own check
//! on each database. It changes nothing: what it finds is reported, and
//! recovery is a restore from backup.
//!
//! With a manifest it also reads every revision's bytes through its chain
//! and writes one line per revision, `input<TAB>number<TAB>length<TAB>blake3`,
//! ordered by input and number, so that a manifest computed independently
//! from another copy of the same inputs can be compared byte for byte.

use crate::{ReadPool, Result, StoreError};
use rusqlite::params;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;

/// Blobs read per catalog query; the reader is released between batches so
/// a long check pins no snapshot.
const BLOB_BATCH: i64 = 256;
/// Inputs listed per query when writing a manifest.
const INPUT_BATCH: i64 = 256;

/// What one verification found.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Verification {
    /// Row counts in one consistent read, taken before the bytes are read.
    pub counts: Counts,
    /// Inputs, revisions and revision bytes per input format.
    pub formats: Vec<FormatCount>,
    /// Observations per class (`first`, `appended`, `rewritten`, `deleted`, …).
    pub observation_classes: Vec<ClassCount>,
    /// Revisions per way they were produced (`first`, `appended`, `imported`, …).
    pub revision_kinds: Vec<ClassCount>,
    pub blobs: BlobCheck,
    /// SQLite's `quick_check` of each database the archive holds.
    pub databases: Vec<DatabaseCheck>,
    /// Present when a manifest was written.
    pub manifest: Option<ManifestSummary>,
    /// Everything that does not match, in the order found.
    pub problems: Vec<Problem>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Counts {
    pub inputs: u64,
    pub revisions: u64,
    pub extents: u64,
    pub observations: u64,
    pub blobs: u64,
    /// Decoded bytes of every blob in the catalog.
    pub blob_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct FormatCount {
    pub format: String,
    pub inputs: u64,
    pub revisions: u64,
    /// The sum of the revisions' lengths.
    pub revision_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ClassCount {
    pub class: String,
    pub count: u64,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct BlobCheck {
    /// Blobs whose bytes were read and checked.
    pub checked: u64,
    /// Their decoded bytes.
    pub bytes: u64,
    /// Their bytes as stored in the segment log.
    pub stored_bytes: u64,
    /// Blobs with a problem.
    pub mismatched: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct DatabaseCheck {
    /// The schema name on the reader: `main` is the index, `a` the archive.
    pub schema: String,
    pub file: String,
    /// `["ok"]`, or SQLite's messages.
    pub result: Vec<String>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ManifestSummary {
    pub revisions: u64,
    /// Bytes read to hash them: each chain is read once, to its longest revision.
    pub bytes_read: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Problem {
    pub kind: ProblemKind,
    /// What it concerns: a blob hash, `input/chain@offset` or `input#number`,
    /// a counter or a database.
    pub subject: String,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProblemKind {
    /// Stored bytes differ from the committed digest or the blob's name, or
    /// cannot be read.
    BlobBytes,
    /// The catalog's sizes differ from the blob's stored descriptor.
    BlobCatalog,
    /// An extent cites a blob the catalog does not hold.
    ExtentWithoutBlob,
    /// An extent's range lies outside its blob.
    ExtentOutsideBlob,
    /// A chain's extents leave a gap or overlap.
    ExtentGap,
    /// A revision is longer than its chain's extents cover.
    RevisionUncovered,
    /// A revision's bytes could not be read while writing the manifest.
    RevisionUnreadable,
    /// A kept counter differs from the rows it counts.
    RowCount,
    /// SQLite's `quick_check` reported something.
    Database,
}

impl Verification {
    fn problem(
        &mut self,
        kind: ProblemKind,
        subject: impl Into<String>,
        detail: impl Into<String>,
    ) {
        self.problems.push(Problem {
            kind,
            subject: subject.into(),
            detail: detail.into(),
        });
    }
}

/// Waits for foreground work and stops on cancellation. Its argument ends
/// any chain read transaction the check holds; it runs only when the wait
/// will block, so reads that need not wait keep their transaction.
pub type Pace<'a> = dyn FnMut(&dyn Fn()) -> Result<()> + 'a;

/// Runs the whole check. `pace` is called before each blob, between the
/// frames of a large blob, and before each chain and each read of a chain
/// while writing a manifest, holding no reader: the daemon
/// waits there for foreground work and stops on cancellation. A manifest,
/// when given, receives one line per revision.
pub fn verify(
    pool: &ReadPool,
    blobs: &Arc<crate::blobs::Blobs>,
    manifest: Option<&mut dyn Write>,
    pace: &mut Pace<'_>,
) -> Result<Verification> {
    let mut found = Verification::default();
    pace(&|| ())?;
    rows(pool, &mut found)?;
    pace(&|| ())?;
    databases(pool, &mut found)?;
    blob_bytes(pool, blobs.root(), &mut found, pace)?;
    if let Some(out) = manifest {
        found.manifest = Some(write_manifest(pool, blobs, out, &mut found, pace)?);
    }
    Ok(found)
}

fn count(n: i64) -> u64 {
    u64::try_from(n).unwrap_or(0)
}

/// Counts, per-class counts, counters and extent structure, in one read.
fn rows(pool: &ReadPool, found: &mut Verification) -> Result<()> {
    let reader = pool.reader()?;
    let conn = reader.conn()?;
    let one = |sql: &str| -> Result<i64> { Ok(conn.query_row(sql, [], |row| row.get(0))?) };
    found.counts = Counts {
        inputs: count(one("SELECT count(*) FROM source_input")?),
        revisions: count(one("SELECT count(*) FROM source_revision")?),
        extents: count(one("SELECT count(*) FROM source_extent")?),
        observations: count(one("SELECT count(*) FROM source_observation")?),
        blobs: count(one("SELECT count(*) FROM source_blob")?),
        blob_bytes: count(one("SELECT coalesce(sum(size), 0) FROM source_blob")?),
    };
    let mut statement = conn.prepare(
        "SELECT i.format, count(DISTINCT i.id), count(r.number), coalesce(sum(r.length), 0)
         FROM source_input i LEFT JOIN source_revision r ON r.input = i.id
         GROUP BY i.format ORDER BY i.format",
    )?;
    found.formats = statement
        .query_map([], |row| {
            Ok(FormatCount {
                format: row.get(0)?,
                inputs: count(row.get(1)?),
                revisions: count(row.get(2)?),
                revision_bytes: count(row.get(3)?),
            })
        })?
        .collect::<rusqlite::Result<_>>()?;
    let classes = |sql: &str| -> Result<Vec<ClassCount>> {
        let mut statement = conn.prepare(sql)?;
        Ok(statement
            .query_map([], |row| {
                Ok(ClassCount {
                    class: row.get(0)?,
                    count: count(row.get(1)?),
                })
            })?
            .collect::<rusqlite::Result<_>>()?)
    };
    found.observation_classes =
        classes("SELECT class, count(*) FROM source_observation GROUP BY class ORDER BY class")?;
    found.revision_kinds = classes(
        "SELECT produced, count(*) FROM source_revision GROUP BY produced ORDER BY produced",
    )?;

    // The counters `status` reads instead of counting.
    let kept: Vec<(String, i64)> = conn
        .prepare("SELECT name, n FROM source_row_count ORDER BY name")?
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    for (name, n) in kept {
        let counted = match name.as_str() {
            "input" => found.counts.inputs,
            "revision" => found.counts.revisions,
            "stored_bytes" => found.counts.blob_bytes,
            _ => continue,
        };
        if count(n) != counted || n < 0 {
            found.problem(
                ProblemKind::RowCount,
                name,
                format!("kept {n}, counted {counted}"),
            );
        }
    }

    let mut listed = |kind: ProblemKind, sql: &str| -> Result<()> {
        let mut statement = conn.prepare(sql)?;
        let rows: Vec<(String, String)> = statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        for (subject, detail) in rows {
            found.problem(kind, subject, detail);
        }
        Ok(())
    };
    listed(
        ProblemKind::ExtentWithoutBlob,
        "SELECT e.input || '/' || e.chain || '@' || e.offset, 'blob ' || lower(hex(e.blob))
         FROM source_extent e
         WHERE NOT EXISTS (SELECT 1 FROM source_blob b WHERE b.hash = e.blob)
         ORDER BY e.input, e.chain, e.offset",
    )?;
    listed(
        ProblemKind::ExtentOutsideBlob,
        "SELECT e.input || '/' || e.chain || '@' || e.offset,
                'bytes ' || e.blob_offset || '..' || (e.blob_offset + e.length)
                || ' of a blob of ' || b.size
         FROM source_extent e JOIN source_blob b ON b.hash = e.blob
         WHERE e.length <= 0 OR e.offset < 0 OR e.blob_offset < 0
            OR e.blob_offset + e.length > b.size
         ORDER BY e.input, e.chain, e.offset",
    )?;
    listed(
        ProblemKind::ExtentGap,
        "SELECT g.input || '/' || g.chain || '@' || g.start,
                CASE WHEN g.start > g.previous_end THEN 'gap after ' ELSE 'overlap with bytes before ' END
                || g.previous_end
         FROM (SELECT x.input AS input, x.chain AS chain, x.offset AS start,
                      lag(x.offset + x.length, 1, 0)
                        OVER (PARTITION BY x.input, x.chain ORDER BY x.offset) AS previous_end
               FROM source_extent x) g
         WHERE g.start != g.previous_end
         ORDER BY g.input, g.chain, g.start",
    )?;
    listed(
        ProblemKind::RevisionUncovered,
        "SELECT r.input || '#' || r.number,
                'length ' || r.length || ', extents of chain ' || r.chain || ' cover '
                || coalesce((SELECT max(e.offset + e.length) FROM source_extent e
                             WHERE e.input = r.input AND e.chain = r.chain), 0)
         FROM source_revision r
         WHERE r.length > coalesce((SELECT max(e.offset + e.length) FROM source_extent e
                                    WHERE e.input = r.input AND e.chain = r.chain), 0)
         ORDER BY r.input, r.number",
    )?;
    Ok(())
}

/// SQLite's `quick_check` of every attached database but the temporary one.
fn databases(pool: &ReadPool, found: &mut Verification) -> Result<()> {
    let reader = pool.reader()?;
    let conn = reader.conn()?;
    let attached: Vec<(String, String)> = conn
        .prepare("PRAGMA database_list")?
        .query_map([], |row| Ok((row.get(1)?, row.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    for (schema, file) in attached {
        if schema == "temp"
            || !schema
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            continue;
        }
        let result: Vec<String> = conn
            .prepare(&format!("PRAGMA \"{schema}\".quick_check"))?
            .query_map([], |row| row.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        if result != ["ok"] {
            found.problem(ProblemKind::Database, &file, result.join("; "));
        }
        found.databases.push(DatabaseCheck {
            schema,
            file,
            result,
        });
    }
    Ok(())
}

/// Reads every blob in the catalog, in hash order, checking its stored
/// descriptor against the catalog's sizes, its encoded bytes against the
/// committed digest and its decoded bytes against its name.
fn blob_bytes(
    pool: &ReadPool,
    root: &Path,
    found: &mut Verification,
    pace: &mut Pace<'_>,
) -> Result<()> {
    let segments = hashed_blobs::Reader::new(root);
    let mut after: Vec<u8> = Vec::new();
    loop {
        let batch: Vec<(Vec<u8>, i64, i64, Vec<u8>)> = {
            let reader = pool.reader()?;
            let conn = reader.conn()?;
            let mut statement = conn.prepare(
                "SELECT hash, size, stored_size, storage FROM source_blob
                 WHERE hash > ?1 ORDER BY hash LIMIT ?2",
            )?;
            statement
                .query_map(params![after, BLOB_BATCH], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                })?
                .collect::<rusqlite::Result<_>>()?
        };
        let Some(last) = batch.last() else {
            return Ok(());
        };
        after = last.0.clone();
        for (hash, size, stored_size, storage) in batch {
            pace(&|| ())?;
            let name = hex(&hash);
            found.blobs.checked += 1;
            let extent = match hashed_blobs::Extent::decode(&storage) {
                Ok(extent) => extent,
                Err(error) => {
                    found.blobs.mismatched += 1;
                    found.problem(
                        ProblemKind::BlobCatalog,
                        name,
                        format!("descriptor: {error}"),
                    );
                    continue;
                }
            };
            found.blobs.bytes += extent.decoded_len;
            found.blobs.stored_bytes += extent.len;
            if extent.hash.0.as_slice() != hash.as_slice()
                || count(size) != extent.decoded_len
                || count(stored_size) != extent.len
            {
                found.blobs.mismatched += 1;
                found.problem(
                    ProblemKind::BlobCatalog,
                    name,
                    format!(
                        "catalog size {size} stored {stored_size}; descriptor size {} stored {}",
                        extent.decoded_len, extent.len
                    ),
                );
                continue;
            }
            let mut stopped = None;
            let scanned = segments.scan_checked(&[extent], || {
                pace(&|| ()).map_err(|error| {
                    stopped = Some(error);
                    hashed_blobs::Error::Refused("verification stopped".into())
                })
            });
            if let Some(error) = stopped {
                return Err(error);
            }
            let mismatches = scanned?;
            if let Some(mismatch) = mismatches.first() {
                found.blobs.mismatched += 1;
                found.problem(ProblemKind::BlobBytes, name, mismatch.error.clone());
            }
        }
    }
}

/// Hashes every revision from its chain's bytes, each chain read once.
fn write_manifest(
    pool: &ReadPool,
    blobs: &Arc<crate::blobs::Blobs>,
    out: &mut dyn Write,
    found: &mut Verification,
    pace: &mut Pace<'_>,
) -> Result<ManifestSummary> {
    let mut summary = ManifestSummary::default();
    let mut after = String::new();
    loop {
        let inputs: Vec<String> = {
            let reader = pool.reader()?;
            let conn = reader.conn()?;
            let mut statement =
                conn.prepare("SELECT id FROM source_input WHERE id > ?1 ORDER BY id LIMIT ?2")?;
            statement
                .query_map(params![after, INPUT_BATCH], |row| row.get(0))?
                .collect::<rusqlite::Result<_>>()?
        };
        let Some(last) = inputs.last() else {
            out.flush()?;
            return Ok(summary);
        };
        after = last.clone();
        for input in inputs {
            pace(&|| ())?;
            let mut lines = input_revisions(pool, blobs, &input, &mut summary, found, pace)?;
            lines.sort_by_key(|line| line.0);
            for (number, length, hash) in lines {
                writeln!(out, "{input}\t{number}\t{length}\t{hash}")?;
                summary.revisions += 1;
            }
        }
    }
}

/// A chain, its revisions as `(number, length)`, and its stored extents.
type ChainRevisions = (i64, Vec<(i64, u64)>, Option<crate::blobs::StoredChain>);

/// One input's revisions as `(number, length, blake3)`; revisions whose
/// bytes cannot be read are reported as problems and left out.
fn input_revisions(
    pool: &ReadPool,
    blobs: &Arc<crate::blobs::Blobs>,
    input: &str,
    summary: &mut ManifestSummary,
    found: &mut Verification,
    pace: &mut Pace<'_>,
) -> Result<Vec<(i64, u64, String)>> {
    // Each chain's revisions, shortest first, and its extents checked
    // under one read; the bytes are read after the reader is released, each
    // chain in its own read transaction that it ends before waiting.
    let mut chains: Vec<ChainRevisions> = Vec::new();
    {
        let reader = pool.reader()?;
        let conn = reader.conn()?;
        let revisions: Vec<(i64, i64, i64)> = conn
            .prepare(
                "SELECT chain, number, length FROM source_revision
                 WHERE input = ?1 ORDER BY chain, length, number",
            )?
            .query_map([input], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .collect::<rusqlite::Result<_>>()?;
        for (chain, number, length) in revisions {
            let length = count(length);
            match chains.last_mut() {
                Some((current, members, _)) if *current == chain => members.push((number, length)),
                _ => chains.push((chain, vec![(number, length)], None)),
            }
        }
        for (chain, members, stored) in &mut chains {
            let longest = members.last().map_or(0, |member| member.1);
            match reader.stored_chain(input, *chain, longest) {
                Ok(chain) => *stored = Some(chain),
                Err(error) => {
                    for (number, _) in members.iter() {
                        found.problem(
                            ProblemKind::RevisionUnreadable,
                            format!("{input}#{number}"),
                            error.to_string(),
                        );
                    }
                }
            }
        }
    }
    let mut lines = Vec::new();
    for (chain, members, stored) in chains {
        let Some(stored) = stored else {
            continue;
        };
        pace(&|| ())?;
        let (hashes, read, failure) = hash_prefixes(&stored, blobs, &members, pace)?;
        summary.bytes_read += read;
        // Revisions read in full before a failure keep their hashes.
        for (&(number, length), hash) in members.iter().zip(&hashes) {
            lines.push((number, length, hash.clone()));
        }
        if let Some(error) = failure {
            for (number, _) in &members[hashes.len()..] {
                found.problem(
                    ProblemKind::RevisionUnreadable,
                    format!("{input}#{number}"),
                    format!("chain {chain}: {error}"),
                );
            }
        }
    }
    Ok(lines)
}

/// The BLAKE3 of each prefix `members` names (shortest first), from one
/// sequential read of the chain, and the bytes read. A read failure ends
/// the read: the hashes finished before it are returned with the failure.
/// Only an error from `pace` (cancellation) ends the whole check.
fn hash_prefixes(
    stored: &crate::blobs::StoredChain,
    blobs: &Arc<crate::blobs::Blobs>,
    members: &[(i64, u64)],
    pace: &mut Pace<'_>,
) -> Result<(Vec<String>, u64, Option<StoreError>)> {
    let mut reader = stored.reader(Arc::clone(blobs));
    let mut digest = blake3::Hasher::new();
    let mut hashes = Vec::with_capacity(members.len());
    let mut at = 0u64;
    let mut buffer = vec![0; 64 << 10];
    for &(_, length) in members {
        while at < length {
            pace(&|| reader.release())?;
            let want = usize::try_from((length - at).min(buffer.len() as u64))
                .expect("buffer length fits");
            let n = match reader.read(&mut buffer[..want]) {
                Ok(0) => {
                    let error =
                        StoreError::Invalid(format!("stored bytes end at {at}, before {length}"));
                    return Ok((hashes, at, Some(error)));
                }
                Ok(n) => n,
                Err(error) => return Ok((hashes, at, Some(error.into()))),
            };
            digest.update(&buffer[..n]);
            at += n as u64;
        }
        hashes.push(digest.clone().finalize().to_hex().to_string());
    }
    Ok((hashes, at, None))
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
}

#[cfg(test)]
mod tests;
