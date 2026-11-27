//! The on-demand check of stored bytes (`polylogue status --verify`). It reads every
//! committed blob and checks its encoded bytes against the committed digest
//! and, for a plain blob, its decoded bytes against its name (a sealed
//! blob is checked by digest alone: the check holds no key), checks every extent against the
//! blob it cites and every revision against its chain's extents, compares
//! the kept counters with the rows they count, and runs SQLite's own check
//! on each database. It changes nothing: what it finds is reported, and
//! recovery is a restore from backup.
//!
//! With a manifest it also reads every revision's bytes through its chain
//! and writes one line per revision, `input<TAB>number<TAB>length<TAB>blake3`,
//! ordered by input and number, so that a manifest computed independently
//! from another copy of the same inputs can be compared byte for byte.

use crate::ConnExt as _;
use crate::row::{Rows as _, Statement, row};
use crate::{ReadPool, Result, StoreError};
use polylogue_model::hex;
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

row! {
    struct OriginCount {
        origin: String,
        inputs: i64,
        revisions: i64,
        revision_bytes: i64,
    }
    statements {
        const ALL = "SELECT i.origin, count(DISTINCT i.id) AS inputs,
            count(r.number) AS revisions, coalesce(sum(r.length), 0) AS revision_bytes
            FROM source_input i LEFT JOIN source_revision r ON r.input = i.id
            GROUP BY i.origin ORDER BY i.origin";
    }
}

row! {
    struct CountByClass {
        class: String,
        count: i64,
    }
    statements {
        const OBSERVATIONS = "SELECT class, count(*) AS count FROM source_observation
            GROUP BY class ORDER BY class";
        const REVISIONS = "SELECT produced AS class, count(*) AS count FROM source_revision
            GROUP BY produced ORDER BY produced";
    }
}

row! {
    struct StoredCount {
        name: String,
        count: i64,
    }
    statements {
        const ALL = "SELECT name, count FROM source_row_count ORDER BY name";
    }
}

row! {
    struct ProblemRow {
        subject: String,
        detail: String,
    }
    statements {
        const EXTENTS_WITHOUT_BLOBS = "SELECT e.input || '/' || e.chain || '@' || e.offset AS subject, 'blob ' || lower(hex(e.blob)) AS detail
         FROM source_extent e
         WHERE NOT EXISTS (SELECT 1 FROM source_blob b WHERE b.hash = e.blob)
         ORDER BY e.input, e.chain, e.offset";
        const EXTENTS_OUTSIDE_BLOBS = "SELECT e.input || '/' || e.chain || '@' || e.offset AS subject,
                'bytes ' || e.blob_offset || '..' || (e.blob_offset + e.length)
                || ' of a blob of ' || b.size AS detail
         FROM source_extent e JOIN source_blob b ON b.hash = e.blob
         WHERE e.length <= 0 OR e.offset < 0 OR e.blob_offset < 0
            OR e.blob_offset + e.length > b.size
         ORDER BY e.input, e.chain, e.offset";
        const EXTENT_GAPS = "SELECT g.input || '/' || g.chain || '@' || g.start AS subject,
                CASE WHEN g.start > g.previous_end THEN 'gap after ' ELSE 'overlap with bytes before ' END
                || g.previous_end AS detail
         FROM (SELECT x.input AS input, x.chain AS chain, x.offset AS start,
                      lag(x.offset + x.length, 1, 0)
                        OVER (PARTITION BY x.input, x.chain ORDER BY x.offset) AS previous_end
               FROM source_extent x) g
         WHERE g.start != g.previous_end
         ORDER BY g.input, g.chain, g.start";
        const UNCOVERED_REVISIONS = "SELECT r.input || '#' || r.number AS subject,
                'length ' || r.length || ', extents of chain ' || r.chain || ' cover '
                || coalesce((SELECT max(e.offset + e.length) FROM source_extent e
                             WHERE e.input = r.input AND e.chain = r.chain), 0) AS detail
         FROM source_revision r
         WHERE r.length < 0
            OR r.length > coalesce((SELECT max(e.offset + e.length) FROM source_extent e
                                    WHERE e.input = r.input AND e.chain = r.chain), 0)
         ORDER BY r.input, r.number";
    }
}

row! {
    struct DatabaseFile {
        seq: i64,
        name: String,
        file: String,
    }
    statements {
        const ALL = "PRAGMA database_list";
    }
}

row! {
    struct QuickCheck {
        quick_check: String,
    }
    statements {
        const MAIN = "PRAGMA \"main\".quick_check";
    }
}

row! {
    struct BlobRow {
        hash: Vec<u8>,
        size: i64,
        stored_size: i64,
        storage: Vec<u8>,
    }
    statements {
        const BATCH = "SELECT hash, size, stored_size, storage FROM source_blob
            WHERE hash > ?1 ORDER BY hash LIMIT ?2";
    }
}

row! {
    struct ManifestInput {
        id: String,
    }
    statements {
        const BATCH = "SELECT id FROM source_input WHERE id > ?1 ORDER BY id LIMIT ?2";
    }
}

row! {
    struct ManifestRevision {
        chain: i64,
        number: i64,
        length: i64,
    }
    statements {
        const OF_INPUT = "SELECT chain, number, length FROM source_revision
            WHERE input = ?1 ORDER BY chain, length, number";
    }
}

#[cfg(test)]
pub(crate) fn declared_rows(check: &mut crate::row::Check<'_>) {
    check.row::<OriginCount>();
    check.row::<CountByClass>();
    check.row::<StoredCount>();
    check.row::<ProblemRow>();
    check.row::<DatabaseFile>();
    check.row::<QuickCheck>();
    check.row::<BlobRow>();
    check.row::<ManifestInput>();
    check.row::<ManifestRevision>();
}

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

/// A stored count; a negative value cannot be one, and every check that could
/// meet one (kept counters, blob sizes, revision lengths) reports it as a problem.
fn count(n: i64) -> u64 {
    u64::try_from(n).unwrap_or(0)
}

/// Counts, per-class counts, counters and extent structure, in one read.
fn rows(pool: &ReadPool, found: &mut Verification) -> Result<()> {
    let reader = pool.reader()?;
    let conn = reader.conn()?;
    let one = |sql: &str| -> Result<i64> { conn.scalar(sql, []) };
    found.counts = Counts {
        inputs: count(one("SELECT count(*) FROM source_input")?),
        revisions: count(one("SELECT count(*) FROM source_revision")?),
        extents: count(one("SELECT count(*) FROM source_extent")?),
        observations: count(one("SELECT count(*) FROM source_observation")?),
        blobs: count(one("SELECT count(*) FROM source_blob")?),
        blob_bytes: count(one("SELECT coalesce(sum(size), 0) FROM source_blob")?),
    };
    found.formats = conn
        .all(&OriginCount::ALL, [])?
        .into_iter()
        .map(|row| FormatCount {
            format: row.origin,
            inputs: count(row.inputs),
            revisions: count(row.revisions),
            revision_bytes: count(row.revision_bytes),
        })
        .collect();
    let classes = |statement: &Statement<CountByClass>| -> Result<Vec<ClassCount>> {
        Ok(conn
            .all(statement, [])?
            .into_iter()
            .map(|row| ClassCount {
                class: row.class,
                count: count(row.count),
            })
            .collect())
    };
    found.observation_classes = classes(&CountByClass::OBSERVATIONS)?;
    found.revision_kinds = classes(&CountByClass::REVISIONS)?;

    // The counters `status` reads instead of counting.
    for StoredCount { name, count: n } in conn.all(&StoredCount::ALL, [])? {
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

    for (kind, statement) in [
        (
            ProblemKind::ExtentWithoutBlob,
            &ProblemRow::EXTENTS_WITHOUT_BLOBS,
        ),
        (
            ProblemKind::ExtentOutsideBlob,
            &ProblemRow::EXTENTS_OUTSIDE_BLOBS,
        ),
        (ProblemKind::ExtentGap, &ProblemRow::EXTENT_GAPS),
        (
            ProblemKind::RevisionUncovered,
            &ProblemRow::UNCOVERED_REVISIONS,
        ),
    ] {
        for ProblemRow { subject, detail } in conn.all(statement, [])? {
            found.problem(kind, subject, detail);
        }
    }
    Ok(())
}

/// SQLite's `quick_check` of every attached database but the temporary one.
fn databases(pool: &ReadPool, found: &mut Verification) -> Result<()> {
    let reader = pool.reader()?;
    let conn = reader.conn()?;
    for DatabaseFile {
        name: schema, file, ..
    } in conn.all(&DatabaseFile::ALL, [])?
    {
        if schema == "temp"
            || !schema
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            continue;
        }
        let attached;
        let statement = if schema == "main" {
            &QuickCheck::MAIN
        } else {
            attached = Statement::built(format!("PRAGMA \"{schema}\".quick_check"));
            &attached
        };
        let result: Vec<String> = conn
            .all(statement, [])?
            .into_iter()
            .map(|row| row.quick_check)
            .collect();
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
/// committed digest and, for a plain blob, its decoded bytes against its name.
fn blob_bytes(
    pool: &ReadPool,
    root: &Path,
    found: &mut Verification,
    pace: &mut Pace<'_>,
) -> Result<()> {
    let segments = hashed_blobs::Reader::new(root);
    let mut after: Vec<u8> = Vec::new();
    loop {
        let batch = {
            let reader = pool.reader()?;
            reader
                .conn()?
                .all(&BlobRow::BATCH, params![after, BLOB_BATCH])?
        };
        let Some(last) = batch.last() else {
            return Ok(());
        };
        after = last.hash.clone();
        for BlobRow {
            hash,
            size,
            stored_size,
            storage,
        } in batch
        {
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
                || u64::try_from(size).ok() != Some(extent.decoded_len)
                || u64::try_from(stored_size).ok() != Some(extent.len)
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
        let inputs = {
            let reader = pool.reader()?;
            reader
                .conn()?
                .all(&ManifestInput::BATCH, params![after, INPUT_BATCH])?
        };
        let Some(last) = inputs.last() else {
            out.flush()?;
            return Ok(summary);
        };
        after = last.id.clone();
        for ManifestInput { id: input } in inputs {
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
        for ManifestRevision {
            chain,
            number,
            length,
        } in conn.all(&ManifestRevision::OF_INPUT, [input])?
        {
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
                    let error = StoreError::stored_data(format!(
                        "stored bytes end at {at}, before {length}"
                    ));
                    return Ok((hashes, at, Some(error)));
                }
                Ok(n) => n,
                Err(error) => return Ok((hashes, at, Some(error.into()))),
            };
            digest.update(&buffer[..n]);
            at += n as u64;
        }
        hashes.push(digest.finalize().to_hex().to_string());
    }
    Ok((hashes, at, None))
}

#[cfg(test)]
mod tests;
