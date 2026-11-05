//! `polylogue import --from-archive PATH`: an archive written by a Polylogue
//! build before the version 1 baseline, imported once as a declared older
//! input format ([`polylogue_store::older_archive`]).
//!
//! Every row of the older archive lands here or is reported as refused with
//! its reason. Stored bytes keep their inputs, revisions, extents and
//! observations exactly; notes and settings become assertions; deliveries,
//! their events and kept results become the requests that wrote them;
//! vectors keep their values. Every row other than the stored bytes' is also
//! kept verbatim as one imported input (`archive.db!/rows.jsonl`). Importing
//! the same older archive again changes nothing.

use super::*;
use polylogue_store::older_archive::{self, OlderArchive, authored};
use polylogue_store::writer::{OlderGroup, Restored};
use std::collections::BTreeSet;

/// Inputs read and written together, at most.
const INPUTS_PER_GROUP: usize = 256;
/// Bytes of prepared blobs held for one group before it is written.
const BYTES_PER_GROUP: u64 = 256 << 20;
/// Vectors written in one transaction.
const VECTORS_PER_GROUP: usize = 4096;

/// What became of the rows of one class. `found` is always the sum of the
/// other four.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct ClassCount {
    /// Rows the older archive holds.
    pub found: u64,
    /// Written by this import (in a dry run: would be written).
    pub stored: u64,
    /// Already here unchanged, from an earlier import.
    pub present: u64,
    /// Not carried; each is listed in the report's `refused` with its reason.
    pub refused: u64,
    /// Kept only verbatim in the rows input: rows this archive recomputes,
    /// and requests whose effect is carried as a note or setting.
    pub kept: u64,
}

/// One row, or one table's rows, the import did not carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct RefusedRow {
    pub class: String,
    /// The row's identity in the older archive.
    pub row: String,
    pub reason: String,
}

#[derive(Debug, Default, Clone, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct OlderArchiveReport {
    pub path: PathBuf,
    /// The older archive's `user_version`.
    pub version: u32,
    /// Read and checked the older archive without writing anything here.
    /// Rows already here are not told apart in a dry run: they count as stored.
    pub dry_run: bool,
    /// Per class of row, by the older archive's table name.
    pub classes: BTreeMap<String, ClassCount>,
    pub refused: Vec<RefusedRow>,
    /// Observations whose location the older format wrote as display text
    /// that may not be the exact path bytes; they keep no exact location.
    pub inexact_locations: u64,
    /// The imported input keeping every row other than the stored bytes'
    /// verbatim, by its ID.
    pub rows_input: Option<String>,
    /// The imported input setting aside every row and the stored bytes of
    /// each refused input, with its refusal reason, by its ID.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refused_inputs: Option<String>,
    /// What a dry run cannot tell.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry_run_limits: Option<String>,
}

impl OlderArchiveReport {
    fn count(&mut self, class: &str) -> &mut ClassCount {
        self.classes.entry(class.to_owned()).or_default()
    }

    fn tally(&mut self, class: &str, n: u64, outcome: &Restored, row: &str) {
        let count = self.count(class);
        count.found += n;
        match outcome {
            Restored::Stored => count.stored += n,
            Restored::Present => count.present += n,
            Restored::Refused(reason) => {
                count.refused += n;
                let reason = reason.clone();
                self.refused.push(RefusedRow {
                    class: class.to_owned(),
                    row: row.to_owned(),
                    reason,
                });
            }
        }
    }

    /// Rows counted with the row that carries them (a revision with its
    /// input): refused with it, without a refusal of their own.
    fn tally_with(&mut self, class: &str, n: u64, outcome: &Restored) {
        let count = self.count(class);
        count.found += n;
        match outcome {
            Restored::Stored => count.stored += n,
            Restored::Present => count.present += n,
            Restored::Refused(_) => count.refused += n,
        }
    }

    fn keep(&mut self, class: &str, n: u64) {
        let count = self.count(class);
        count.found += n;
        count.kept += n;
    }
}

fn restored(outcome: &polylogue_model::user::UserOutcome) -> Restored {
    if outcome.repeat {
        Restored::Present
    } else {
        Restored::Stored
    }
}

/// A store error about the row itself (its content conflicts with what is
/// here, is invalid, or names something missing) refuses that row; any
/// other error (the writer, the disk, SQLite) stops the import, so a
/// failure of this host is never reported as a refused row.
fn refusal(error: polylogue_store::StoreError) -> Result<Restored, EngineError> {
    match error {
        polylogue_store::StoreError::Conflict(_)
        | polylogue_store::StoreError::Invalid(_)
        | polylogue_store::StoreError::NotFound(_) => Ok(Restored::Refused(error.to_string())),
        error => Err(error.into()),
    }
}

/// What is known of each blob the import has met.
enum BlobState {
    /// Held here, or prepared in the current group.
    Ready,
    Refused(String),
}

impl Engine {
    /// Imports the older archive in `root` into this archive, or with
    /// `dry_run` only reads and checks it.
    pub fn import_older_archive(
        &mut self,
        root: &Path,
        dry_run: bool,
        authorship: Option<&polylogue_model::authorship::Authorship>,
    ) -> Result<OlderArchiveReport, EngineError> {
        if self.archive.is_primary() {
            return Err(std::io::Error::other(
                "an archive in the older format is imported by a standalone archive",
            )
            .into());
        }
        let scratch = self
            .archive
            .archive_db()
            .parent()
            .map(|root| root.join("older-archive-copies"))
            .ok_or_else(|| std::io::Error::other("the archive has no folder"))?;
        let older = OlderArchive::open(root, &scratch)?;
        let mut report = OlderArchiveReport {
            path: root.to_path_buf(),
            version: older.version(),
            dry_run,
            dry_run_limits: dry_run.then(|| {
                "writer-side refusals are only known on import: an input this archive holds \
                 differently, an observation ID already taken, an assertion or request that \
                 conflicts with one held, and a vector of the wrong size or different from one held"
                    .to_owned()
            }),
            ..Default::default()
        };
        let counts = older.row_counts()?;
        let found = |table: &str| counts.get(table).copied().unwrap_or(0);

        // The rows input: every row other than the stored bytes', verbatim.
        let (rows_bytes, _) = self.archive.prepare_write(|out| older.write_rows(out))?;
        let rows_id = rows_bytes.hash().to_string();
        report.rows_input = Some(rows_id.clone());
        let current = if dry_run {
            None
        } else {
            self.store_kept_member(root, "rows.jsonl", rows_bytes, authorship)?;
            let chain = self
                .reads
                .reader()?
                .input_progress(&rows_id)?
                .ok_or_else(|| std::io::Error::other("the rows input was not stored"))?
                .chain;
            Some(polylogue_model::authored::ImportAttribution {
                machine: self.config.machine.clone(),
                input: rows_id.clone(),
                chain,
                actor: authorship.map_or_else(
                    || polylogue_model::Author::Script("polylogue-import".into()),
                    |authorship| authorship.author.clone(),
                ),
                at: self.config.clock.now().into(),
            })
        };

        self.import_older_inputs(&older, dry_run, &mut report)?;
        // Refused inputs are set aside whole, with their bytes and reasons.
        let refused_inputs: Vec<(String, String)> = report
            .refused
            .iter()
            .filter(|row| row.class == "input")
            .map(|row| (row.row.clone(), row.reason.clone()))
            .collect();
        if !refused_inputs.is_empty() {
            let (set_aside, _) = self
                .archive
                .prepare_write(|out| older.write_refused_inputs(&refused_inputs, out))?;
            report.refused_inputs = Some(set_aside.hash().to_string());
            if !dry_run {
                self.store_kept_member(root, "refused-inputs.jsonl", set_aside, authorship)?;
            }
        }
        self.import_older_authored(&older, current.as_ref(), &mut report)?;
        self.import_older_vectors(&older, dry_run, &mut report)?;

        for table in older_archive::QUEUE_TABLES {
            let n = found(table);
            if n > 0 {
                report.tally(
                    table,
                    n,
                    &Restored::Refused(
                        "chat actions and backfill jobs are operational queues with no counterpart here; their rows are in the rows input".into(),
                    ),
                    &format!("{n} rows"),
                );
            }
        }
        let members = found("observation_zip");
        if members > 0 {
            report.tally(
                "observation_zip",
                members,
                &Restored::Refused(
                    "export membership was recorded without the member's name; the inputs read as members are refused".into(),
                ),
                &format!("{members} rows"),
            );
        }
        for table in older_archive::EXCISION_TABLES {
            let n = found(table);
            if n > 0 {
                report.tally(
                    table,
                    n,
                    &Restored::Refused(
                        "this archive has no excision; the rows are in the rows input".into(),
                    ),
                    &format!("{n} rows"),
                );
            }
        }
        for table in older_archive::DERIVED_TABLES {
            report.keep(table, found(table));
        }
        if !dry_run {
            // The next pass reloads the watched locations, now including
            // the imported inputs, and reads their chains into the index.
            self.loaded = false;
        }
        Ok(report)
    }

    /// Stores `bytes` as an imported kept file, `archive.db!/<member>` of
    /// the older folder `root`.
    fn store_kept_member(
        &mut self,
        root: &Path,
        member: &str,
        bytes: polylogue_store::staging::PreparedBytes,
        authorship: Option<&polylogue_model::authorship::Authorship>,
    ) -> Result<(), EngineError> {
        let mut import = ImportReport::default();
        let at = root.join("archive.db!").join(member);
        let origin = polylogue_sources::origin::by_name(KEPT)
            .ok_or_else(|| std::io::Error::other("the kept-file origin is not declared"))?;
        let imported = polylogue_store::ImportMember {
            version: 1,
            import_id: import_identity(
                &self.config.machine,
                root,
                authorship,
                std::iter::once((at.as_path(), bytes.hash())),
            ),
            authorship: authorship.cloned(),
            member: member.into(),
            format: None,
        };
        self.store_import_group(
            vec![(
                at,
                origin,
                None,
                Captured {
                    prepared: bytes,
                    #[cfg(test)]
                    head: Vec::new(),
                    member: Some(imported),
                    container: None,
                },
            )],
            &mut import,
        )?;
        if let Some(error) = import.pass.errors.first() {
            return Err(
                std::io::Error::other(format!("{member} could not be stored: {error}")).into(),
            );
        }
        Ok(())
    }

    /// Stored bytes: inputs with every revision, extent and observation, and
    /// every blob, cited or not.
    fn import_older_inputs(
        &mut self,
        older: &OlderArchive,
        dry_run: bool,
        report: &mut OlderArchiveReport,
    ) -> Result<(), EngineError> {
        let machines = older.machines().to_vec();
        let mut blobs: HashMap<polylogue_model::BlobHash, BlobState> = HashMap::new();
        let mut cited: BTreeSet<polylogue_model::BlobHash> = BTreeSet::new();
        let mut machines_sent = false;
        let mut after: Option<String> = None;
        loop {
            let ids = older.input_ids(after.as_deref(), INPUTS_PER_GROUP)?;
            let Some(last) = ids.last() else {
                break;
            };
            after = Some(last.clone());
            let mut inputs = ids
                .iter()
                .map(|id| older.input(id))
                .collect::<polylogue_store::Result<Vec<_>>>()?;
            // An input this archive does not hold must not land where this
            // archive already watches another input the older archive does
            // not hold either: one file would have two inputs.
            {
                let reader = self.reads.reader()?;
                for input in &mut inputs {
                    report.inexact_locations += input.inexact_locations;
                    if input.refusal.is_some() || reader.input_progress(&input.id)?.is_some() {
                        continue;
                    }
                    let mut checked = BTreeSet::new();
                    for observation in &input.observations {
                        let Some(path) = &observation.path_bytes else {
                            continue;
                        };
                        if !checked.insert((observation.machine.clone(), path.clone())) {
                            continue;
                        }
                        if let Some(known) =
                            reader.known_watched_input_at(&observation.machine, path)?
                            && known.input_id != input.id
                            && !older.has_input(&known.input_id)?
                        {
                            input.refusal = Some(format!(
                                "this archive already watches {} as input {}; import into a fresh archive",
                                observation.path, known.input_id
                            ));
                            break;
                        }
                    }
                }
            }
            let mut group = OlderGroup::default();
            let mut prepared_bytes = 0;
            for mut input in inputs {
                for hash in input.blobs() {
                    cited.insert(hash);
                    if let std::collections::hash_map::Entry::Vacant(slot) = blobs.entry(hash) {
                        slot.insert(self.prepare_older_blob(
                            older,
                            &hash,
                            dry_run,
                            &mut group,
                            &mut prepared_bytes,
                            report,
                        )?);
                    }
                    if input.refusal.is_none()
                        && let Some(BlobState::Refused(reason)) = blobs.get(&hash)
                    {
                        input.refusal = Some(format!("its blob {hash} is refused: {reason}"));
                    }
                }
                group.inputs.push(input);
                if prepared_bytes >= BYTES_PER_GROUP {
                    self.write_older_group(
                        std::mem::take(&mut group),
                        &machines,
                        &mut machines_sent,
                        dry_run,
                        report,
                    )?;
                    prepared_bytes = 0;
                }
            }
            self.write_older_group(group, &machines, &mut machines_sent, dry_run, report)?;
        }
        // Blobs no extent cites (a chat action's attachment, or bytes left
        // by an older build) are carried too.
        let mut group = OlderGroup::default();
        let mut prepared_bytes = 0;
        for hash in older.blob_hashes()? {
            if cited.contains(&hash) || blobs.contains_key(&hash) {
                continue;
            }
            let state = self.prepare_older_blob(
                older,
                &hash,
                dry_run,
                &mut group,
                &mut prepared_bytes,
                report,
            )?;
            blobs.insert(hash, state);
            if prepared_bytes >= BYTES_PER_GROUP {
                self.write_older_group(
                    std::mem::take(&mut group),
                    &machines,
                    &mut machines_sent,
                    dry_run,
                    report,
                )?;
                prepared_bytes = 0;
            }
        }
        self.write_older_group(group, &machines, &mut machines_sent, dry_run, report)?;
        if !machines_sent {
            self.write_older_group(
                OlderGroup::default(),
                &machines,
                &mut machines_sent,
                dry_run,
                report,
            )?;
        }
        Ok(())
    }

    /// Checks one blob and, unless this archive holds it already, prepares
    /// its bytes for `group`. Counts it in the report.
    fn prepare_older_blob(
        &self,
        older: &OlderArchive,
        hash: &polylogue_model::BlobHash,
        dry_run: bool,
        group: &mut OlderGroup,
        prepared_bytes: &mut u64,
        report: &mut OlderArchiveReport,
    ) -> Result<BlobState, EngineError> {
        let Some(blob) = older.blob(hash)? else {
            return Ok(BlobState::Refused(
                "an extent cites it but the older archive has no row for it".into(),
            ));
        };
        if !dry_run && self.blobs.extent(&hashed_blobs::Hash(hash.0)).is_ok() {
            report.tally("blob", 1, &Restored::Present, &hash.to_string());
            return Ok(BlobState::Ready);
        }
        let (prepared, checked) = self
            .archive
            .prepare_write(|out| older.copy_blob(&blob, out))?;
        let checked = checked.and_then(|()| {
            if prepared.hash() == *hash && prepared.len() == blob.size {
                Ok(())
            } else {
                Err("its prepared bytes differ from its name".into())
            }
        });
        match checked {
            Ok(()) => {
                report.tally("blob", 1, &Restored::Stored, &hash.to_string());
                if !dry_run {
                    *prepared_bytes += prepared.len();
                    group.files.push(prepared);
                }
                Ok(BlobState::Ready)
            }
            Err(reason) => {
                report.tally(
                    "blob",
                    1,
                    &Restored::Refused(reason.clone()),
                    &hash.to_string(),
                );
                Ok(BlobState::Refused(reason))
            }
        }
    }

    fn write_older_group(
        &mut self,
        mut group: OlderGroup,
        machines: &[older_archive::Machine],
        machines_sent: &mut bool,
        dry_run: bool,
        report: &mut OlderArchiveReport,
    ) -> Result<(), EngineError> {
        if group.inputs.is_empty() && group.files.is_empty() && *machines_sent {
            return Ok(());
        }
        let first = !*machines_sent;
        if first {
            group.machines = machines.to_vec();
            *machines_sent = true;
        }
        let outcome = if dry_run {
            polylogue_store::writer::GroupOutcome {
                inputs: group
                    .inputs
                    .iter()
                    .map(|input| match &input.refusal {
                        Some(reason) => Restored::Refused(reason.clone()),
                        None => Restored::Stored,
                    })
                    .collect(),
                machines: vec![Restored::Stored; group.machines.len()],
            }
        } else {
            let inputs = std::mem::take(&mut group.inputs);
            let machines = group.machines.clone();
            let outcome = self.writer.sender()?.restore_older(OlderGroup {
                files: group.files,
                machines,
                inputs: inputs.clone(),
            })?;
            group.inputs = inputs;
            outcome
        };
        for (machine, restored) in group.machines.iter().zip(&outcome.machines) {
            report.tally(
                "machine",
                1,
                restored,
                &format!("{} {}", machine.id, machine.host),
            );
        }
        for (input, restored) in group.inputs.iter().zip(&outcome.inputs) {
            report.tally("input", 1, restored, &input.id);
            report.tally_with("revision", input.revisions.len() as u64, restored);
            report.tally_with("extent", input.extents.len() as u64, restored);
            report.tally_with("observation", input.observations.len() as u64, restored);
        }
        Ok(())
    }

    /// Notes, settings, deliveries with their events, kept results and the
    /// requests that wrote them.
    fn import_older_authored(
        &mut self,
        older: &OlderArchive,
        current: Option<&polylogue_model::authored::ImportAttribution>,
        report: &mut OlderArchiveReport,
    ) -> Result<(), EngineError> {
        let writer = self.writer.sender()?;
        let restore_fact =
            |fact: polylogue_model::assertion::AssertionFact| -> Result<Restored, EngineError> {
                let Some(_) = current else {
                    return Ok(Restored::Stored);
                };
                Ok(match writer.restore_assertion(fact) {
                    Ok(outcome) => restored(&outcome),
                    Err(error) => refusal(error)?,
                })
            };

        let notes = older.notes()?;
        let roots = authored::roots(&notes);
        for note in &notes {
            let outcome = match authored::note(note, &roots) {
                Ok(fact) => restore_fact(fact)?,
                Err(reason) => Restored::Refused(reason),
            };
            report.tally("note", 1, &outcome, &note.id);
            report.tally_with("note_target", note.targets.len() as u64, &outcome);
            report.tally_with("note_relation", note.relations.len() as u64, &outcome);
        }
        for (table, row) in older.loose_note_rows()? {
            report.tally(
                &table,
                1,
                &Restored::Refused("the note it belongs to is not in the older archive".into()),
                &row,
            );
        }
        for (id, mapped) in authored::settings(&older.settings()?, older.machines()) {
            let outcome = match mapped {
                Ok(fact) => restore_fact(fact)?,
                Err(reason) => Restored::Refused(reason),
            };
            report.tally("setting", 1, &outcome, &id);
        }

        let deliveries = older.deliveries()?;
        let events = older.delivery_events()?;
        let rows = authored::requests(
            &older.requests()?,
            &deliveries,
            &events,
            &older.kept_results()?,
        );
        // Which deliveries and events were here before: the rest that are
        // here afterwards were stored by this import, whichever of the
        // requests carrying a delivery's record stored it.
        let before = match current {
            Some(_) => Some(self.reads.reader()?.stored_deliveries()?),
            None => None,
        };
        // Why each delivery's own request did not store it.
        let mut not_stored: BTreeMap<String, String> = rows.deliveries.iter().cloned().collect();
        for mapped in rows.requests {
            let outcome = match mapped.request {
                Err(reason) => Restored::Refused(reason),
                Ok(None) => {
                    report.keep("client_request", 1);
                    continue;
                }
                Ok(Some(prepared)) => {
                    let carried = &prepared.effect;
                    let delivery = mapped.class == Some(authored::RequestClass::Delivery);
                    let kept = (mapped.class == Some(authored::RequestClass::KeptResult))
                        .then(|| carried.kept_results.first().map(|k| k.id.clone()))
                        .flatten();
                    let outcome = match current {
                        None => Restored::Stored,
                        Some(_) => match writer.restore_authored(prepared) {
                            Ok(outcome) => restored(&outcome),
                            Err(error) => refusal(error)?,
                        },
                    };
                    if delivery && let Restored::Refused(reason) = &outcome {
                        not_stored.insert(mapped.id.clone(), reason.clone());
                    }
                    if let Some(kept) = kept {
                        report.tally("kept_result", 1, &outcome, &kept);
                    }
                    outcome
                }
            };
            report.tally("client_request", 1, &outcome, &mapped.id);
        }
        let after = match current {
            Some(_) => Some(self.reads.reader()?.stored_deliveries()?),
            None => None,
        };
        for delivery in &deliveries {
            let outcome = match (&before, &after) {
                (Some(before), Some(after)) => {
                    if !after.ids.contains(&delivery.id) {
                        Restored::Refused(
                            not_stored
                                .get(&delivery.id)
                                .cloned()
                                .unwrap_or_else(|| "no request carrying it stored it".to_owned()),
                        )
                    } else if before.ids.contains(&delivery.id) {
                        Restored::Present
                    } else {
                        Restored::Stored
                    }
                }
                // A dry run: carried unless its mapping refused it.
                _ => match not_stored.get(&delivery.id) {
                    Some(reason) => Restored::Refused(reason.clone()),
                    None => Restored::Stored,
                },
            };
            report.tally("delivery", 1, &outcome, &delivery.id);
            for event in events.iter().filter(|e| e.delivery == delivery.id) {
                let key = (event.delivery.clone(), event.what.clone(), event.at);
                let event_outcome = match (&before, &after, &outcome) {
                    // An event is refused with its delivery.
                    (_, _, Restored::Refused(_)) | (None, _, _) | (_, None, _) => outcome.clone(),
                    (Some(before), Some(after), _) => {
                        if !after.events.contains(&key) {
                            Restored::Refused("no request carrying its delivery stored it".into())
                        } else if before.events.contains(&key) {
                            Restored::Present
                        } else {
                            Restored::Stored
                        }
                    }
                };
                if matches!(outcome, Restored::Refused(_)) {
                    report.tally_with("delivery_event", 1, &event_outcome);
                } else {
                    report.tally(
                        "delivery_event",
                        1,
                        &event_outcome,
                        &format!("{} {} {}", event.delivery, event.what, event.at),
                    );
                }
            }
        }
        for (id, reason) in rows.kept_results {
            report.tally("kept_result", 1, &Restored::Refused(reason), &id);
        }
        Ok(())
    }

    /// Vectors and permanent embedding failures, with their original values.
    fn import_older_vectors(
        &mut self,
        older: &OlderArchive,
        dry_run: bool,
        report: &mut OlderArchiveReport,
    ) -> Result<(), EngineError> {
        let reader = self.reads.reader()?;
        let mut vectors = if dry_run {
            None
        } else {
            Some(reader.vectors_writer()?)
        };
        let mut batch = Vec::new();
        let mut write = |batch: &mut Vec<older_archive::StoredVector>,
                         report: &mut OlderArchiveReport|
         -> polylogue_store::Result<()> {
            let outcomes = match vectors.as_mut() {
                Some(vectors) => vectors.restore(batch)?,
                None => vec![Restored::Stored; batch.len()],
            };
            for (vector, outcome) in batch.iter().zip(&outcomes) {
                let row = format!(
                    "{} {} chunk {}",
                    vector.model,
                    vector
                        .text_hash
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<String>(),
                    vector.chunk
                );
                report.tally("embedded", 1, outcome, &row);
            }
            batch.clear();
            Ok(())
        };
        older.vectors(|vector| {
            batch.push(vector);
            if batch.len() >= VECTORS_PER_GROUP {
                write(&mut batch, report)?;
            }
            Ok(())
        })?;
        write(&mut batch, report)?;
        let failures = older.vector_failures()?;
        let outcomes = match vectors.as_mut() {
            Some(vectors) => vectors.restore_failures(&failures)?,
            None => vec![Restored::Stored; failures.len()],
        };
        for (failure, outcome) in failures.iter().zip(&outcomes) {
            report.tally("embed_failure", 1, outcome, &failure.model);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
