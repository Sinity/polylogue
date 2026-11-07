//! Importing exports and other files by hand: a file, a
//! directory of them, or a nested container. A declared package structure
//! fixes the origins of its members before any bytes are committed.

use super::*;
use polylogue_sources::origin::Arrival;
use std::collections::BTreeSet;

mod assets;
mod authored;
mod bundle;
pub(crate) mod exports;
mod legacy;
pub(crate) mod older_archive;
mod package;
mod recognition;

impl Engine {
    /// Imports immutable content identified by the BLAKE3 of its original
    /// bytes. The bytes are hashed and compressed into a private file; the
    /// writer commits their durable extents before they are parsed.
    /// Origins read as whole documents still parse one document in one job.
    pub fn import(&mut self, paths: &[PathBuf]) -> Result<ImportReport, EngineError> {
        self.import_with_options(paths, &ImportOptions::default())
    }

    pub fn import_with_options(
        &mut self,
        paths: &[PathBuf],
        options: &ImportOptions,
    ) -> Result<ImportReport, EngineError> {
        let started = Instant::now();
        let mut report = ImportReport::default();
        if let Some(root) = &options.from_archive {
            if !paths.is_empty() || options.format.is_some() || options.explain {
                return Err(std::io::Error::other(
                    "--from-archive names one archive and takes no other paths, --format or --explain",
                )
                .into());
            }
            report.older_archive = Some(self.import_older_archive(
                root,
                options.dry_run,
                options.authorship.as_ref(),
            )?);
            report.pass.seconds = started.elapsed().as_secs_f64();
            return Ok(report);
        }
        if options.dry_run {
            return Err(std::io::Error::other("--dry-run applies to --from-archive").into());
        }
        for path in paths {
            let members = match package::stage(path, &self.archive, options.format.as_deref()) {
                Ok(members) => members,
                Err(error) => {
                    recognition::refuse(&mut report, path, error.to_string());
                    continue;
                }
            };
            let mut package = ImportPackage {
                path: path.clone(),
                ..Default::default()
            };
            let mut ids = Vec::new();
            let mut authored_records = Vec::new();
            let mut items = Vec::new();
            let mut hashes = std::collections::BTreeMap::new();
            let import_id = import_identity(
                &self.config.machine,
                path,
                options.authorship.as_ref(),
                members
                    .iter()
                    .map(|member| (member.path.as_path(), member.bytes.hash())),
            );
            for (index, member) in members.into_iter().enumerate() {
                let Some(origin) = member.origin else {
                    recognition::refuse(&mut report, &member.path, "no declared structure matched");
                    report.members.push(ImportClassification {
                        path: member.path,
                        origin: None,
                        class: "refused".into(),
                        reason: Some("no declared structure matched".into()),
                    });
                    continue;
                };
                report.members.push(ImportClassification {
                    path: member.path.clone(),
                    origin: Some(origin.name.into()),
                    class: if origin.support == Support::Parsed {
                        "parsed"
                    } else {
                        "asset"
                    }
                    .into(),
                    reason: None,
                });
                if origin.support != Support::Parsed && !member.container {
                    package.assets += 1;
                }
                if options.explain {
                    continue;
                }
                let authored_bytes =
                    (origin.name == "polylogue-bundle").then(|| member.bytes.clone());
                let mut captured = Captured {
                    prepared: member.bytes,
                    member: None,
                    container: None,
                    #[cfg(test)]
                    head: Vec::new(),
                };
                ids.push(captured.prepared.hash().to_string());
                hashes.insert(index, captured.prepared.hash().to_string());
                captured.member = Some(polylogue_store::ImportMember {
                    version: 1,
                    import_id: import_id.clone(),
                    authorship: Some(options.authorship.clone().unwrap_or_else(|| {
                        polylogue_model::authorship::Authorship {
                            author: polylogue_model::Author::Script("polylogue-import".into()),
                            via: polylogue_model::authorship::Channel::Import,
                        }
                    })),
                    member: if member.parent.is_some() {
                        member
                            .path
                            .to_string_lossy()
                            .rsplit("!/")
                            .next()
                            .expect("member name")
                            .to_owned()
                    } else if member.path == *path {
                        member
                            .path
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into_owned()
                    } else {
                        member
                            .path
                            .strip_prefix(path)
                            .unwrap_or(&member.path)
                            .to_string_lossy()
                            .into_owned()
                    },
                    format: None,
                });
                captured.container = member.parent.and_then(|(parent, member_index)| {
                    hashes
                        .get(&parent)
                        .map(|input| polylogue_store::InContainer {
                            input: input.clone(),
                            member_index,
                            kind: member
                                .container_kind
                                .clone()
                                .unwrap_or_else(|| "container".into()),
                            path_bytes: member.member_path_bytes.clone().unwrap_or_default(),
                        })
                });
                if origin.name == "polylogue-bundle" {
                    authored_records.push((
                        member.path.clone(),
                        captured.prepared.hash().to_string(),
                        authored_bytes.expect("bundle reader"),
                    ));
                }
                items.push((member.path, origin, None, captured));
            }
            if !options.explain {
                let executed = self.store_import_group(items, &mut report)?;
                for (at, id, bytes) in authored_records {
                    if executed.contains(&id) {
                        continue;
                    }
                    if self.reads.reader()?.input_progress(&id)?.is_some() {
                        self.restore_bundle_authored(&at, &mut bytes.reader()?, &mut report)?;
                    }
                }
                let counts = self.reads.reader()?.import_counts(&ids)?;
                package.sessions = counts.sessions;
                package.documents = counts.documents;
            }
            report.packages.push(package);
        }
        if options.explain {
            report.pass.seconds = started.elapsed().as_secs_f64();
            return Ok(report);
        }
        report.user_state.applied += report.pass.authored.applied;
        report.user_state.present += report.pass.authored.present;
        report.user_state.pending += report.pass.authored.pending;
        self.refresh_import_assets(&mut report.pass)?;
        report.pass.sessions_derived = self.derive()?;
        report.pass.seconds = started.elapsed().as_secs_f64();
        Ok(report)
    }

    fn store_import_group(
        &mut self,
        items: Vec<(PathBuf, &'static OriginSpec, Option<FileFacts>, Captured)>,
        report: &mut ImportReport,
    ) -> Result<BTreeSet<String>, EngineError> {
        let mut executed = BTreeSet::new();
        // Every explicit import reaches the writer, including stored bytes:
        // it checks the format and records this receiver's observation.
        // The blob writer deduplicates bytes within the same transaction.
        let mut metadata = Vec::new();
        let mut group = Vec::new();
        for (at, origin, facts, captured) in items {
            let mut batch = self.import_batch(
                &at,
                origin,
                captured.member,
                facts,
                captured.prepared.hash(),
                captured.prepared.len(),
            );
            batch
                .observation
                .as_mut()
                .expect("import observation")
                .container = captured.container;
            batch.new_blob = Some(polylogue_store::NewBlob {
                hash: captured.prepared.hash(),
                size: captured.prepared.len(),
            });
            metadata.push((at, origin, facts, batch.input.clone()));
            group.push(polylogue_store::staging::PreparedBatch {
                batch,
                files: vec![captured.prepared],
            });
        }
        if group.is_empty() {
            return Ok(executed);
        }
        let outcomes = self
            .writer
            .sender()?
            .store_group(group, Arc::clone(&self.blobs))?;
        for ((at, origin, facts, input), outcome) in metadata.into_iter().zip(outcomes) {
            let stored = match outcome {
                Ok(stored) => stored,
                Err(polylogue_store::StoreError::PrimaryPending { .. }) => {
                    report.pass.capture_pending = true;
                    continue;
                }
                Err(error) => {
                    report.pass.errors.push(error.to_string());
                    continue;
                }
            };
            if stored.already_stored {
                report.already_stored.push(at.clone());
            } else {
                report.imported.push(at.clone());
                report.pass.bytes_stored += stored.length;
            }
            if let Some(plan) = self.stored_import_plan(&at, origin, input, &stored, facts)? {
                executed.insert(plan.input.id.clone());
                self.execute(vec![plan], &mut report.pass)?;
            }
        }
        Ok(executed)
    }

    pub(crate) fn restore_captured_authored(
        &mut self,
        plan: &crate::plan::Plan,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        if plan.origin.name != "polylogue-bundle" {
            return Ok(());
        }
        let reader = self.reads.reader()?;
        let chain = match plan.change {
            Change::Reparse { chain } => chain,
            _ => {
                reader
                    .input_progress(&plan.input.id)?
                    .ok_or_else(|| {
                        polylogue_store::StoreError::Invalid(
                            "a stored bundle has no committed source revision".into(),
                        )
                    })?
                    .chain
            }
        };
        let stored = reader.stored_chain(&plan.input.id, chain, plan.size)?;
        drop(reader);
        polylogue_store::fault::check("import.before_authored_restore", &plan.input.id)?;
        authored::replay(stored.reader(Arc::clone(&self.blobs)), |entry| {
            let outcome = entry
                .restore(&self.writer)
                .map_err(|error| error.to_string())?;
            if outcome.pending {
                pass.authored.pending += 1;
            } else if outcome.repeat {
                pass.authored.present += 1;
            } else {
                pass.authored.applied += 1;
            }
            Ok(())
        })
        .map_err(EngineError::Parse)?;
        polylogue_store::fault::check("import.after_authored_restore", &plan.input.id)?;
        Ok(())
    }

    fn restore_bundle_authored(
        &mut self,
        path: &Path,
        file: &mut impl std::io::Read,
        report: &mut ImportReport,
    ) -> Result<(), EngineError> {
        let result = authored::replay(file, |entry| {
            let outcome = entry
                .restore(&self.writer)
                .map_err(|error| error.to_string())?;
            if outcome.pending {
                report.user_state.pending += 1;
            } else if outcome.repeat {
                report.user_state.present += 1;
            } else {
                report.user_state.applied += 1;
            }
            Ok(())
        });
        if let Err(error) = result {
            report
                .pass
                .errors
                .push(format!("{}: authored restoration: {error}", path.display()));
        }
        Ok(())
    }

    #[cfg(test)]
    fn import_captured(
        &mut self,
        path: &Path,
        origin: &'static OriginSpec,
        captured: Captured,
        facts: Option<FileFacts>,
        report: &mut ImportReport,
    ) -> Result<(), EngineError> {
        if let Some(plan) = self.import_plan(path, origin, captured, facts, report)? {
            self.execute(vec![plan], &mut report.pass)?;
        }
        Ok(())
    }

    /// Stores the complete import once, then parses it without making
    /// further new blobs. A retry can rebuild parser state from offset zero.
    #[cfg(test)]
    fn import_plan(
        &mut self,
        path: &Path,
        origin: &'static OriginSpec,
        captured: Captured,
        facts: Option<FileFacts>,
        report: &mut ImportReport,
    ) -> Result<Option<Plan>, EngineError> {
        let size = captured.prepared.len();
        let hash = captured.prepared.hash();
        let batch = self.import_batch(path, origin, captured.member, facts, hash, size);
        let input = batch.input.clone();
        let stored = match self.writer.store_bytes(captured.prepared, batch) {
            Ok(stored) => stored,
            Err(error) => {
                report.pass.errors.push(error.to_string());
                return Ok(None);
            }
        };
        if stored.already_stored {
            report.already_stored.push(path.to_path_buf());
        } else {
            report.imported.push(path.to_path_buf());
            report.pass.bytes_stored += size;
        }
        self.stored_import_plan(path, origin, input, &stored, facts)
    }

    fn import_batch(
        &self,
        path: &Path,
        origin: &'static OriginSpec,
        member: Option<polylogue_store::ImportMember>,
        facts: Option<FileFacts>,
        hash: polylogue_model::BlobHash,
        size: u64,
    ) -> polylogue_store::Batch {
        let mut observation = self.observation(path, facts);
        observation.import = member.map(|mut member| {
            member.format = Some(origin.name.into());
            member
        });
        polylogue_store::Batch {
            required_disposition: None,
            input: InputKey {
                id: hash.to_string(),
                format: origin.name.into(),
                arrival_rank: 2,
            },
            change: Change::First,
            observation: Some(observation),
            continues: false,
            new_blob: None,
            blobs: Vec::new(),
            length: size,
            parse: Vec::new(),
            acquisition: Some(
                self.acquisition_context(
                    path,
                    origin.name,
                    polylogue_store::primary::AcquisitionRoute::ExplicitImport,
                )
                .finish(
                    polylogue_store::primary::RawRange {
                        start: 0,
                        end: size,
                    },
                    hash,
                ),
            ),
            basis: None,
            release: Default::default(),
        }
    }

    fn stored_import_plan(
        &self,
        path: &Path,
        origin: &'static OriginSpec,
        input: InputKey,
        stored: &polylogue_store::staging::StoredInput,
        facts: Option<FileFacts>,
    ) -> Result<Option<Plan>, EngineError> {
        let size = stored.length;
        if origin.support != Support::Parsed {
            return Ok(None);
        }
        let reader = self.reads.reader()?;
        let progress = reader
            .input_progress(&input.id)?
            .ok_or_else(|| std::io::Error::other("a stored import has no revision"))?;
        if (progress.parsed_upto == size || progress.waiting) && progress.parsed {
            return Ok(None);
        }
        let chain = reader.stored_chain(&input.id, stored.chain, size)?;
        let source = if matches!(
            (origin.open)(path),
            Some(polylogue_sources::Parse::Export(_))
        ) {
            pool::Bytes::Stored(Arc::new(chain))
        } else {
            pool::Bytes::Import(Arc::new(Mutex::new(read::Imported::from_chain(
                chain,
                Arc::clone(&self.blobs),
            ))))
        };
        let mut plan = Plan::stored(
            path.to_path_buf(),
            origin,
            input,
            Change::Unchanged,
            facts,
            size,
            0,
            source,
            self.config.chunk,
            false,
        );
        // The operator is waiting on an import.
        plan.priority = Priority::Live;
        Ok(Some(plan))
    }
}

/// One immutable acquisition at one receiving location and by one author.
/// Retrying that acquisition keeps its namespace; another path, receiver,
/// member body or author establishes a different namespace.
fn import_identity<'a>(
    machine: &str,
    path: &Path,
    authorship: Option<&polylogue_model::authorship::Authorship>,
    members: impl Iterator<Item = (&'a Path, polylogue_model::BlobHash)>,
) -> String {
    let mut digest = blake3::Hasher::new();
    digest.update(b"polylogue-import-1\0");
    let mut field = |value: &[u8]| {
        digest.update(&(value.len() as u64).to_le_bytes());
        digest.update(value);
    };
    field(machine.as_bytes());
    field(path.as_os_str().as_encoded_bytes());
    field(&serde_json::to_vec(&authorship).expect("authorship encodes"));
    for (path, hash) in members {
        field(path.as_os_str().as_encoded_bytes());
        field(&hash.0);
    }
    format!("import:{}", digest.finalize())
}

/// Generation and length must still describe the descriptor that was captured.
fn unchanged(file: &std::fs::File, before: FileFacts, length: u64) -> std::io::Result<()> {
    if FileFacts::of(&file.metadata()?) != before || length != before.size {
        return Err(std::io::Error::other("source changed during import"));
    }
    Ok(())
}

/// A completed import: its bytes hashed and compressed into a private
/// file, with their first bytes for recognition.
struct Captured {
    prepared: polylogue_store::staging::PreparedBytes,
    #[cfg(test)]
    head: Vec<u8>,
    member: Option<polylogue_store::ImportMember>,
    container: Option<polylogue_store::InContainer>,
}

impl Captured {
    /// Bytes that can be read only once, inspected through their encoded preparation.
    #[cfg(test)]
    fn stream(archive: &Archive, reader: impl std::io::Read) -> polylogue_store::Result<Self> {
        let mut reader = CaptureReader {
            reader,
            size: 0,
            head: Vec::new(),
        };
        let prepared = archive.prepare_bytes(&mut reader)?;
        Ok(Self {
            prepared,
            head: reader.head,
            member: None,
            container: None,
        })
    }
}

#[cfg(test)]
struct CaptureReader<R> {
    reader: R,
    size: u64,
    head: Vec<u8>,
}

#[cfg(test)]
impl<R: std::io::Read> std::io::Read for CaptureReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.reader.read(buf)?;
        self.size += n as u64;
        let take = n.min(64 - self.head.len());
        self.head.extend_from_slice(&buf[..take]);
        Ok(n)
    }
}

/// The bytes a zip file starts with.
pub(super) const ZIP_MAGIC: &[u8] = b"PK\x03\x04";

/// The origin of export members no parser reads: kept as bytes.
pub(super) const KEPT: &str = "export-file";

/// `path` itself when it is a file, every file below it when it is a
/// directory, sorted so an import's order does not depend on the
/// filesystem. Symbolic links are not followed into directories.
pub(super) fn files_under(path: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let meta = std::fs::symlink_metadata(path)?;
    if !meta.is_dir() {
        out.push(path.to_path_buf());
        return Ok(());
    }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(path)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<_>>()?;
    entries.sort();
    for entry in entries {
        files_under(&entry, out)?;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests;
