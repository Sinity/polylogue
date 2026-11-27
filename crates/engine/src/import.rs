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
mod package;
pub(crate) mod python_polylogue;
mod recognition;

impl Engine {
    /// Imports immutable content identified by the BLAKE3 of its original
    /// bytes. The writer confirms a matching stored extent or commits the
    /// newly prepared bytes before they are parsed.
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
        if options.origin.as_deref() == Some(python_polylogue::VECTORS) {
            let [path] = paths else {
                return Err(polylogue_model::Reason::ImportOptions {
                    problem: format!(
                        "--format {} names one Python Polylogue vector store",
                        python_polylogue::VECTORS
                    ),
                }
                .into());
            };
            if options.dry_run {
                return Err(polylogue_model::Reason::ImportOptions {
                    problem: format!("--dry-run does not apply to {}", python_polylogue::VECTORS),
                }
                .into());
            }
            report.python_import = Some(self.import_python_vectors(path)?);
            report.pass.seconds = started.elapsed().as_secs_f64();
            return Ok(report);
        }
        for path in paths {
            let members = match package::stage(path, &self.archive, options.origin.as_deref()) {
                Ok(members) => members,
                Err(error) => {
                    recognition::refuse(&mut report, path, error.to_string());
                    continue;
                }
            };
            self.import_members(path, members, options, &mut report)?;
        }
        if options.dry_run {
            report.pass.seconds = started.elapsed().as_secs_f64();
            return Ok(report);
        }
        report.user_state.applied += report.pass.authored.applied;
        report.user_state.present += report.pass.authored.present;
        report.user_state.pending += report.pass.authored.pending;
        self.refresh_import_assets(&mut report.pass, None)?;
        let changed = !report.imported.is_empty()
            || report.pass.bytes_stored != 0
            || report.pass.inputs_seen != 0
            || report.pass.inputs_changed != 0
            || report.pass.bytes_read != 0
            || report.pass.messages_written != 0
            || report.user_state.applied != 0;
        if changed || self.archive.index_build().is_some() {
            report.pass.sessions_derived = self.derive()?;
        } else {
            report.pass.derivation_pending = self.writer.derive_in_background()?;
        }
        // With a current installed index the import was parsed into it;
        // with none, the rebuild in progress or the incremental update
        // after its install parses it.
        if !self.archive.index_files()?.serves()
            && let Some(build) = self.archive.index_build()
        {
            report.readable = Readable::AfterBuild { build };
        }
        report.position = self.writer.archive().committed_position();
        report.pass.seconds = started.elapsed().as_secs_f64();
        Ok(report)
    }

    fn import_members(
        &mut self,
        path: &Path,
        members: Vec<package::Member>,
        options: &ImportOptions,
        report: &mut ImportReport,
    ) -> Result<(), EngineError> {
        let mut package = ImportPackage {
            path: path.to_path_buf(),
            ..Default::default()
        };
        let mut ids = Vec::new();
        let mut authored_records = Vec::new();
        let mut items = Vec::new();
        let mut hashes = std::collections::BTreeMap::new();
        let classify_only = options.dry_run;
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
                let reason = member
                    .refusal
                    .unwrap_or_else(|| "no declared structure matched".into());
                recognition::refuse(report, &member.path, reason.clone());
                report.members.push(ImportClassification {
                    path: member.path,
                    origin: None,
                    class: "refused".into(),
                    reason: Some(reason),
                });
                continue;
            };
            classify_accepted(report, &mut package, &member.path, origin, member.container);
            if classify_only {
                continue;
            }
            let authored_bytes = (origin.name == polylogue_sources::polylogue_bundle::ORIGIN)
                .then(|| member.bytes.clone());
            let mut captured = Captured {
                prepared: member.bytes,
                member: None,
                container: None,
                #[cfg(test)]
                head: Vec::new(),
            };
            ids.push(captured.prepared.hash().to_string());
            hashes.insert(index, captured.prepared.hash().to_string());
            captured.member = Some(import_member(
                path,
                &member.path,
                member.parent.is_some(),
                import_id.clone(),
                options,
            ));
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
            if origin.name == polylogue_sources::polylogue_bundle::ORIGIN {
                authored_records.push((
                    member.path.clone(),
                    captured.prepared.hash().to_string(),
                    authored_bytes.expect("bundle reader"),
                ));
            }
            items.push((member.path, origin, None, captured));
        }
        if !classify_only {
            let executed = self.store_import_group(items, report)?;
            for (at, id, bytes) in authored_records {
                if executed.contains(&id) {
                    continue;
                }
                if self.reads.reader()?.latest_chain(&id)?.is_some() {
                    self.restore_bundle_authored(&at, &mut bytes.reader()?, report)?;
                }
            }
            let counts = self.reads.reader()?.import_counts(&ids)?;
            package.sessions = counts.sessions;
            package.documents = counts.documents;
        }
        report.packages.push(package);
        Ok(())
    }

    fn store_import_group(
        &mut self,
        items: Vec<(PathBuf, &'static OriginSpec, Option<FileFacts>, Captured)>,
        report: &mut ImportReport,
    ) -> Result<BTreeSet<String>, EngineError> {
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
            batch.new_blob = Some(polylogue_store::NewBlob::whole(
                captured.prepared.hash(),
                captured.prepared.len(),
            ));
            metadata.push((at, origin, facts, batch.input.clone()));
            group.push(polylogue_store::staging::PreparedBatch {
                batch,
                files: vec![captured.prepared],
            });
        }
        if group.is_empty() {
            return Ok(BTreeSet::new());
        }
        let outcomes = self.writer.store_group(group)?;
        self.finish_import_group(metadata, outcomes, report)
    }

    fn finish_import_group(
        &mut self,
        metadata: Vec<(PathBuf, &'static OriginSpec, Option<FileFacts>, InputKey)>,
        outcomes: Vec<polylogue_store::Result<polylogue_store::staging::StoredInput>>,
        report: &mut ImportReport,
    ) -> Result<BTreeSet<String>, EngineError> {
        let mut executed = BTreeSet::new();
        // Retain the raw group before feeding independent line streams (any
        // input its origin reads line by line) to the pool. Repeated inputs
        // or paths need the preceding parse checkpoint. Other formats keep
        // their authored and document execution order.
        const LINE_PLANS: usize = 128;
        let mut lines: Vec<Plan> = Vec::with_capacity(LINE_PLANS);
        for ((at, origin, facts, input), outcome) in metadata.into_iter().zip(outcomes) {
            let batch_lines =
                matches!((origin.open)(&at), Some(polylogue_sources::Parse::Lines(_)));
            if !batch_lines
                || outcome.is_err()
                || lines.len() == LINE_PLANS
                || lines
                    .iter()
                    .any(|plan| plan.input.id == input.id || plan.path == at)
            {
                self.execute(std::mem::take(&mut lines), &mut report.pass)?;
            }
            let stored = match outcome {
                Ok(stored) => stored,
                Err(error @ polylogue_store::StoreError::ScopeChanged) => return Err(error.into()),
                Err(error) => {
                    report
                        .pass
                        .errors
                        .push(polylogue_model::Failure::new(at.display(), &error));
                    continue;
                }
            };
            if stored.already_stored {
                report.already_stored.push(at.clone());
            } else {
                report.imported.push(at.clone());
                report.pass.bytes_stored += stored.length;
            }
            let plan = self.stored_import_plan(&at, origin, input, &stored, facts);
            if plan.is_err() {
                self.execute(std::mem::take(&mut lines), &mut report.pass)?;
            }
            if let Some(plan) = plan? {
                executed.insert(plan.input.id.clone());
                if batch_lines {
                    lines.push(plan);
                } else {
                    self.execute(vec![plan], &mut report.pass)?;
                }
            }
        }
        self.execute(lines, &mut report.pass)?;
        Ok(executed)
    }

    pub(crate) fn restore_captured_authored(
        &mut self,
        plan: &crate::plan::Plan,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        if plan.origin.name != polylogue_sources::polylogue_bundle::ORIGIN {
            return Ok(());
        }
        let stored = self.reads.reader().and_then(|reader| {
            let chain = match plan.change {
                Change::Reparse { chain } => chain,
                _ => {
                    reader
                        .latest_chain(&plan.input.id)?
                        .ok_or_else(|| {
                            polylogue_store::StoreError::internal(
                                "a stored bundle has no committed source revision",
                            )
                        })?
                        .chain
                }
            };
            reader.stored_chain(&plan.input.id, chain, plan.size)
        })?;
        polylogue_store::fault::check("import.before_authored_restore", &plan.input.id)?;
        // The replay carries errors as text: a failed write is kept typed
        // beside it, so only the bundle's own bytes count against its chain.
        let mut restore_failed = None;
        let replayed = authored::replay(stored.reader(Arc::clone(&self.blobs)), |entry| {
            let outcome = entry.restore(&self.writer).map_err(|error| {
                let text = error.to_string();
                restore_failed = Some(error);
                text
            })?;
            if outcome.pending {
                pass.authored.pending += 1;
            } else if outcome.repeat {
                pass.authored.present += 1;
            } else {
                pass.authored.applied += 1;
            }
            Ok(())
        });
        if let Some(error) = restore_failed {
            return Err(error.into());
        }
        replayed.map_err(polylogue_model::ChainFailure::malformed)?;
        polylogue_store::fault::check("import.after_authored_restore", &plan.input.id)?;
        Ok(())
    }

    fn restore_bundle_authored(
        &mut self,
        path: &Path,
        file: &mut impl std::io::Read,
        report: &mut ImportReport,
    ) -> Result<(), EngineError> {
        let mut restore_failed = None;
        let result = authored::replay(file, |entry| {
            let outcome = entry.restore(&self.writer).map_err(|error| {
                let text = error.to_string();
                restore_failed = Some(error);
                text
            })?;
            if outcome.pending {
                report.user_state.pending += 1;
            } else if outcome.repeat {
                report.user_state.present += 1;
            } else {
                report.user_state.applied += 1;
            }
            Ok(())
        });
        if let Some(error) = restore_failed {
            report.pass.errors.push(
                polylogue_model::Failure::new(path.display(), &error).doing("authored restoration"),
            );
        } else if let Err(error) = result {
            report.pass.errors.push(polylogue_model::Failure::stated(
                path.display(),
                polylogue_model::Cause::Content,
                polylogue_model::Code::Import,
                format!("authored restoration: {error}"),
            ));
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

    /// Stores the complete import once, then plans its reading from stored bytes.
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
                report
                    .pass
                    .errors
                    .push(polylogue_model::Failure::new(path.display(), &error));
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
            input: InputKey {
                id: hash.to_string(),
                format: origin.name.into(),
                route: origin.route.imported(),
            },
            change: Change::First,
            observation: Some(observation),
            continues: false,
            new_blob: None,
            blobs: Vec::new(),
            length: size,
            parse: Vec::new(),
            capture_read: Some(
                self.source_read_context(
                    path,
                    origin.name,
                    polylogue_store::capture_read::ReadRoute::ExplicitImport,
                )
                .finish(
                    polylogue_store::capture_read::RawRange {
                        start: 0,
                        end: size,
                    },
                    hash,
                ),
            ),
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
        let version = origin.parse_version();
        let state = reader.chain_state(&input.id, stored.chain)?;
        let due = polylogue_model::chain::due(
            &state,
            &polylogue_model::chain::ChainFacts {
                stored: size,
                repeated_by: None,
                own_from: 0,
                parser: Some(version),
                finished: true,
            },
        );
        if due == polylogue_model::chain::Due::No {
            return Ok(None);
        }
        let parser = (origin.open)(path);
        // A line import interrupted at a record boundary goes on from there.
        let resume_from = match due {
            polylogue_model::chain::Due::Extend { from }
                if matches!(parser.as_ref(), Some(polylogue_sources::Parse::Lines(_)))
                    && (1..size).contains(&from) =>
            {
                from
            }
            _ => 0,
        };
        let interrupted_lines = resume_from > 0;
        let source = if interrupted_lines {
            // The ordinary stored-input loader reopens this checkpoint and
            // starts range reads at its confirmed cursor.
            pool::Bytes::Unloaded {
                chain: stored.chain,
                length: size,
                decoded: false,
            }
        } else {
            let chain = reader.stored_chain(&input.id, stored.chain, size)?;
            if matches!(parser, Some(polylogue_sources::Parse::Export(_))) {
                pool::Bytes::Stored(Arc::new(chain))
            } else {
                pool::Bytes::Import(Arc::new(Mutex::new(read::Imported::from_chain(
                    chain,
                    Arc::clone(&self.blobs),
                ))))
            }
        };
        let mut plan = Plan::stored(
            path.to_path_buf(),
            origin,
            input,
            Change::Unchanged,
            facts,
            size,
            resume_from,
            source,
            self.config.chunk,
            false,
        );
        plan.parser_chain = Some(stored.chain);
        plan.input_finished = true;
        // The operator is waiting on an import.
        plan.priority = Priority::Live;
        Ok(Some(plan))
    }
}

/// A member's name as text. Bytes that are not UTF-8 are written as `%XX`
/// for every byte, as a nested member's are, so distinct names stay distinct.
fn member_name(name: &std::ffi::OsStr) -> String {
    let bytes = name.as_encoded_bytes();
    match std::str::from_utf8(bytes) {
        Ok(text) => text.to_owned(),
        Err(_) => bytes.iter().map(|byte| format!("%{byte:02X}")).collect(),
    }
}

fn import_member(
    path: &Path,
    at: &Path,
    nested: bool,
    import_id: String,
    options: &ImportOptions,
) -> polylogue_store::ImportMember {
    polylogue_store::ImportMember {
        version: 1,
        import_id,
        authorship: Some(
            options
                .authorship
                .clone()
                .unwrap_or_else(default_import_authorship),
        ),
        member: if nested {
            at.to_string_lossy()
                .rsplit("!/")
                .next()
                .expect("member name")
                .to_owned()
        } else if at == path {
            member_name(at.file_name().unwrap_or_default())
        } else {
            member_name(at.strip_prefix(path).unwrap_or(at).as_os_str())
        },
        format: None,
    }
}

fn default_import_authorship() -> polylogue_model::authorship::Authorship {
    polylogue_model::authorship::Authorship {
        author: polylogue_model::Author::Script("polylogue-import".into()),
        via: polylogue_model::authorship::Channel::Import,
    }
}

fn classify_accepted(
    report: &mut ImportReport,
    package: &mut ImportPackage,
    at: &Path,
    origin: &OriginSpec,
    container: bool,
) {
    report.members.push(ImportClassification {
        path: at.to_path_buf(),
        origin: Some(origin.name.into()),
        class: match origin.support {
            Support::Parsed => "parsed",
            Support::NoParser => "no parser yet",
            Support::Kept => "asset",
        }
        .into(),
        reason: None,
    });
    if origin.support == Support::Kept && !container {
        package.assets += 1;
    }
}

/// One immutable read at one receiving location and by one author.
/// Retrying that read keeps its namespace; another path, receiver,
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

/// File facts and length must still describe the descriptor that was captured.
fn unchanged(file: &std::fs::File, before: FileFacts, length: u64) -> std::io::Result<()> {
    if FileFacts::of(&file.metadata()?) != before || length != before.size {
        return Err(std::io::Error::other(
            polylogue_store::StoreError::SourceChanged("source changed during import"),
        ));
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
        // A link to a directory, or a broken link, has no bytes to stage.
        if !meta.file_type().is_symlink() || std::fs::metadata(path).is_ok_and(|m| m.is_file()) {
            out.push(path.to_path_buf());
        }
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
