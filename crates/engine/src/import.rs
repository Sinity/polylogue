//! Importing exports and other files by hand (architecture 5.1): a file, a
//! directory of them, or a zip whose members are recognized one by one.
//! An imported file's origin is decided by its content and, when several
//! origins accept it, by the layout its path sits in.

use super::*;
use polylogue_sources::origin::Route;

mod assets;
mod exports;
mod replay;
mod zip_import;

impl Engine {
    /// Imports immutable content identified by the BLAKE3 of its original
    /// bytes. Acquisition prepares private bytes; the writer durably reserves
    /// their ownership before installation, then admits them for parsing.
    /// Document-framed origins still parse a whole document in one job.
    pub fn import(&mut self, paths: &[PathBuf]) -> Result<ImportReport, EngineError> {
        use std::io::{Read, Seek, SeekFrom};
        let started = Instant::now();
        let mut report = ImportReport::default();
        let reconciled = self
            .writer
            .reconcile_retained_bytes(Arc::clone(&self.blobs))?;
        report.pass.unfinished_inputs = reconciled.unfinished;
        if reconciled.completed > 0 {
            self.replay_imports(None, &mut report.pass)?;
        }
        let mut files = Vec::new();
        for path in paths {
            let mut selected = Vec::new();
            files_under(path, &mut selected)?;
            let acquisition = uuid::Uuid::now_v7().to_string();
            let root = if path.is_dir() {
                path.as_path()
            } else {
                path.parent().unwrap_or(Path::new(""))
            };
            for selected in selected {
                let member = selected
                    .strip_prefix(root)
                    .map_err(std::io::Error::other)?
                    .to_string_lossy()
                    .into_owned();
                files.push((
                    selected,
                    polylogue_store::ImportMember {
                        version: 1,
                        acquisition: acquisition.clone(),
                        member,
                        format: None,
                        container: None,
                    },
                ));
            }
        }
        for (path, membership) in &files {
            let mut file = std::fs::File::open(path)?;
            let facts = FileFacts::of(&file.metadata()?);
            let mut head = [0u8; 64];
            let n = file.read(&mut head)?;
            let head = &head[..n];
            file.seek(SeekFrom::Start(0))?;
            let whole = if polylogue_sources::polylogue_export::recognize_reader(&mut file)? {
                polylogue_sources::origin::by_name("polylogue-export")
            } else {
                recognize_import(path, head)
            };
            file.seek(SeekFrom::Start(0))?;
            if whole.is_none() && head.starts_with(ZIP_MAGIC) {
                self.import_zip(path, file, facts, membership, &mut report)?;
            } else {
                let mut captured =
                    Captured::read(&self.archive, &self.config.blob_roots, &mut file)?;
                captured.member = Some(membership.clone());
                unchanged(&file, facts, captured.prepared.len())?;
                let origin = self
                    .recognize_captured(path, &captured)?
                    .unwrap_or_else(|| {
                        polylogue_sources::origin::by_name(KEPT).expect("declared origin")
                    });
                self.import_captured(path, origin, captured, Some(facts), &mut report)?;
            }
        }
        self.restore_failed_writes()?;
        for pending in self.reads.reader()?.status()?.unfinished_inputs {
            if !report
                .pass
                .unfinished_inputs
                .iter()
                .any(|known| known.input == pending.input && known.blob == pending.blob)
            {
                report.pass.unfinished_inputs.push(pending);
            }
        }
        self.refresh_import_assets(&mut report.pass)?;
        report.pass.sessions_derived = self.writer.derive()?;
        report.pass.seconds = started.elapsed().as_secs_f64();
        Ok(report)
    }

    fn recognize_captured(
        &self,
        path: &Path,
        captured: &Captured,
    ) -> Result<Option<&'static OriginSpec>, EngineError> {
        if captured.polylogue_export {
            return Ok(polylogue_sources::origin::by_name("polylogue-export"));
        }
        Ok(recognize_import(path, &captured.head))
    }

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

    /// Retains the complete acquisition once, then parses it without making
    /// further pieces. A retry can rebuild parser state from offset zero.
    fn import_plan(
        &mut self,
        path: &Path,
        origin: &'static OriginSpec,
        captured: Captured,
        facts: Option<FileFacts>,
        report: &mut ImportReport,
    ) -> Result<Option<Plan>, EngineError> {
        if origin.support != Support::Parsed {
            report.unrecognized.push(path.to_path_buf());
        }
        let size = captured.prepared.len();
        let id = captured.prepared.hash().to_string();
        let batch = self.import_batch(path, origin, captured.member, facts, id, size);
        let input = batch.input.clone();
        let admitted = match self.writer.retain_bytes(captured.prepared, batch) {
            Ok(admitted) => admitted,
            Err(error) => {
                report.pass.errors.push(error.to_string());
                return Ok(None);
            }
        };
        if admitted.already_stored {
            report.already_stored.push(path.to_path_buf());
        } else {
            report.imported.push(path.to_path_buf());
            report.pass.bytes_stored += size;
        }
        self.admitted_import_plan(path, origin, input, &admitted, facts)
    }

    fn import_batch(
        &self,
        path: &Path,
        origin: &'static OriginSpec,
        member: Option<polylogue_store::ImportMember>,
        facts: Option<FileFacts>,
        id: String,
        size: u64,
    ) -> polylogue_store::Batch {
        let mut observation = self.observation(path, facts);
        observation.import = member.map(|mut member| {
            member.format = Some(origin.name.into());
            member
        });
        polylogue_store::Batch {
            input: InputKey {
                id,
                format: origin.name.into(),
                route_rank: 2,
            },
            change: Change::First,
            observation: Some(observation),
            continues: false,
            piece: None,
            blobs: Vec::new(),
            length: size,
            parse: Vec::new(),
            release: Default::default(),
        }
    }

    fn admitted_import_plan(
        &self,
        path: &Path,
        origin: &'static OriginSpec,
        input: InputKey,
        admitted: &polylogue_store::retention::Retained,
        facts: Option<FileFacts>,
    ) -> Result<Option<Plan>, EngineError> {
        let size = admitted.length;
        if origin.support != Support::Parsed {
            return Ok(None);
        }
        let reader = self.reads.reader()?;
        let retained = reader
            .retained_input(&input.id)?
            .ok_or_else(|| std::io::Error::other("admitted import has no retained revision"))?;
        if (retained.parsed_upto == size || retained.waiting)
            && retained.parser_version == Some(origin.parser_version)
        {
            return Ok(None);
        }
        let chain = reader.stored_chain(&input.id, admitted.chain, size)?;
        let source = if matches!(
            (origin.open)(path),
            Some(polylogue_sources::Parse::Export(_))
        ) {
            pool::Bytes::Stored(Arc::new(chain))
        } else {
            pool::Bytes::Import(Arc::new(Mutex::new(read::Imported::from_chain(
                chain,
                Arc::clone(&self.blobs),
            )?)))
        };
        Ok(Some(Plan {
            path: path.to_path_buf(),
            origin,
            input,
            change: Change::Unchanged,
            observation: None,
            facts,
            size,
            piece_pos: size,
            parse_pos: 0,
            piece_started: true,
            carry: None,
            tail: None,
            span: span(origin, path, self.config.chunk),
            chunk: self.config.chunk,
            priority: Priority::Live,
            tracked: false,
            failed: false,
            waiting: false,
            source,
        }))
    }
}

/// Generation and length must still describe the descriptor that was captured.
fn unchanged(file: &std::fs::File, before: FileFacts, length: u64) -> std::io::Result<()> {
    if FileFacts::of(&file.metadata()?) != before || length != before.size {
        return Err(std::io::Error::other("source changed during import"));
    }
    Ok(())
}

/// A completed acquisition. The blob store hashes and compresses bounded
/// frames while this reader counts original bytes and remembers the prefix.
struct Captured {
    prepared: polylogue_store::retention::PreparedBytes,
    head: Vec<u8>,
    polylogue_export: bool,
    member: Option<polylogue_store::ImportMember>,
}

impl Captured {
    fn read(
        archive: &Archive,
        roots: &[PathBuf],
        reader: impl std::io::Read,
    ) -> polylogue_store::Result<Self> {
        let mut reader = CaptureReader {
            reader,
            size: 0,
            head: Vec::new(),
        };
        let (prepared, polylogue_export) =
            archive.inspect_and_prepare_bytes(roots, &mut reader, |scratch| {
                Ok(polylogue_sources::polylogue_export::recognize_reader(
                    scratch,
                )?)
            })?;
        Ok(Self {
            prepared,
            head: reader.head,
            polylogue_export,
            member: None,
        })
    }
}

struct CaptureReader<R> {
    reader: R,
    size: u64,
    head: Vec<u8>,
}

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

/// The origin an imported file belongs to. Several recognizers can accept
/// one file (Claude Code takes any JSON-lines transcript, since an exported
/// transcript can sit anywhere); the one whose own layout the path ends in
/// is preferred, so a Grok CLI history or a hook carrier copied with its
/// directories keeps its origin.
pub(super) fn recognize_import(path: &Path, head: &[u8]) -> Option<&'static OriginSpec> {
    let found = ORIGINS
        .iter()
        .filter(|o| o.support == Support::Parsed && (o.recognize)(path, head));
    let first = found.clone().next()?;
    // Provider layout wins; outside it, an export's content signature wins
    // over a live parser that accepts arbitrary JSONL filenames.
    found
        .clone()
        .find(|o| o.locations.iter().any(|l| collect::ends_in(path, l)))
        .or_else(|| found.clone().find(|o| o.route == Route::Import))
        .or_else(|| {
            found.clone().find(|o| {
                matches!(
                    (o.open)(path),
                    Some(polylogue_sources::parse::Parse::Export(_))
                )
            })
        })
        .or(Some(first))
}

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
mod tests;
