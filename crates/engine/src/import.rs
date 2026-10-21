//! Importing exports and other files by hand (architecture 5.1): a file, a
//! directory of them, or a zip whose members are recognized one by one.
//! An imported file's origin is decided by its content and, when several
//! origins accept it, by the layout its path sits in.

use super::*;

mod exports;

impl Engine {
    /// Imports immutable content identified by the BLAKE3 of its original
    /// bytes. Acquisition streams into the blob store before the writer
    /// records a complete revision; parsing reads ranges of that same blob.
    /// Document-framed origins still parse a whole document in one job.
    pub fn import(&mut self, paths: &[PathBuf]) -> Result<ImportReport, EngineError> {
        use std::io::{Read, Seek, SeekFrom};
        let started = Instant::now();
        let mut report = ImportReport::default();
        let mut files = Vec::new();
        for path in paths {
            files_under(path, &mut files)?;
        }
        for path in &files {
            let mut file = std::fs::File::open(path)?;
            let facts = FileFacts::of(&file.metadata()?);
            let mut head = [0u8; 64];
            let n = file.read(&mut head)?;
            let head = &head[..n];
            let whole = recognize_import(path, head);
            file.seek(SeekFrom::Start(0))?;
            if whole.is_none() && head.starts_with(ZIP_MAGIC) {
                let generation = file.try_clone()?;
                let mut archive = zip::ZipArchive::new(file)
                    .map_err(|e| std::io::Error::other(format!("{}: {e}", path.display())))?;
                for i in 0..archive.len() {
                    let mut member = archive
                        .by_index(i)
                        .map_err(|e| std::io::Error::other(format!("{}: {e}", path.display())))?;
                    if !member.is_file() {
                        continue;
                    }
                    let at = PathBuf::from(format!("{}!/{}", path.display(), member.name()));
                    let declared = member.size();
                    let captured = Captured::read(&self.blobs, &mut member)?;
                    unchanged(&generation, facts, facts.size)?;
                    if captured.piece.size != declared {
                        return Err(std::io::Error::other(
                            "ZIP member length changed during import",
                        )
                        .into());
                    }
                    let origin = recognize_member(&at, &captured.head);
                    self.import_captured(&at, origin, captured, None, &mut report)?;
                }
            } else if whole.is_some() {
                let captured = Captured::read(&self.blobs, &mut file)?;
                unchanged(&file, facts, captured.piece.size)?;
                // Recognition uses the captured prefix, so a concurrent rewrite
                // cannot pair one version's identity with another's parser.
                if let Some(origin) = recognize_import(path, &captured.head) {
                    self.import_captured(path, origin, captured, Some(facts), &mut report)?;
                } else {
                    report.unrecognized.push(path.clone());
                }
            } else {
                report.unrecognized.push(path.clone());
            }
        }
        report.pass.sessions_derived = self.writer.derive()?;
        report.pass.seconds = started.elapsed().as_secs_f64();
        Ok(report)
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
        let piece = captured.piece;
        let id = piece.hash.to_string();
        let retained = self.reads.reader()?.retained_input(&id)?;
        let complete = retained.as_ref().is_some_and(|r| r.length == piece.size);
        if complete {
            report.already_stored.push(path.to_path_buf());
            if origin.support != Support::Parsed
                || retained.as_ref().is_some_and(|r| {
                    (r.parsed_upto == piece.size || r.waiting)
                        && r.parser_version == Some(origin.parser_version)
                })
            {
                return Ok(None);
            }
        } else {
            report.imported.push(path.to_path_buf());
        }
        let input = InputKey {
            id,
            format: origin.name.into(),
            route_rank: 2,
        };
        if !complete {
            self.writer.send(polylogue_store::Batch {
                input: input.clone(),
                change: if retained.is_some() {
                    Change::Rewritten
                } else {
                    Change::First
                },
                observation: Some(self.observation(path, facts)),
                continues: false,
                piece: Some(piece.clone()),
                blobs: Vec::new(),
                length: piece.size,
                parse: Vec::new(),
                release: Default::default(),
            })?;
            let written = self.writer.flush()?;
            if !written.errors.is_empty() {
                report.pass.errors.extend(written.errors);
                return Ok(None);
            }
            report.pass.bytes_stored += piece.size;
        }
        Ok(Some(Plan {
            path: path.to_path_buf(),
            origin,
            input,
            change: Change::Unchanged,
            observation: None,
            facts,
            size: piece.size,
            piece_pos: piece.size,
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
            source: pool::Bytes::Import(Arc::new(Mutex::new(read::Imported::open(
                &self.blobs,
                piece.hash,
                piece.size,
            )?))),
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
    piece: polylogue_store::Piece,
    head: Vec<u8>,
}

impl Captured {
    fn read(store: &hashed_blobs::Store, reader: impl std::io::Read) -> Result<Self, EngineError> {
        let mut reader = CaptureReader {
            reader,
            size: 0,
            head: Vec::new(),
        };
        let hash = store.put_plain_checked::<hashed_blobs::Error>(
            &mut reader,
            polylogue_store::blobs::LEVEL,
            || Ok(()),
        )?;
        Ok(Self {
            piece: polylogue_store::Piece {
                hash: polylogue_model::BlobHash(hash.0),
                size: reader.size,
                stored: polylogue_store::blobs::Stored::File {
                    stored_size: store.stored_size(&hash)?,
                },
            },
            head: reader.head,
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

/// The origin that reads a zip member, or the one that keeps it as bytes.
pub(super) fn recognize_member(at: &Path, bytes: &[u8]) -> &'static OriginSpec {
    let head = &bytes[..bytes.len().min(64)];
    recognize_import(at, head)
        .unwrap_or_else(|| polylogue_sources::origin::by_name(KEPT).expect("declared origin"))
}

/// The origin an imported file belongs to. Several recognizers can accept
/// one file (Claude Code takes any JSON-lines transcript, since an exported
/// transcript can sit anywhere); the one whose own layout the path ends in
/// is preferred, so a Grok CLI history or a hook carrier copied with its
/// directories keeps its origin.
pub(super) fn recognize_import(path: &Path, head: &[u8]) -> Option<&'static OriginSpec> {
    let mut found = ORIGINS
        .iter()
        .filter(|o| o.support == Support::Parsed && (o.recognize)(path, head));
    let first = found.next()?;
    let in_layout = |o: &OriginSpec| o.locations.iter().any(|l| collect::ends_in(path, l));
    if in_layout(first) {
        return Some(first);
    }
    Some(found.find(|o| in_layout(o)).unwrap_or(first))
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
