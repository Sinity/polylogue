//! Importing exports and other files by hand (architecture 5.1): a file, a
//! directory of them, or a zip whose members are recognized one by one.
//! An imported file's origin is decided by its content and, when several
//! origins accept it, by the layout its path sits in.

use super::*;

impl Engine {
    /// Brings in export files by explicit request (architecture 5.1). An
    /// imported file is an input named by the BLAKE3 of its bytes, so
    /// importing the same file again stores nothing. A zip the origins do
    /// not read whole is opened and each member imported on its own: the
    /// ones an origin recognizes are parsed, the rest (images, audio, the
    /// export's own HTML) are kept as bytes. Members are read one at a time
    /// and imported in groups of about [`IMPORT_GROUP`] bytes, so an export
    /// larger than memory still imports.
    pub fn import(&mut self, paths: &[PathBuf]) -> Result<ImportReport, EngineError> {
        let started = Instant::now();
        let mut report = ImportReport::default();
        let mut group = Vec::new();
        let mut group_bytes = 0u64;
        // A directory (an old spool, an unpacked export) imports every file
        // under it, in path order.
        let mut files = Vec::new();
        for path in paths {
            files_under(path, &mut files)?;
        }
        for path in &files {
            let mut head = [0u8; 64];
            let n = std::io::Read::read(&mut std::fs::File::open(path)?, &mut head)?;
            let head = &head[..n];
            let whole = recognize_import(path, head);
            if whole.is_none() && head.starts_with(ZIP_MAGIC) {
                let mut archive = zip::ZipArchive::new(std::fs::File::open(path)?)
                    .map_err(|e| std::io::Error::other(format!("{}: {e}", path.display())))?;
                for i in 0..archive.len() {
                    let mut member = archive
                        .by_index(i)
                        .map_err(|e| std::io::Error::other(format!("{}: {e}", path.display())))?;
                    if !member.is_file() {
                        continue;
                    }
                    let name = member.name().to_owned();
                    let mut bytes = Vec::with_capacity(member.size() as usize);
                    std::io::Read::read_to_end(&mut member, &mut bytes)?;
                    let at = PathBuf::from(format!("{}!/{name}", path.display()));
                    let origin = recognize_member(&at, &bytes);
                    if let Some(plan) =
                        self.import_plan(&at, origin, Arc::new(bytes), &mut report)?
                    {
                        group_bytes += plan.size;
                        group.push(plan);
                    }
                    if group_bytes >= IMPORT_GROUP {
                        self.execute(std::mem::take(&mut group), &mut report.pass)?;
                        group_bytes = 0;
                    }
                }
                continue;
            }
            let Some(origin) = whole else {
                report.unrecognized.push(path.clone());
                continue;
            };
            let bytes = Arc::new(std::fs::read(path)?);
            if let Some(plan) = self.import_plan(path, origin, bytes, &mut report)? {
                group_bytes += plan.size;
                group.push(plan);
            }
            if group_bytes >= IMPORT_GROUP {
                self.execute(std::mem::take(&mut group), &mut report.pass)?;
                group_bytes = 0;
            }
        }
        self.execute(group, &mut report.pass)?;
        report.pass.sessions_derived = self.writer.derive()?;
        report.pass.seconds = started.elapsed().as_secs_f64();
        Ok(report)
    }

    /// One imported input's plan, or `None` when its exact bytes are
    /// already stored.
    pub(super) fn import_plan(
        &mut self,
        path: &Path,
        origin: &'static OriginSpec,
        bytes: Arc<Vec<u8>>,
        report: &mut ImportReport,
    ) -> Result<Option<Plan>, EngineError> {
        let id = blake3::hash(&bytes).to_hex().to_string();
        if self.reads.reader()?.has_input(&id)? {
            report.already_stored.push(path.to_path_buf());
            return Ok(None);
        }
        let facts = std::fs::metadata(path).ok().map(|m| FileFacts::of(&m));
        report.imported.push(path.to_path_buf());
        Ok(Some(Plan {
            path: path.to_path_buf(),
            origin,
            input: InputKey {
                id,
                format: origin.name.into(),
                route_rank: 2,
            },
            change: Change::First,
            observation: Some(self.observation(path, facts)),
            facts,
            size: bytes.len() as u64,
            piece_pos: 0,
            parse_pos: 0,
            piece_started: false,
            carry: None,
            tail: None,
            span: span(origin, self.config.chunk),
            chunk: self.config.chunk,
            priority: Priority::Live,
            tracked: false,
            failed: false,
            waiting: false,
            source: pool::Bytes::Memory(bytes),
        }))
    }
}

/// The bytes a zip file starts with.
pub(super) const ZIP_MAGIC: &[u8] = b"PK\x03\x04";

/// Bytes of imported inputs held in memory and run through the pool at
/// once. A member larger than this runs alone.
pub(super) const IMPORT_GROUP: u64 = 1 << 30;

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
