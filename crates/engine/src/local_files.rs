//! Local-image references in session files are read when observed. Recorded directories
//! alone resolve relative paths; coverage facts survive with the captured bytes.
use crate::*;
use polylogue_model::{Json, JsonExt};
use std::io::{BufRead, Seek, Write};
use std::os::unix::fs::OpenOptionsExt;

fn visit(value: &Json, emit: &mut impl FnMut(&str) -> std::io::Result<()>) -> std::io::Result<()> {
    match value {
        Json::Object(o) => {
            if let Some(images) = value.array("local_images") {
                for image in images {
                    // A typed entry is emitted once by the recursion below.
                    let path = image.as_str().or_else(|| {
                        image
                            .str("path")
                            .filter(|_| image.str("type") != Some("local_image"))
                    });
                    if let Some(path) = path {
                        emit(path)?;
                    }
                }
            }
            if value.str("type") == Some("local_image")
                && let Some(path) = value.str("path")
            {
                emit(path)?;
            }
            if matches!(value.str("type"), Some("input_image" | "image"))
                && let Some(path) = o
                    .get("image_url")
                    .or_else(|| o.get("url"))
                    .and_then(Json::as_str)
                    .filter(|p| p.starts_with("file://"))
            {
                emit(path)?;
            }
            for value in o.values() {
                visit(value, emit)?;
            }
        }
        Json::Array(values) => {
            for value in values {
                visit(value, emit)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Whether a record refers to a local file by a relative path.
fn has_relative_reference(value: &Json) -> bool {
    let mut relative = false;
    let _ = visit(value, &mut |raw| {
        relative |= !Path::new(raw.strip_prefix("file://").unwrap_or(raw)).is_absolute();
        Ok(())
    });
    relative
}

/// The working directory the last record before `start` recorded: read
/// backwards from `start` in windows, whole lines only.
fn directory_before(
    stored: &polylogue_store::blobs::StoredChain,
    blobs: &polylogue_store::blobs::Blobs,
    start: u64,
) -> Result<Option<PathBuf>, EngineError> {
    const WINDOW: u64 = 1 << 16;
    let mut end = start;
    let mut carry = Vec::new();
    while end > 0 {
        let from = end.saturating_sub(WINDOW);
        let mut bytes = stored.read_range(blobs, from, end)?;
        bytes.extend_from_slice(&carry);
        // The first piece may be the end of a line that began earlier.
        let first_line_end = if from == 0 {
            0
        } else {
            match bytes.iter().position(|b| *b == b'\n') {
                Some(at) => at + 1,
                None => {
                    carry = bytes;
                    end = from;
                    continue;
                }
            }
        };
        for line in bytes[first_line_end..].split(|b| *b == b'\n').rev() {
            if let Ok(value) = serde_json::from_slice::<Json>(line)
                && let Some(cwd) = polylogue_sources::codex::recorded_directory(&value)
            {
                return Ok(Some(PathBuf::from(cwd)));
            }
        }
        carry = bytes[..first_line_end].to_vec();
        end = from;
    }
    Ok(None)
}

impl Engine {
    /// Source positions cannot advance before attachment observations. This
    /// plan path also resumes a capture interrupted after byte commit.
    pub(crate) fn observe_local_references(
        &mut self,
        plan: &crate::plan::Plan,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        // A loop without sources reads no file outside the archive, a
        // referenced image included.
        if !self.config.reads_sources
            || plan.origin.name != "codex"
            || plan.input.route.is_imported()
        {
            return Ok(());
        }
        let Change::Reparse { chain } = plan.change else {
            return Ok(());
        };
        let stored = self
            .reads
            .read_current(|reader| reader.stored_chain(&plan.input.id, chain, plan.size))?;
        // Only the records this reading parses are looked at: an append's
        // records, or every record when the reading starts again.
        let start = plan.parse_pos.min(plan.size);
        let blobs = Arc::clone(&self.blobs);
        let mut before = || directory_before(&stored, &blobs, start);
        if start == 0 {
            return self.capture_local_files(
                std::io::BufReader::new(stored.reader(Arc::clone(&self.blobs))),
                0,
                &mut before,
                &plan.input.id,
                pass,
            );
        }
        let appended = stored.read_range(&self.blobs, start, plan.size)?;
        self.capture_local_files(
            std::io::Cursor::new(appended),
            start,
            &mut before,
            &plan.input.id,
            pass,
        )
    }

    pub(crate) fn capture_local_files(
        &mut self,
        mut source: impl BufRead,
        start: u64,
        directory_before: &mut dyn FnMut() -> Result<Option<PathBuf>, EngineError>,
        transcript: &str,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        let mut manifest = tempfile::tempfile_in(self.archive.staging_dir())?;
        let mut directory: Option<PathBuf> = None;
        // The directory recorded before `start`, looked up only when a
        // relative reference needs it.
        let mut looked_before = start == 0;
        let mut line = Vec::new();
        let mut offset = start;
        let mut references = 0u64;
        while source.read_until(b'\n', &mut line)? > 0 {
            if let Ok(value) = serde_json::from_slice::<Json>(&line) {
                if let Some(cwd) = polylogue_sources::codex::recorded_directory(&value) {
                    // A relative recorded cwd is insufficient to resolve a ref.
                    directory = Some(PathBuf::from(cwd));
                }
                let relative = has_relative_reference(&value);
                if relative && directory.is_none() && !looked_before {
                    looked_before = true;
                    directory = directory_before()?;
                }
                visit(&value, &mut |raw| {
                    references += 1;
                    let path = PathBuf::from(raw.strip_prefix("file://").unwrap_or(raw));
                    let resolved = if path.is_absolute() {
                        Some(path)
                    } else {
                        directory
                            .as_ref()
                            .filter(|p| p.is_absolute())
                            .map(|p| p.join(path))
                    };
                    let mut record = serde_json::json!({"type":"polylogue.local_reference","transcript":transcript,"source_start":offset,"source_end":offset+line.len() as u64,"reference":raw,"cwd":directory.as_ref().map(|p|p.to_string_lossy()),"path":resolved.as_ref().map(|p|p.to_string_lossy())});
                    if let Some(path) = resolved {
                        match self.capture_local_file(&path, pass) {
                            Ok(Some(hash)) => {
                                record["status"] = "collected".into();
                                record["input"] = hash.into();
                            }
                            Ok(None) => record["status"] = "missing".into(),
                            Err(error) => {
                                record["status"] = "unreadable".into();
                                record["reason"] = error.to_string().into();
                            }
                        }
                    } else {
                        record["status"] = "unresolved".into();
                        record["reason"] = "relative reference has no recorded absolute cwd".into();
                    }
                    serde_json::to_writer(&mut manifest, &record)?;
                    manifest.write_all(b"\n")?;
                    Ok(())
                })?;
            }
            offset += line.len() as u64;
            line.clear();
        }
        if references > 0 {
            manifest.rewind()?;
            let prepared = self.archive.prepare_bytes(&mut manifest)?;
            let id = prepared.hash().to_string();
            if self.reads.reader()?.input_progress(&id)?.is_none() {
                let origin =
                    polylogue_sources::origin::by_name("tool-context").expect("declared origin");
                let observation = self.observation(
                    Path::new(&format!("codex-local-references/{transcript}.jsonl")),
                    None,
                );
                let batch = polylogue_store::Batch {
                    required_outcome: None,
                    input: InputKey {
                        id,
                        format: origin.name.into(),
                        route: polylogue_model::Route::Copy,
                    },
                    change: Change::First,
                    observation: Some(observation),
                    continues: false,
                    new_blob: Some(polylogue_store::NewBlob::whole(
                        prepared.hash(),
                        prepared.len(),
                    )),
                    blobs: Vec::new(),
                    length: prepared.len(),
                    parse: vec![polylogue_store::ParseStep::without_parser(
                        0..prepared.len(),
                        true,
                        origin.parse_version(),
                    )],
                    capture_read: None,
                    base: None,
                    release: Default::default(),
                };
                self.writer
                    .store_group(
                        vec![polylogue_store::staging::PreparedBatch {
                            batch,
                            files: vec![prepared],
                        }],
                        Arc::clone(&self.blobs),
                    )?
                    .pop()
                    .expect("coverage result")?;
            }
        }
        Ok(())
    }

    fn capture_local_file(
        &mut self,
        path: &Path,
        pass: &mut Pass,
    ) -> Result<Option<String>, EngineError> {
        let origin =
            polylogue_sources::origin::by_name("codex-local-file").expect("declared origin");
        let context = self.sampling_context(path, origin.name)?;
        // A FIFO can block during open, before metadata can reject it. Check
        // the opened descriptor so replacing the path cannot bypass the check.
        let mut file = match std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(path)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let before = file.metadata()?;
        if !before.is_file() {
            return Err(
                polylogue_store::StoreError::Failed(polylogue_model::Failure::stated(
                    "",
                    polylogue_model::Cause::Environment,
                    polylogue_model::Code::Io,
                    "local reference is not a regular file",
                ))
                .into(),
            );
        }
        let facts = FileFacts::of(&before);
        let prepared = self.archive.prepare_bytes(&mut file)?;
        if facts != FileFacts::of(&file.metadata()?) || facts.size != prepared.len() {
            return Err(polylogue_store::StoreError::SourceChanged(
                "local file changed during capture",
            )
            .into());
        }
        let id = prepared.hash().to_string();
        pass.bytes_read += prepared.len();
        let exists = self.reads.reader()?.input_progress(&id)?.is_some();
        let observation = self.observation(path, Some(facts));
        let source_read = context.finish(
            polylogue_store::primary::RawRange {
                start: 0,
                end: prepared.len(),
            },
            prepared.hash(),
        );
        let batch = polylogue_store::Batch {
            required_outcome: None,
            input: InputKey {
                id: id.clone(),
                format: origin.name.into(),
                route: polylogue_model::Route::Copy,
            },
            change: if exists {
                Change::Unchanged
            } else {
                Change::First
            },
            observation: Some(observation),
            continues: false,
            new_blob: None,
            blobs: Vec::new(),
            length: prepared.len(),
            parse: Vec::new(),
            capture_read: Some(source_read),
            base: context.base,
            release: Default::default(),
        };
        if !exists {
            pass.bytes_stored += prepared.len();
        }
        // Equal content may have another observed source path.
        if exists {
            self.writer
                .store_group(
                    vec![polylogue_store::staging::PreparedBatch {
                        batch,
                        files: Vec::new(),
                    }],
                    Arc::clone(&self.blobs),
                )?
                .pop()
                .expect("one local result")?;
        } else {
            self.writer.store_bytes(prepared, batch)?;
        }
        Ok(Some(id))
    }
}

#[cfg(test)]
mod tests;
