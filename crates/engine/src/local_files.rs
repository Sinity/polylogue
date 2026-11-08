//! Native local-image references are read when observed. Recorded directories
//! alone resolve relative paths; coverage facts survive with the captured bytes.
use crate::*;
use polylogue_model::Json;
use std::io::{BufRead, Seek, Write};
use std::os::unix::fs::OpenOptionsExt;

fn visit(value: &Json, emit: &mut impl FnMut(&str) -> std::io::Result<()>) -> std::io::Result<()> {
    match value {
        Json::Object(o) => {
            if let Some(images) = o.get("local_images").and_then(Json::as_array) {
                for image in images {
                    if let Some(path) = image
                        .as_str()
                        .or_else(|| image.get("path").and_then(Json::as_str))
                    {
                        emit(path)?;
                    }
                }
            }
            if o.get("type").and_then(Json::as_str) == Some("local_image")
                && let Some(path) = o.get("path").and_then(Json::as_str)
            {
                emit(path)?;
            }
            if matches!(
                o.get("type").and_then(Json::as_str),
                Some("input_image" | "image")
            ) && let Some(path) = o
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

impl Engine {
    /// Source positions cannot advance before attachment observations. This
    /// ordinary plan path also resumes a capture interrupted after byte commit.
    pub(crate) fn observe_local_references(
        &mut self,
        plan: &crate::plan::Plan,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        // A loop without sources reads no file outside the archive, a
        // referenced image included.
        if !self.config.reads_sources || plan.origin.name != "codex" || plan.input.arrival_rank >= 2
        {
            return Ok(());
        }
        let Change::Reparse { chain } = plan.change else {
            return Ok(());
        };
        let reader = self.reads.reader()?;
        let stored = reader.stored_chain(&plan.input.id, chain, plan.size)?;
        drop(reader);
        self.capture_local_files(
            std::io::BufReader::new(stored.reader(Arc::clone(&self.blobs))),
            &plan.input.id,
            pass,
        )
    }

    pub(crate) fn capture_local_files(
        &mut self,
        mut source: impl BufRead,
        transcript: &str,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        let mut manifest = tempfile::tempfile_in(self.archive.staging_dir())?;
        let mut directory: Option<PathBuf> = None;
        let mut line = Vec::new();
        let mut offset = 0u64;
        let mut references = 0u64;
        while source.read_until(b'\n', &mut line)? > 0 {
            if let Ok(value) = serde_json::from_slice::<Json>(&line) {
                if matches!(
                    value.get("type").and_then(Json::as_str),
                    Some("session_meta" | "turn_context" | "older_header")
                ) && let Some(cwd) = value
                    .get("payload")
                    .and_then(|p| p.get("cwd"))
                    .and_then(Json::as_str)
                {
                    // A relative recorded cwd is insufficient to resolve a ref.
                    directory = Some(PathBuf::from(cwd));
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
                    required_disposition: None,
                    input: InputKey {
                        id,
                        format: origin.name.into(),
                        arrival_rank: 2,
                    },
                    change: Change::First,
                    observation: Some(observation),
                    continues: false,
                    new_blob: Some(polylogue_store::NewBlob {
                        hash: prepared.hash(),
                        size: prepared.len(),
                    }),
                    blobs: Vec::new(),
                    length: prepared.len(),
                    parse: {
                        let mut step = polylogue_store::ParseStep::new(
                            polylogue_model::Output::default(),
                            polylogue_model::RecordCounts::default(),
                            0..prepared.len(),
                            true,
                        );
                        step.progress =
                            polylogue_model::batch::StepProgress::NoParserYet { at_end: true };
                        step.reading_version = origin.reading_version();
                        vec![step]
                    },
                    acquisition: None,
                    basis: None,
                    release: Default::default(),
                };
                self.writer
                    .sender()?
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
        let context = self.sampling_context(path, origin.name, &origin.reading_version())?;
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
            return Err(std::io::Error::other("local reference is not a regular file").into());
        }
        let facts = FileFacts::of(&before);
        let prepared = self.archive.prepare_bytes(&mut file)?;
        if facts != FileFacts::of(&file.metadata()?) || facts.size != prepared.len() {
            return Err(std::io::Error::other("local file changed during capture").into());
        }
        let id = prepared.hash().to_string();
        pass.bytes_read += prepared.len();
        let exists = self.reads.reader()?.input_progress(&id)?.is_some();
        let observation = self.observation(path, Some(facts));
        let acquisition = context.finish(
            polylogue_store::primary::RawRange {
                start: 0,
                end: prepared.len(),
            },
            prepared.hash(),
        );
        let batch = polylogue_store::Batch {
            required_disposition: None,
            input: InputKey {
                id: id.clone(),
                format: "codex-local-file".into(),
                arrival_rank: 2,
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
            acquisition: Some(acquisition),
            basis: context.basis,
            release: Default::default(),
        };
        if !exists {
            pass.bytes_stored += prepared.len();
        }
        // Equal content may have another observed source path.
        if exists {
            self.writer
                .sender()?
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
