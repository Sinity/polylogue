//! Rollouts share recorded identities across locations and encodings. Decoding
//! streams into compressed preparation; no plaintext transcript is staged.
use crate::plan::{InputState, Plan};
use crate::*;
use polylogue_store::staging::{PreparedBatch, PreparedBytes};
use std::io::{BufRead, BufReader, Read, Seek};

/// A file named by the session ID it states rather than by its place: a
/// Codex rollout, plain or compressed. Codex's other files are locations.
pub(crate) fn is_rollout(found: &collect::Found) -> bool {
    found.origin.input_identity == polylogue_sources::InputIdentity::ProviderId
        && (found.origin.name == "codex-compressed"
            || matches!(
                polylogue_sources::codex::File::from_path(&found.path),
                Some(polylogue_sources::codex::File::Rollout { .. })
            ))
}

fn decoded(source: &std::fs::File, compressed: bool) -> std::io::Result<Box<dyn BufRead>> {
    let mut file = source.try_clone()?;
    file.rewind()?;
    if compressed {
        Ok(Box::new(BufReader::new(zstd::stream::read::Decoder::new(
            file,
        )?)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

fn recorded_copy(
    reader: &polylogue_store::Reader<'_>,
    blobs: &Arc<polylogue_store::blobs::Blobs>,
    input: &str,
    source: &std::fs::File,
    compressed: bool,
) -> polylogue_store::Result<Option<polylogue_store::read::RevisionPoint>> {
    let mut stream = decoded(source, compressed)?;
    let mut hash = blake3::Hasher::new();
    let mut length = 0u64;
    let mut buffer = [0; 65536];
    loop {
        let n = stream.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
        length += n as u64;
    }
    let expected = hash.finalize();
    let mut found = None;
    reader.visit_revisions_at_length(input, length, |point| {
        let chain = reader.stored_chain(input, point.chain, point.length)?;
        let mut stream = chain.reader(Arc::clone(blobs));
        let mut actual = blake3::Hasher::new();
        loop {
            let n = stream.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            actual.update(&buffer[..n]);
        }
        let matches = actual.finalize() == expected;
        if matches {
            found = Some(point);
        }
        Ok(matches)
    })?;
    Ok(found)
}

impl Engine {
    pub(crate) fn capture_rollout(
        &mut self,
        found: &collect::Found,
        facts: FileFacts,
        pass: &mut Pass,
    ) -> Result<Option<Plan>, EngineError> {
        let path = &found.path;
        if self.archive.is_primary() {
            self.refresh_confirmed_input(path)?;
        }
        let origin = polylogue_sources::origin::by_name("codex").expect("declared origin");
        let compressed = found.origin.name == "codex-compressed";
        let mut context = self.acquisition_context(
            path,
            origin.name,
            polylogue_store::primary::AcquisitionRoute::NativeCapture,
        );
        let snapshot = self.writer.source_snapshot(context.source.clone(), "")?;
        context.basis = Some(snapshot.basis.clone());
        context.facts = Some(facts);
        let raw_context = if compressed {
            Some(self.sampling_context(path, found.origin.name, "")?)
        } else {
            None
        };
        let has_pending = snapshot.pending.head.is_some();
        if let Some(raw) = &snapshot.pending.raw
            && let polylogue_store::primary::Capture::Input(input) = &raw.captured
            && input.rows.acquisition.as_ref().is_some_and(|receipt| {
                receipt.route == polylogue_store::primary::AcquisitionRoute::NativeCapture
            })
            && input
                .raw_range
                .is_some_and(|range| range.start == 0 && range.end == input.length)
            && input.rows.observation.as_ref().is_some_and(|observation| {
                observation.device == Some(facts.device as i64)
                    && observation.inode == Some(facts.inode as i64)
                    && observation.size == Some(facts.size as i64)
                    && observation.mtime_ns == i64::try_from(facts.mtime_ns).ok()
                    && observation.ctime_ns == i64::try_from(facts.ctime_ns).ok()
            })
        {
            pass.capture_pending = true;
            return self.stored_plan(path);
        }
        if self
            .inputs
            .get(path)
            .is_some_and(|s| s.facts == Some(facts))
            && !has_pending
        {
            let id = self.inputs[path].input_id.clone();
            if let Some(progress) = self.reads.reader()?.input_progress(&id)? {
                let state = self.inputs.get_mut(path).expect("known rollout");
                if state.chain != Some(progress.chain) || state.stored != progress.length {
                    state.parser_state = None;
                    state.tail = None;
                }
                state.chain = Some(progress.chain);
                state.stored = progress.length;
                state.parsed_upto = progress.parsed_upto;
                state.parsed = progress.parsed;
                state.waiting = progress.waiting;
            }
            return self.stored_plan(path);
        }
        let mut source = std::fs::File::open(path)?;
        let observed = crate::acquisition::file_snapshot(&source.metadata()?)?;
        // A rollout is named by its session ID, which its first record
        // states. Until that record is complete there is nothing to name the
        // input by, so nothing is stored; the file is read again when it grows.
        let Some(native) = polylogue_sources::codex::rollout_id(decoded(&source, compressed)?)?
        else {
            return Ok(None);
        };
        let id = format!("codex:rollout:{native}");
        let raw = if compressed {
            // Reading the session ID moved the shared file offset.
            source.rewind()?;
            Some(self.archive.prepare_bytes(&mut source)?)
        } else {
            None
        };
        let reader = self.reads.reader()?;
        let previous = reader.input_progress(&id)?;
        let mut stream = decoded(&source, compressed)?;
        let mut same = false;
        let mut appended = false;
        let mut historical = None;
        if !has_pending && let Some(previous) = &previous {
            let chain = reader.stored_chain(&id, previous.chain, previous.length)?;
            let mut stored = chain.reader(Arc::clone(&self.blobs));
            let mut left = [0; 65536];
            let mut right = [0; 65536];
            let mut matches = true;
            loop {
                let n = stored.read(&mut left)?;
                if n == 0 {
                    break;
                }
                match stream.read_exact(&mut right[..n]) {
                    Ok(()) if left[..n] == right[..n] => {}
                    Ok(()) => {
                        matches = false;
                        break;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
                        matches = false;
                        break;
                    }
                    Err(error) => return Err(error.into()),
                }
            }
            if matches {
                same = stream.fill_buf()?.is_empty();
                appended = !same;
            } else {
                historical = recorded_copy(&reader, &self.blobs, &id, &source, compressed)?;
                stream = decoded(&source, compressed)?;
            }
        }
        // Equal logical content still has a real acquired body. Retain its
        // encoding through the immutable Accepted/Unresolved disposition.
        let prepared = Some(
            self.archive
                .prepare_bytes(if same || historical.is_some() {
                    decoded(&source, compressed)?
                } else {
                    stream
                })?,
        );
        drop(reader);
        if FileFacts::of(&source.metadata()?) != facts
            || std::fs::metadata(path).map(|m| FileFacts::of(&m))? != facts
        {
            return Err(std::io::Error::other("source changed during rollout capture").into());
        }
        let length = if let Some(point) = historical {
            point.length
        } else if same {
            previous.as_ref().expect("same stored input").length
        } else {
            prepared.as_ref().expect("new bytes").len()
                + if appended {
                    previous.as_ref().expect("prefix").length
                } else {
                    0
                }
        };
        let added = if same || historical.is_some() {
            0
        } else {
            prepared.as_ref().expect("new bytes").len()
        };
        let mut observation = self.observation(path, Some(facts));
        let mut group = Vec::new();
        let mut raw_added = 0;
        if let Some(raw) = raw {
            let raw_id = raw.hash().to_string();
            let exists = self.reads.reader()?.input_progress(&raw_id)?.is_some();
            if !exists {
                raw_added = raw.len();
            }
            let raw_length = raw.len();
            group.push(
                self.rollout_batch(
                    if exists {
                        Change::Unchanged
                    } else {
                        Change::First
                    },
                    self.observation(path, Some(facts)),
                    Some(raw.clone()),
                    raw_context.as_ref().expect("compressed acquisition"),
                    raw_context
                        .as_ref()
                        .expect("compressed acquisition")
                        .file_generation(
                            InputKey {
                                id: raw_id.clone(),
                                format: found.origin.name.into(),
                                arrival_rank: 2,
                            },
                            observed.clone(),
                            polylogue_store::primary::RawStreamRole::ProviderBytes,
                            polylogue_store::primary::RawRange {
                                start: 0,
                                end: raw_length,
                            },
                        ),
                ),
            );
            observation.container = Some(polylogue_store::InContainer {
                input: raw_id,
                member_index: 0,
                kind: "zstd".into(),
                path_bytes: path
                    .with_extension("")
                    .file_name()
                    .unwrap_or_default()
                    .as_encoded_bytes()
                    .to_vec(),
            });
        }
        let change = if let Some(point) = historical {
            Change::Observed {
                revision: point.number,
                chain: point.chain,
            }
        } else if same {
            Change::Unchanged
        } else if appended {
            Change::Appended {
                from: previous.as_ref().expect("prefix").length,
            }
        } else if previous.is_some() {
            Change::Rewritten
        } else {
            Change::First
        };
        let role = if compressed {
            polylogue_store::primary::RawStreamRole::DecodedMember {
                codec: "zstd".into(),
                member_index: 0,
                path_bytes: observation
                    .container
                    .as_ref()
                    .expect("decoded member")
                    .path_bytes
                    .clone(),
            }
        } else {
            polylogue_store::primary::RawStreamRole::ProviderBytes
        };
        group.push(self.rollout_batch(
            change,
            observation,
            prepared,
            &context,
            context.file_generation(
                InputKey {
                    id: id.clone(),
                    format: origin.name.into(),
                    arrival_rank: origin.arrival.rank(),
                },
                observed.clone(),
                role,
                polylogue_store::primary::RawRange {
                    start: 0,
                    end: length,
                },
            ),
        ));
        let outcomes = self
            .writer
            .sender()?
            .store_group(group, Arc::clone(&self.blobs))?;
        let mut pending = false;
        for outcome in outcomes {
            match outcome {
                Ok(_) => {}
                Err(polylogue_store::StoreError::PrimaryPending { .. }) => pending = true,
                Err(error) => return Err(error.into()),
            }
        }
        pass.bytes_read += facts.size;
        pass.bytes_stored += raw_added + added;
        if pending {
            pass.capture_pending = true;
            return Ok(None);
        }
        let progress = self.reads.reader()?.input_progress(&id)?.ok_or_else(|| {
            std::io::Error::other("captured native input has no current revision")
        })?;
        self.inputs.insert(
            path.clone(),
            InputState {
                input_id: id,
                origin,
                facts: Some(facts),
                stored: progress.length,
                parsed_upto: progress.parsed_upto,
                parsed: progress.parsed,
                waiting: progress.waiting,
                parser_state: None,
                tail: None,
                chain: Some(progress.chain),
                unread: None,
                working: None,
            },
        );
        self.stored_plan(path)
    }
    fn rollout_batch(
        &self,
        change: Change,
        observation: Observation,
        bytes: Option<PreparedBytes>,
        context: &crate::acquisition::AcquisitionContext,
        generation: polylogue_store::primary::SourceGeneration,
    ) -> PreparedBatch {
        let input = generation.stream().input.clone();
        let length = generation.window().end;
        let body_only = matches!(change, Change::Unchanged | Change::Observed { .. });
        let acquisition = bytes.as_ref().map(|bytes| {
            context.finish_generation(
                polylogue_store::primary::RawRange {
                    start: length - bytes.len(),
                    end: length,
                },
                bytes.hash(),
                generation,
            )
        });
        PreparedBatch {
            batch: polylogue_store::Batch {
                required_disposition: None,
                input,
                change,
                observation: Some(observation),
                continues: false,
                length,
                new_blob: bytes.as_ref().filter(|_| !body_only).map(|bytes| {
                    polylogue_store::NewBlob {
                        hash: bytes.hash(),
                        size: bytes.len(),
                    }
                }),
                blobs: bytes
                    .as_ref()
                    .filter(|_| body_only)
                    .map(|bytes| polylogue_store::NewBlob {
                        hash: bytes.hash(),
                        size: bytes.len(),
                    })
                    .into_iter()
                    .collect(),
                parse: Vec::new(),
                acquisition,
                basis: context.basis.clone(),
                release: Default::default(),
            },
            files: bytes.into_iter().collect(),
        }
    }
}
#[cfg(test)]
mod tests;
