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
        && (is_compressed(found)
            || matches!(
                polylogue_sources::codex::File::from_path(&found.path),
                Some(polylogue_sources::codex::File::Rollout { .. })
            ))
}

fn is_compressed(found: &collect::Found) -> bool {
    found.origin.name == "codex-compressed"
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

fn matching_revision(
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

/// What the feeder decides for a found rollout: a plan over what is already
/// stored, or a capture of its bytes for a parse worker to run.
pub(crate) enum RolloutStep {
    Planned(Option<Box<Plan>>),
    Capture(Box<RolloutCapture>),
}

/// The writer's answer to a stored group.
type GroupReply = Receiver<Vec<polylogue_store::Result<polylogue_store::staging::StoredInput>>>;

/// A rollout capture prepared on the thread that hands out work, holding
/// everything the capture needs from the engine, so a parse worker prepares
/// the bytes and hands them to the writer while parsing goes on
/// ([`RolloutCapture::run`]).
pub(crate) struct RolloutCapture {
    pub(crate) path: PathBuf,
    /// The input the rollout's first record names.
    pub(crate) input: String,
    priority: pool::Priority,
    found_origin: &'static OriginSpec,
    pub(crate) facts: FileFacts,
    context: crate::capture_read::SourceReadContext,
    original_context: Option<crate::capture_read::SourceReadContext>,
    observation: Observation,
    /// A build's storing turn stores the rollout and leaves its reading to
    /// the build's workers, as it does for every other found file
    /// ([`Plan::storing_only`]).
    pub(crate) storing_only: bool,
}

/// What a worker's rollout capture did, for the feeder to record.
pub(crate) struct CapturedRollout {
    pub(crate) path: PathBuf,
    pub(crate) input: String,
    found_origin: &'static OriginSpec,
    priority: pool::Priority,
    facts: FileFacts,
    storing_only: bool,
    /// None when the file changed before or while it was read: nothing was
    /// stored, and the capture is prepared again from the new file facts.
    outcome: Result<Option<Captured>, EngineError>,
    /// The writer's answer for the group the capture handed it.
    reply: Option<GroupReply>,
}

struct Captured {
    previous: Option<polylogue_store::read::ChainProgress>,
    original_added: u64,
    added: u64,
}

impl CapturedRollout {
    /// Waits for the writer's answer to the stored group: off the thread
    /// that hands out work, which then plans the reading.
    pub(crate) fn answered(&mut self) {
        let Some(reply) = self.reply.take() else {
            return;
        };
        let outcomes = match reply.recv() {
            Ok(outcomes) => outcomes,
            Err(_) => {
                self.outcome = Err(polylogue_store::StoreError::WriterGone.into());
                return;
            }
        };
        for outcome in outcomes {
            if let Err(error) = outcome {
                self.outcome = Err(error.into());
                return;
            }
        }
    }
}

impl RolloutCapture {
    /// Prepares the rollout's bytes (and a compressed file's own bytes) and
    /// hands them to the writer as one group. Runs on a parse worker, which
    /// goes on to other work while the writer stores them.
    pub(crate) fn run(
        self,
        archive: &polylogue_store::Archive,
        blobs: &Arc<polylogue_store::blobs::Blobs>,
        writer: &polylogue_store::writer::WriterSender,
        read_memory: u64,
    ) -> CapturedRollout {
        let (path, input, facts) = (self.path.clone(), self.input.clone(), self.facts);
        let (found_origin, priority, storing_only) =
            (self.found_origin, self.priority, self.storing_only);
        let stored = polylogue_store::timing::time(polylogue_store::timing::Stage::Store, || {
            self.store(archive, blobs, writer, read_memory)
        });
        let (outcome, reply) = match stored {
            Ok(Some((captured, reply))) => (Ok(Some(captured)), Some(reply)),
            Ok(None) => (Ok(None), None),
            Err(error) => (Err(error), None),
        };
        CapturedRollout {
            path,
            input,
            found_origin,
            priority,
            facts,
            storing_only,
            outcome,
            reply,
        }
    }

    fn store(
        self,
        archive: &polylogue_store::Archive,
        blobs: &Arc<polylogue_store::blobs::Blobs>,
        writer: &polylogue_store::writer::WriterSender,
        read_memory: u64,
    ) -> Result<Option<(Captured, GroupReply)>, EngineError> {
        let RolloutCapture {
            path,
            input: id,
            found_origin,
            facts,
            context,
            original_context,
            mut observation,
            ..
        } = self;
        let path = &path;
        let origin = polylogue_sources::origin::by_name("codex").expect("declared origin");
        let compressed = found_origin.name == "codex-compressed";
        let reads = polylogue_store::ReadPool::with_budget(archive, 1, read_memory)?;
        let mut source = std::fs::File::open(path)?;
        if FileFacts::of(&source.metadata()?) != facts {
            return Ok(None);
        }
        let observed = crate::read::file_snapshot(&source.metadata()?)?;
        // The compressed file itself, with the read context taken before it.
        let original = match original_context {
            Some(context) => Some((archive.prepare_bytes(&mut source)?, context)),
            None => None,
        };
        // A read holds the index file against the writer's commits, so each
        // read ends before the file's bytes are compared or prepared.
        let (previous, chain) = {
            let reader = reads.reader()?;
            let previous = reader.input_progress(&id)?;
            let chain = match &previous {
                Some(previous) => {
                    Some(reader.stored_chain(&id, previous.chain, previous.length)?)
                }
                _ => None,
            };
            (previous, chain)
        };
        let mut stream = decoded(&source, compressed)?;
        let mut same = false;
        let mut appended = false;
        let mut historical = None;
        if let Some(chain) = chain {
            let mut stored = chain.reader(Arc::clone(blobs));
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
                historical = matching_revision(&reads.reader()?, blobs, &id, &source, compressed)?;
                stream = decoded(&source, compressed)?;
            }
        }
        // Equal decoded content still has a body that was read. Keep its
        // encoding through the immutable Accepted/Unresolved outcome.
        let prepared = Some(archive.prepare_bytes(if same || historical.is_some() {
            decoded(&source, compressed)?
        } else {
            stream
        })?);
        if FileFacts::of(&source.metadata()?) != facts
            || std::fs::metadata(path).map(|m| FileFacts::of(&m))? != facts
        {
            return Ok(None);
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
        let original_observation = observation.clone();
        let mut group = Vec::new();
        let mut original_added = 0;
        if let Some((original, original_context)) = original {
            let original_id = original.hash().to_string();
            let exists = reads.reader()?.input_progress(&original_id)?.is_some();
            if !exists {
                original_added = original.len();
            }
            let original_length = original.len();
            group.push(Engine::rollout_batch(
                if exists {
                    Change::Unchanged
                } else {
                    Change::First
                },
                original_observation,
                Some(original.clone()),
                &original_context,
                original_context.observed_file(
                    InputKey {
                        id: original_id.clone(),
                        format: found_origin.name.into(),
                        route: found_origin.route.imported(),
                    },
                    observed.clone(),
                    polylogue_store::capture_read::RawStreamRole::ProviderBytes,
                    polylogue_store::capture_read::RawRange {
                        start: 0,
                        end: original_length,
                    },
                ),
            ));
            observation.container = Some(polylogue_store::InContainer {
                input: original_id,
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
            polylogue_store::capture_read::RawStreamRole::DecodedMember {
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
            polylogue_store::capture_read::RawStreamRole::ProviderBytes
        };
        group.push(Engine::rollout_batch(
            change,
            observation,
            prepared,
            &context,
            context.observed_file(
                InputKey {
                    id: id.clone(),
                    format: origin.name.into(),
                    route: origin.route,
                },
                observed.clone(),
                role,
                polylogue_store::capture_read::RawRange {
                    start: 0,
                    end: length,
                },
            ),
        ));
        let reply = writer.enqueue_group(group, Arc::clone(blobs))?;
        Ok(Some((
            Captured {
                previous,
                original_added,
                added,
            },
            reply,
        )))
    }
}

impl Engine {
    /// Decides what a found rollout needs. What is already stored is planned
    /// here; new bytes become a [`RolloutCapture`] for a parse worker.
    pub(crate) fn rollout_step(
        &mut self,
        found: &collect::Found,
        facts: FileFacts,
        priority: pool::Priority,
    ) -> Result<RolloutStep, EngineError> {
        let path = &found.path;
        let origin = polylogue_sources::origin::by_name("codex").expect("declared origin");
        let compressed = is_compressed(found);
        let mut context = self.source_read_context(
            path,
            origin.name,
            polylogue_store::capture_read::ReadRoute::NativeCapture,
        );
        context.facts = Some(facts);
        let original_context = if compressed {
            Some(self.source_read_context(
                path,
                found.origin.name,
                polylogue_store::capture_read::ReadRoute::NativeCapture,
            ))
        } else {
            None
        };
        if self
            .inputs
            .get(path)
            .is_some_and(|s| s.facts == Some(facts))
        {
            let id = self.inputs[path].input_id.clone();
            if let Some(progress) = self.reads.reader()?.input_progress(&id)? {
                self.inputs
                    .get_mut(path)
                    .expect("known rollout")
                    .take_progress(&progress);
            }
            return Ok(RolloutStep::Planned(self.stored_plan(path)?.map(Box::new)));
        }
        // A rollout is named by its session ID, which its first record
        // states. Until that record is complete there is nothing to name the
        // input by, so nothing is stored; the file is read again when it grows.
        let source = std::fs::File::open(path)?;
        let Some(session_id) = polylogue_sources::codex::rollout_id(decoded(&source, compressed)?)?
        else {
            return Ok(RolloutStep::Planned(None));
        };
        Ok(RolloutStep::Capture(Box::new(RolloutCapture {
            path: path.clone(),
            input: format!("codex:rollout:{session_id}"),
            priority,
            found_origin: found.origin,
            facts,
            context,
            original_context,
            observation: self.observation(path, Some(facts)),
            storing_only: false,
        })))
    }

    /// Records a worker's rollout capture and plans the reading of what it
    /// stored, unless the capture only stores.
    pub(crate) fn captured_rollout(
        &mut self,
        captured: CapturedRollout,
        pass: &mut Pass,
    ) -> Result<Option<Plan>, EngineError> {
        let CapturedRollout {
            path,
            input: id,
            found_origin,
            priority,
            facts,
            storing_only,
            outcome,
            reply,
        } = captured;
        debug_assert!(reply.is_none(), "a capture is recorded once answered");
        // A storage condition that passes (the index file held by a long
        // write) and a file that changed are both captured again.
        let outcome = match outcome {
            Err(EngineError::Store(error))
                if polylogue_model::Caused::cause(&error) == polylogue_model::Cause::Transient =>
            {
                Ok(None)
            }
            outcome => outcome,
        };
        let Some(captured) = outcome? else {
            // Read again from what the file holds now; a file that went is
            // left to the next scan.
            let Ok(meta) = std::fs::metadata(&path) else {
                return Ok(None);
            };
            let found = collect::Found {
                path,
                origin: found_origin,
            };
            match self.rollout_step(&found, FileFacts::of(&meta), priority)? {
                RolloutStep::Planned(plan) => return Ok(plan.map(|plan| *plan)),
                RolloutStep::Capture(mut capture) => {
                    capture.storing_only = storing_only;
                    self.rollout_captures.push_front(*capture);
                    return Ok(None);
                }
            }
        };
        pass.bytes_read += facts.size;
        pass.bytes_stored += captured.original_added + captured.added;
        let progress = self.reads.reader()?.input_progress(&id)?.ok_or_else(|| {
            EngineError::internal("captured rollout input has no current revision")
        })?;
        let previous = captured.previous;
        if let Some(build) = self.archive.index_build() {
            let bytes = build
                .bytes
                .saturating_sub(previous.as_ref().map_or(0, |p| p.length))
                .saturating_add(progress.length);
            let read = build
                .bytes_read
                .saturating_sub(previous.as_ref().map_or(0, |p| p.parsed_upto))
                .saturating_add(progress.parsed_upto);
            self.archive.record_build_bytes(bytes, read);
        }
        let origin = polylogue_sources::origin::by_name("codex").expect("declared origin");
        let mut state = InputState::new(id, origin);
        state.facts = Some(facts);
        state.take_progress(&progress);
        self.inputs.insert(path.clone(), state);
        if storing_only {
            return Ok(None);
        }
        self.stored_plan(&path)
    }

    /// Captures a rollout on this thread, as the pass does on a worker.
    #[cfg(test)]
    pub(crate) fn capture_rollout(
        &mut self,
        found: &collect::Found,
        facts: FileFacts,
        pass: &mut Pass,
    ) -> Result<Option<Plan>, EngineError> {
        match self.rollout_step(found, facts, pool::Priority::Bulk)? {
            RolloutStep::Planned(plan) => Ok(plan.map(|plan| *plan)),
            RolloutStep::Capture(capture) => {
                let read_memory = self.config.reader_memory_per_connection()?;
                let mut captured = capture.run(
                    &self.archive,
                    &self.blobs,
                    &self.writer.sender(),
                    read_memory,
                );
                captured.answered();
                self.captured_rollout(captured, pass)
            }
        }
    }

    fn rollout_batch(
        change: Change,
        observation: Observation,
        bytes: Option<PreparedBytes>,
        context: &crate::capture_read::SourceReadContext,
        generation: polylogue_store::capture_read::ObservedState,
    ) -> PreparedBatch {
        let input = generation.stream().input.clone();
        let length = generation.window().end;
        let body_only = matches!(change, Change::Unchanged | Change::Observed { .. });
        let source_read = bytes.as_ref().map(|bytes| {
            context.finish_observed(
                polylogue_store::capture_read::RawRange {
                    start: length - bytes.len(),
                    end: length,
                },
                bytes.hash(),
                generation,
            )
        });
        PreparedBatch {
            batch: polylogue_store::Batch {
                input,
                change,
                observation: Some(observation),
                continues: false,
                length,
                new_blob: bytes
                    .as_ref()
                    .filter(|_| !body_only)
                    .map(|bytes| polylogue_store::NewBlob::whole(bytes.hash(), bytes.len())),
                blobs: bytes
                    .as_ref()
                    .filter(|_| body_only)
                    .map(|bytes| polylogue_store::NewBlob::whole(bytes.hash(), bytes.len()))
                    .into_iter()
                    .collect(),
                parse: Vec::new(),
                capture_read: source_read,
                release: Default::default(),
            },
            files: bytes.into_iter().collect(),
        }
    }
}
#[cfg(test)]
mod tests;
