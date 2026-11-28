//! Rollouts share recorded identities across locations and encodings. Decoding
//! streams into compressed preparation; no plaintext transcript is staged.
use crate::plan::{InputState, Plan};
use crate::*;
use polylogue_sources::origin::{InputIdentity, RecordedId};
use polylogue_store::staging::{PreparedBatch, PreparedBytes};
use std::io::{BufRead, BufReader, Read, Seek};

/// A file named by the ID it states rather than by its place (a Codex
/// rollout, plain or compressed), with how it states it. The origin's
/// other files are locations.
pub(crate) fn provider_id(found: &collect::Found) -> Option<polylogue_sources::origin::ProviderId> {
    match found.origin.input_identity {
        polylogue_sources::InputIdentity::ProviderId(id)
            if is_compressed(found) || (id.named)(&found.path) =>
        {
            Some(id)
        }
        _ => None,
    }
}

pub(crate) fn is_rollout(found: &collect::Found) -> bool {
    provider_id(found).is_some()
}

const UNIDENTIFIED_ROLLOUT: &str = "codex:unidentified-rollout:";

fn rollout_input(
    recorded: RecordedId,
    machine: &str,
    path: &Path,
    compressed: bool,
) -> Option<String> {
    match recorded {
        RecordedId::Named(id) => Some(InputIdentity::provider_named(&id, machine)),
        RecordedId::Incomplete => None,
        RecordedId::Unrecognized => {
            let location = if compressed {
                location_id(machine, &path.with_extension(""))
            } else {
                location_id(machine, path)
            };
            Some(format!("{UNIDENTIFIED_ROLLOUT}{location}"))
        }
    }
}

/// The stored input name records that the provider did not identify it.
/// It still means the same thing when a rebuild reads the chain alone.
pub(crate) fn rollout_refusal(input: &InputKey) -> Option<&'static str> {
    input
        .id
        .starts_with(UNIDENTIFIED_ROLLOUT)
        .then_some("first complete Codex rollout record does not name a session")
}

fn is_compressed(found: &collect::Found) -> bool {
    found.origin.compressed_form_of.is_some()
}

/// The origin whose parser reads a found file: its own, or for a compressed
/// file the origin it is a compressed form of.
fn read_as(found: &'static OriginSpec) -> &'static OriginSpec {
    found
        .compressed_form_of
        .and_then(polylogue_sources::origin::by_name)
        .unwrap_or(found)
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

/// Reads until `buffer` is full or the stream ends; how many bytes it read.
fn read_up_to(stream: &mut impl Read, buffer: &mut [u8]) -> std::io::Result<usize> {
    let mut held = 0;
    while held < buffer.len() {
        match stream.read(&mut buffer[held..]) {
            Ok(0) => break,
            Ok(n) => held += n,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(held)
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
    /// The input named by the rollout or, when it names none, by its location.
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
    previous: Option<polylogue_store::LatestChain>,
    original_added: u64,
    added: u64,
    /// Bytes of the file read to capture it.
    read: u64,
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
            let first = self.store(archive, blobs, writer, read_memory, None)?;
            if first.is_some() || self.found_origin.compressed_form_of.is_some() {
                return Ok(first);
            }
            let frozen = read::freeze_after_growth(&self.path, self.facts, &archive.staging_dir())?;
            match frozen {
                Some(frozen) => self.store(archive, blobs, writer, read_memory, Some(&frozen)),
                None => Ok(None),
            }
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
        &self,
        archive: &polylogue_store::Archive,
        blobs: &Arc<polylogue_store::blobs::Blobs>,
        writer: &polylogue_store::writer::WriterSender,
        read_memory: u64,
        frozen: Option<&read::FrozenFile>,
    ) -> Result<Option<(Captured, GroupReply)>, EngineError> {
        let path = &self.path;
        let id = &self.input;
        let found_origin = self.found_origin;
        let facts = self.facts;
        let context = &self.context;
        let original_context = &self.original_context;
        let mut observation = self.observation.clone();
        let origin = read_as(found_origin);
        let compressed = found_origin.compressed_form_of.is_some();
        let reads = polylogue_store::ReadPool::with_budget(archive, 1, read_memory)?;
        let (mut source, observed) = match frozen {
            Some(frozen) => {
                let source = frozen.open()?;
                let InputIdentity::ProviderId(provider) = found_origin.input_identity else {
                    unreachable!("a captured rollout has a provider ID");
                };
                let recorded = (provider.read)(&mut decoded(&source, false)?)?;
                if rollout_input(recorded, &observation.machine, path, false).as_ref() != Some(id) {
                    return Ok(None);
                }
                (source, frozen.observed.clone())
            }
            None => {
                let source = std::fs::File::open(path)?;
                let metadata = source.metadata()?;
                if FileFacts::of(&metadata) != facts {
                    return Ok(None);
                }
                (source, read::file_snapshot(&metadata)?)
            }
        };
        // The compressed file itself, with the read context taken before it.
        let original = match original_context {
            Some(context) => Some((archive.prepare_bytes(&mut source)?, context)),
            None => None,
        };
        // A read holds the index file against the writer's commits, so each
        // read ends before the file's bytes are compared or prepared.
        let (previous, chain) = {
            let reader = reads.reader()?;
            let previous = reader.latest_chain(id)?;
            let chain = match &previous {
                Some(previous) => Some(reader.stored_chain(id, previous.chain, previous.stored)?),
                _ => None,
            };
            (previous, chain)
        };
        let mut stream = decoded(&source, compressed)?;
        let mut same = false;
        let mut appended = false;
        let mut historical = None;
        // The copy ends inside the stored chain and every byte it holds is
        // the chain's byte at its place: how many bytes it holds.
        let mut shorter = None;
        if let Some(chain) = chain {
            let mut stored = chain.reader(Arc::clone(blobs));
            let mut left = [0; 65536];
            let mut right = [0; 65536];
            let mut matches = true;
            let mut compared = 0u64;
            loop {
                let n = stored.read(&mut left)?;
                if n == 0 {
                    break;
                }
                let held = read_up_to(&mut stream, &mut right[..n])?;
                if left[..held] != right[..held] {
                    matches = false;
                    break;
                }
                compared += held as u64;
                if held < n {
                    // The copy ended inside the chain, equal so far.
                    matches = false;
                    shorter = (compared > 0).then_some(compared);
                    break;
                }
            }
            if matches {
                same = stream.fill_buf()?.is_empty();
                appended = !same;
            } else {
                // A copy ending where a stored revision ends is that
                // revision seen again; one ending between revisions is the
                // chain seen back holding a prefix.
                historical = matching_revision(&reads.reader()?, blobs, id, &source, compressed)?;
                if historical.is_some() {
                    shorter = None;
                }
                stream = decoded(&source, compressed)?;
            }
        }
        // Equal decoded content still has a body that was read. Keep its
        // encoding through the immutable Accepted/Unresolved outcome.
        // A shorter copy stores no bytes: the chain holds them already.
        let prepared = match shorter {
            Some(_) => None,
            None => Some(archive.prepare_bytes(if same || historical.is_some() {
                decoded(&source, compressed)?
            } else {
                stream
            })?),
        };
        if frozen.is_none()
            && (FileFacts::of(&source.metadata()?) != facts
                || std::fs::metadata(path).map(|m| FileFacts::of(&m))? != facts)
        {
            return Ok(None);
        }
        let length = if let Some(held) = shorter {
            held
        } else if let Some(point) = historical {
            point.length
        } else if same {
            previous.as_ref().expect("same stored input").stored
        } else {
            prepared.as_ref().expect("new bytes").len()
                + if appended {
                    previous.as_ref().expect("prefix").stored
                } else {
                    0
                }
        };
        let added = if same || historical.is_some() || shorter.is_some() {
            0
        } else {
            prepared.as_ref().expect("new bytes").len()
        };
        let original_observation = observation.clone();
        let mut group = Vec::new();
        let mut original_added = 0;
        if let Some((original, original_context)) = original {
            let original_id = original.hash().to_string();
            let exists = reads.reader()?.latest_chain(&original_id)?.is_some();
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
                original_context,
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
        let change = if shorter.is_some() {
            // Every byte the copy holds is the stored chain's byte at its
            // place: the chain seen back, as a shorter plain file is.
            Change::Shorter {
                chain: previous.as_ref().expect("a stored chain").chain,
            }
        } else if let Some(point) = historical {
            Change::Observed {
                revision: point.number,
                chain: point.chain,
            }
        } else if same {
            Change::Unchanged
        } else if appended {
            Change::Appended {
                from: previous.as_ref().expect("prefix").stored,
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
            context,
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
        let reply = writer.enqueue_group(group)?;
        Ok(Some((
            Captured {
                previous,
                original_added,
                added,
                read: facts.size,
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
        let origin = read_as(found.origin);
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
            if let Some(progress) = self.reads.reader()?.latest_chain(&id)? {
                self.inputs
                    .get_mut(path)
                    .expect("known rollout")
                    .take_progress(&progress);
            }
            return Ok(RolloutStep::Planned(self.stored_plan(path)?.map(Box::new)));
        }
        // A rollout with an incomplete first record is tried again when it
        // grows. Keep complete files without a provider ID at their location.
        let source = std::fs::File::open(path)?;
        let read_id = provider_id(found).expect("a rollout states its ID").read;
        let recorded = read_id(&mut decoded(&source, compressed)?)?;
        let Some(input) = rollout_input(recorded, &self.config.machine, path, compressed) else {
            return Ok(RolloutStep::Planned(None));
        };
        Ok(RolloutStep::Capture(Box::new(RolloutCapture {
            path: path.clone(),
            input,
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
        pass.bytes_read += captured.read;
        pass.bytes_stored += captured.original_added + captured.added;
        let progress = self.reads.reader()?.latest_chain(&id)?.ok_or_else(|| {
            EngineError::internal("captured rollout input has no current revision")
        })?;
        let previous = captured.previous;
        if let Some(build) = self.archive.index_build() {
            let bytes = build
                .bytes
                .saturating_sub(previous.as_ref().map_or(0, |p| p.stored))
                .saturating_add(progress.stored);
            let read = build
                .bytes_read
                .saturating_sub(previous.as_ref().map_or(0, |p| p.state.parsed_to()))
                .saturating_add(progress.state.parsed_to());
            self.archive.record_build_bytes(bytes, read);
        }
        let mut state = InputState::new(id, read_as(found_origin));
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
