//! Only the receiving IO path creates source-read records; parser payloads do not.
use crate::*;
use polylogue_store::primary::{AcquisitionReceipt, AcquisitionRoute, RawRange, SourceSelector};
/// What every source-read receipt of one source is built from: the source,
/// how it was acquired and when.
#[derive(Debug, Clone)]
pub(crate) struct SourceReadContext {
    pub source: SourceSelector,
    pub route: AcquisitionRoute,
    pub clock: polylogue_model::WallClock,
    /// The source position fixed before the read; none when the read states no
    /// source coordinates.
    pub basis: Option<polylogue_store::primary::SourceBasis>,
    /// The file's facts when the plan was made; the read refuses a file that
    /// no longer matches them.
    pub facts: Option<FileFacts>,
    /// The last body already held in the spool, which the read's first range
    /// continues; it becomes the receipt's predecessor.
    pub previous_body: Option<String>,
    /// The outcome the writer must report for the batch; set when the held
    /// bytes did not prepare cleanly.
    pub required_outcome: Option<polylogue_store::primary::CaptureOutcome>,
}
impl SourceReadContext {
    pub(crate) fn finish(
        &self,
        range: RawRange,
        hash: polylogue_model::BlobHash,
    ) -> AcquisitionReceipt {
        AcquisitionReceipt {
            version: AcquisitionReceipt::VERSION,
            route: self.route,
            source: self.source.clone(),
            range,
            hash,
            observed_at: self.clock.now().into(),
            generation: None,
            raw_total: None,
            raw_predecessor: None,
            chunks: Vec::new(),
        }
    }
    pub(crate) fn finish_file(
        &self,
        range: RawRange,
        hash: polylogue_model::BlobHash,
        input: InputKey,
        observed: polylogue_store::primary::FileSnapshot,
        role: polylogue_store::primary::RawStreamRole,
        window: RawRange,
    ) -> AcquisitionReceipt {
        self.finish_observed(
            range,
            hash,
            self.observed_file(input, observed, role, window),
        )
    }
    pub(crate) fn observed_file(
        &self,
        input: InputKey,
        observed: polylogue_store::primary::FileSnapshot,
        role: polylogue_store::primary::RawStreamRole,
        window: RawRange,
    ) -> polylogue_store::primary::SourceGeneration {
        polylogue_store::primary::SourceGeneration::File {
            stream: polylogue_store::primary::RawStream {
                source: self.source.clone(),
                input,
                role,
                namespace: None,
            },
            observed,
            window,
        }
    }
    pub(crate) fn finish_observed(
        &self,
        range: RawRange,
        hash: polylogue_model::BlobHash,
        generation: polylogue_store::primary::SourceGeneration,
    ) -> AcquisitionReceipt {
        let window = generation.window();
        let mut record = self.finish(range, hash);
        record.generation = Some(generation);
        record.raw_total = (range.end == window.end).then_some(window.end);
        record.raw_predecessor = self.previous_body.clone();
        record
    }
}

pub(crate) fn file_snapshot(
    metadata: &std::fs::Metadata,
) -> std::io::Result<polylogue_store::primary::FileSnapshot> {
    use polylogue_store::primary::{FileSnapshot, FileTime};
    use std::os::unix::fs::MetadataExt;
    let time = |seconds, nanos: i64| -> std::io::Result<FileTime> {
        let nanos = u32::try_from(nanos)
            .ok()
            .filter(|nanos| *nanos < 1_000_000_000)
            .ok_or_else(|| {
                std::io::Error::other("provider file has invalid timestamp nanoseconds")
            })?;
        Ok(FileTime { seconds, nanos })
    };
    Ok(FileSnapshot {
        device: metadata.dev(),
        inode: metadata.ino(),
        size: metadata.size(),
        modified: time(metadata.mtime(), metadata.mtime_nsec())?,
        changed: time(metadata.ctime(), metadata.ctime_nsec())?,
    })
}
/// Headers read per page when looking for a group member; a page is one
/// store read, so this bounds memory per page, not what can be found.
const GROUP_HEADER_PAGE: usize = 32;
impl Engine {
    pub(crate) fn prepare_byte_read(
        &mut self,
        plan: &mut crate::plan::Plan,
    ) -> Result<(), EngineError> {
        if !plan.tracked || !matches!(plan.source, crate::pool::Bytes::File) {
            return Ok(());
        }
        let Some(facts) = plan.facts else {
            return Ok(());
        };
        let file = std::fs::File::open(&plan.path)?;
        let metadata = file.metadata()?;
        if FileFacts::of(&metadata) != facts {
            return Err(
                std::io::Error::other("source file changed before the spool lookup").into(),
            );
        }
        let context = self.source_read_context(
            &plan.path,
            &plan.input.format,
            AcquisitionRoute::NativeCapture,
        );
        let generation = context.observed_file(
            plan.input.clone(),
            file_snapshot(&metadata)?,
            polylogue_store::primary::RawStreamRole::ProviderBytes,
            RawRange {
                start: match plan.change {
                    Change::Appended { from } => from,
                    _ => 0,
                },
                end: facts.size,
            },
        );
        let cursor = self
            .writer
            .byte_cursor(polylogue_store::primary::ByteCursorQuery {
                generation,
                tail_body: None,
            })?;
        if FileFacts::of(&file.metadata()?) != facts
            || FileFacts::of(&std::fs::metadata(&plan.path)?) != facts
        {
            return Err(
                std::io::Error::other("source file changed during the spool lookup").into(),
            );
        }
        if let Some(cursor) = cursor.filter(|cursor| {
            cursor.range().end > plan.stored_upto || (plan.size == 0 && cursor.is_complete_window())
        }) {
            plan.byte_read = Some(crate::plan::ByteRead {
                cursor: Some(cursor),
                spooled: None,
            });
            plan.storing_only = true;
            plan.parser_state = None;
            plan.storing_started = true;
        }
        Ok(())
    }

    pub(crate) fn record_byte_read_status(
        &mut self,
        plan: &crate::plan::Plan,
        source_read: Option<&AcquisitionReceipt>,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        let Some(byte_read) = &plan.byte_read else {
            return Ok(());
        };
        let Some(cursor) = &byte_read.cursor else {
            // Applying confirmed events may remove the spooled body before this
            // byte read returns. Adopt only that same logical source; a
            // missing cursor shows neither spooled bytes nor progress.
            if byte_read.spooled.is_none()
                && plan.tracked
                && matches!(plan.source, crate::pool::Bytes::File)
                && plan.facts.is_some_and(|facts| facts.size == plan.size)
                && let Some(source_read) = source_read
                && source_read.range.end <= plan.size
            {
                let known = self.reads.reader()?.known_watched_input_at(
                    &self.config.machine,
                    plan.path.as_os_str().as_encoded_bytes(),
                )?;
                if let Some(known) = known
                    && known.input_id == plan.input.id
                    && known.format == plan.input.format
                    && known.facts == plan.facts
                    && known.chain.is_some()
                    && known.stored >= source_read.range.end
                {
                    let parse_unfinished =
                        !known.parsed || (!known.waiting && known.parsed_upto < known.stored);
                    pass.capture_pending |=
                        known.stored < plan.size || (has_reading(plan.origin) && parse_unfinished);
                    self.refresh_confirmed_input(&plan.path)?;
                    return Ok(());
                }
            }
            pass.errors.push(format!(
                "bytes kept in the spool for {}; continuation check unavailable",
                plan.input.id
            ));
            return Ok(());
        };
        if !cursor.is_complete_window() {
            pass.errors.push(format!(
                "byte coverage for {} has no verified EOF",
                plan.input.id
            ));
        }
        match cursor.outcome() {
            Some(polylogue_store::primary::CaptureOutcome::Prepared) => pass.capture_pending = true,
            Some(polylogue_store::primary::CaptureOutcome::UnresolvedBasisConflict) => {
                pass.errors.push(format!(
                    "bytes kept in the spool for {}; source acceptance unresolved",
                    plan.input.id
                ))
            }
            None => pass.errors.push(format!(
                "bytes kept in the spool for {}; capture outcome unknown",
                plan.input.id
            )),
        }
        Ok(())
    }

    pub(crate) fn apply_byte_read_result(
        &mut self,
        plan: &mut crate::plan::Plan,
        done: &crate::pool::Done,
    ) -> Result<(), EngineError> {
        let previous = plan
            .byte_read
            .as_ref()
            .and_then(|byte_read| byte_read.cursor.as_ref())
            .map(|cursor| cursor.last_body().to_owned());
        let spooled = done.spooled_group.clone().or_else(|| {
            plan.byte_read
                .as_ref()
                .and_then(|byte_read| byte_read.spooled.clone())
        });
        // A failed write cannot allow another byte read. In particular,
        // an unchanged old prefix must not retry this same failed range forever.
        let cursor = if done.write_failed {
            None
        } else if let Some(source_read) = &done.source_read {
            if let Some(generation) = &source_read.generation {
                let group = done
                    .spooled_group
                    .as_ref()
                    .map(|spooled| spooled.identity.as_str())
                    .or(done.pending.as_deref());
                let tail_body = match group {
                    Some(group) => self
                        .member_tail_body(group, &plan.input, source_read)?
                        .or(previous),
                    None => previous,
                };
                self.writer
                    .byte_cursor(polylogue_store::primary::ByteCursorQuery {
                        generation: generation.clone(),
                        tail_body,
                    })?
                    .filter(|cursor| {
                        cursor.range().end >= source_read.range.end
                            && (source_read.range.end > source_read.range.start
                                || cursor.is_complete_window())
                    })
            } else {
                None
            }
        } else {
            None
        };
        plan.byte_read = Some(crate::plan::ByteRead { cursor, spooled });
        // Parsed rows and checkpoints from a rejected capture stay recorded with
        // its spooled bytes. They do not become the current sampler's continuation.
        plan.storing_only = true;
        plan.parser_state = None;
        plan.storing_started = true;
        Ok(())
    }

    fn member_tail_body(
        &mut self,
        group: &str,
        input: &InputKey,
        source_read: &polylogue_store::primary::AcquisitionReceipt,
    ) -> Result<Option<String>, EngineError> {
        let mut after = None;
        let mut found = None;
        loop {
            let page = self
                .writer
                .pending_group_headers(group.into(), after, GROUP_HEADER_PAGE)?;
            if page.outcome.is_none() {
                return Ok(None);
            }
            for header in page.headers {
                let polylogue_store::primary::Capture::Input(captured) = header.captured else {
                    continue;
                };
                if captured.rows.input == input.id
                    && captured.rows.format == input.format
                    && captured.generation == source_read.generation
                    && captured.raw_range == Some(source_read.range)
                    && captured.hash == source_read.hash
                    && captured.length == source_read.range.length()?
                {
                    if found.is_some() {
                        return Ok(None);
                    }
                    found = Some(header.identity);
                }
            }
            let Some(next) = page.next else {
                return Ok(found);
            };
            if after.is_some_and(|previous| next <= previous) {
                return Err(polylogue_store::StoreError::invalid(
                    "spooled group page did not advance",
                )
                .into());
            }
            after = Some(next);
        }
    }

    pub(crate) fn sampling_context(
        &mut self,
        path: &Path,
        origin: &str,
    ) -> Result<SourceReadContext, EngineError> {
        let mut context = self.source_read_context(path, origin, AcquisitionRoute::NativeCapture);
        let snapshot = self.writer.source_snapshot(context.source.clone())?;
        context.basis = Some(snapshot.basis);
        Ok(context)
    }
    pub(crate) fn source_read_context(
        &self,
        path: &Path,
        origin: &str,
        route: AcquisitionRoute,
    ) -> SourceReadContext {
        SourceReadContext {
            source: SourceSelector {
                origin: origin.into(),
                machine: self.config.machine.clone(),
                path_bytes: path.as_os_str().as_encoded_bytes().to_vec(),
            },
            route,
            clock: self.config.clock.clone(),
            basis: None,
            facts: None,
            previous_body: None,
            required_outcome: None,
        }
    }
}
