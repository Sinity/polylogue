//! Only the receiving IO path creates body evidence; parser payloads do not.
use crate::*;
use polylogue_store::primary::{AcquisitionReceipt, AcquisitionRoute, RawRange, SourceSelector};
#[derive(Debug, Clone)]
pub(crate) struct AcquisitionContext {
    pub source: SourceSelector,
    pub route: AcquisitionRoute,
    pub clock: polylogue_model::WallClock,
    pub basis: Option<polylogue_store::primary::SourceBasis>,
    pub facts: Option<FileFacts>,
    pub raw_predecessor: Option<String>,
    pub required_disposition: Option<polylogue_store::primary::CaptureDisposition>,
}
impl AcquisitionContext {
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
        self.finish_generation(
            range,
            hash,
            self.file_generation(input, observed, role, window),
        )
    }
    pub(crate) fn file_generation(
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
    pub(crate) fn finish_generation(
        &self,
        range: RawRange,
        hash: polylogue_model::BlobHash,
        generation: polylogue_store::primary::SourceGeneration,
    ) -> AcquisitionReceipt {
        let window = generation.window();
        let mut receipt = self.finish(range, hash);
        receipt.generation = Some(generation);
        receipt.raw_total = (range.end == window.end).then_some(window.end);
        receipt.raw_predecessor = self.raw_predecessor.clone();
        receipt
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
impl Engine {
    pub(crate) fn prepare_raw_plan(
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
                std::io::Error::other("native generation changed before custody lookup").into(),
            );
        }
        let context = self.acquisition_context(
            &plan.path,
            &plan.input.format,
            AcquisitionRoute::NativeCapture,
        );
        let generation = context.file_generation(
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
        let cursor =
            self.writer
                .raw_acquisition_cursor(polylogue_store::primary::RawAcquisitionQuery {
                    generation,
                    anchor: None,
                })?;
        if FileFacts::of(&file.metadata()?) != facts
            || FileFacts::of(&std::fs::metadata(&plan.path)?) != facts
        {
            return Err(
                std::io::Error::other("native generation changed during custody lookup").into(),
            );
        }
        if let Some(cursor) = cursor.filter(|cursor| {
            cursor.range().end > plan.stored_upto || plan.size == 0 && cursor.is_complete_window()
        }) {
            plan.raw = Some(crate::plan::RawState {
                cursor: Some(cursor),
                custody: None,
            });
            plan.parser_deferred = true;
            plan.parser_state = None;
            plan.storing_started = true;
        }
        Ok(())
    }

    pub(crate) fn record_raw_status(
        &mut self,
        plan: &crate::plan::Plan,
        acquisition: Option<&AcquisitionReceipt>,
        pass: &mut Pass,
    ) -> Result<(), EngineError> {
        let Some(raw) = &plan.raw else { return Ok(()) };
        let Some(cursor) = &raw.cursor else {
            // Confirmed application may retire the producer body before this
            // physical attempt returns. Adopt only that exact logical source;
            // an absent raw cursor itself proves no custody or progress.
            if raw.custody.is_none()
                && plan.tracked
                && matches!(plan.source, crate::pool::Bytes::File)
                && plan.facts.is_some_and(|facts| facts.size == plan.size)
                && let Some(acquisition) = acquisition
                && acquisition.range.end <= plan.size
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
                    && known.stored >= acquisition.range.end
                {
                    pass.capture_pending |= known.stored < plan.size
                        || has_reading(plan.origin)
                            && (!known.parsed
                                || !known.waiting && known.parsed_upto < known.stored);
                    self.refresh_confirmed_input(&plan.path)?;
                    return Ok(());
                }
            }
            pass.errors.push(format!(
                "raw custody retained for {}; continuation proof unavailable",
                plan.input.id
            ));
            return Ok(());
        };
        if !cursor.is_complete_window() {
            pass.errors.push(format!(
                "raw coverage for {} has no verified EOF",
                plan.input.id
            ));
        }
        match cursor.disposition() {
            Some(polylogue_store::primary::CaptureDisposition::Prepared) => {
                pass.capture_pending = true
            }
            Some(polylogue_store::primary::CaptureDisposition::UnresolvedBasisConflict) => {
                pass.errors.push(format!(
                    "raw custody retained for {}; source acceptance unresolved",
                    plan.input.id
                ))
            }
            None => pass.errors.push(format!(
                "raw custody retained for {}; acquisition disposition unknown",
                plan.input.id
            )),
        }
        Ok(())
    }

    pub(crate) fn apply_raw_result(
        &mut self,
        plan: &mut crate::plan::Plan,
        done: &crate::pool::Done,
    ) -> Result<(), EngineError> {
        let previous = plan
            .raw
            .as_ref()
            .and_then(|raw| raw.cursor.as_ref())
            .map(|cursor| cursor.last_body().to_owned());
        let custody = done
            .raw_custody
            .clone()
            .or_else(|| plan.raw.as_ref().and_then(|raw| raw.custody.clone()));
        // A failed write cannot authorize another physical read. In particular,
        // an unchanged old prefix must not retry this same failed range forever.
        let cursor = if done.write_failed {
            None
        } else if let Some(receipt) = &done.raw_acquisition {
            if let Some(generation) = &receipt.generation {
                let group = done
                    .raw_custody
                    .as_ref()
                    .map(|custody| custody.identity.as_str())
                    .or(done.pending.as_deref());
                let anchor = match group {
                    Some(group) => self
                        .raw_member_anchor(group, &plan.input, receipt)?
                        .or(previous),
                    None => previous,
                };
                self.writer
                    .raw_acquisition_cursor(polylogue_store::primary::RawAcquisitionQuery {
                        generation: generation.clone(),
                        anchor,
                    })?
                    .filter(|cursor| {
                        cursor.range().end >= receipt.range.end
                            && (receipt.range.end > receipt.range.start
                                || cursor.is_complete_window())
                    })
            } else {
                None
            }
        } else {
            None
        };
        plan.raw = Some(crate::plan::RawState { cursor, custody });
        // Parsed facts/checkpoints from a rejected capture remain frozen data in
        // its custody. They do not become the current sampler's continuation.
        plan.parser_deferred = true;
        plan.parser_state = None;
        plan.storing_started = true;
        Ok(())
    }

    fn raw_member_anchor(
        &mut self,
        group: &str,
        input: &InputKey,
        receipt: &polylogue_store::primary::AcquisitionReceipt,
    ) -> Result<Option<String>, EngineError> {
        let mut after = None;
        let mut found = None;
        loop {
            let page = self.writer.pending_group_headers(group.into(), after, 32)?;
            if page.disposition.is_none() {
                return Ok(None);
            }
            for header in page.headers {
                let polylogue_store::primary::Capture::Input(captured) = header.captured else {
                    continue;
                };
                if captured.rows.input == input.id
                    && captured.rows.format == input.format
                    && captured.generation == receipt.generation
                    && captured.raw_range == Some(receipt.range)
                    && captured.hash == receipt.hash
                    && captured.length == receipt.range.length()?
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
                return Err(polylogue_store::StoreError::Invalid(
                    "raw custody group page did not advance".into(),
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
    ) -> Result<AcquisitionContext, EngineError> {
        let mut context = self.acquisition_context(path, origin, AcquisitionRoute::NativeCapture);
        let snapshot = self.writer.source_snapshot(context.source.clone())?;
        context.basis = Some(snapshot.basis);
        Ok(context)
    }
    pub(crate) fn acquisition_context(
        &self,
        path: &Path,
        origin: &str,
        route: AcquisitionRoute,
    ) -> AcquisitionContext {
        AcquisitionContext {
            source: SourceSelector {
                origin: origin.into(),
                machine: self.config.machine.clone(),
                path_bytes: path.as_os_str().as_encoded_bytes().to_vec(),
            },
            route,
            clock: self.config.clock.clone(),
            basis: None,
            facts: None,
            raw_predecessor: None,
            required_disposition: None,
        }
    }
}
