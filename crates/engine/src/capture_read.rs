//! The receipts of capture reads: what every capture read of one source is
//! recorded with ([`polylogue_store::capture_read::CaptureRead`]), and the
//! source state a plan decides from. Only the receiving I/O path creates
//! receipts; parser payloads do not.
use crate::*;
use polylogue_store::capture_read::{CaptureRead, RawRange, ReadRoute, SourceSelector};

/// What every capture-read receipt of one source is built from: the source,
/// how it was acquired and when.
#[derive(Debug, Clone)]
pub(crate) struct SourceReadContext {
    pub source: SourceSelector,
    pub route: ReadRoute,
    pub clock: polylogue_model::WallClock,
    /// The file's facts when the plan was made; the read refuses a file that
    /// changed since in any way other than growing.
    pub facts: Option<FileFacts>,
}
impl SourceReadContext {
    pub(crate) fn finish(&self, range: RawRange, hash: polylogue_model::BlobHash) -> CaptureRead {
        CaptureRead {
            version: CaptureRead::VERSION,
            route: self.route,
            source: self.source.clone(),
            range,
            hash,
            observed_at: self.clock.now().into(),
            generation: None,
            raw_total: None,
            previous_revision: None,
            chunks: Vec::new(),
        }
    }
    pub(crate) fn finish_file(
        &self,
        range: RawRange,
        hash: polylogue_model::BlobHash,
        input: InputKey,
        observed: polylogue_store::capture_read::FileSnapshot,
        role: polylogue_store::capture_read::RawStreamRole,
        window: RawRange,
    ) -> CaptureRead {
        self.finish_observed(
            range,
            hash,
            self.observed_file(input, observed, role, window),
        )
    }
    pub(crate) fn observed_file(
        &self,
        input: InputKey,
        observed: polylogue_store::capture_read::FileSnapshot,
        role: polylogue_store::capture_read::RawStreamRole,
        window: RawRange,
    ) -> polylogue_store::capture_read::ObservedState {
        polylogue_store::capture_read::ObservedState::File {
            stream: polylogue_store::capture_read::RawStream {
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
        generation: polylogue_store::capture_read::ObservedState,
    ) -> CaptureRead {
        let window = generation.window();
        let mut record = self.finish(range, hash);
        record.generation = Some(generation);
        record.raw_total = (range.end == window.end).then_some(window.end);
        record
    }
}

/// What checking a planned file against the facts it was planned from found.
pub(crate) enum PlannedFile {
    /// The file still matches its plan.
    Ready,
    /// The file changed after it was planned; it is planned again.
    Changed,
    /// The file could not be opened or examined.
    Unreadable(std::io::Error),
}
impl Engine {
    /// Checks a planned file against the facts it was planned from. A file
    /// that changed, or cannot be read, is that file's condition, never the
    /// pass's.
    pub(crate) fn check_planned_file(&self, plan: &crate::plan::Plan) -> PlannedFile {
        if !plan.tracked || !matches!(plan.source, crate::pool::Bytes::File) {
            return PlannedFile::Ready;
        }
        let Some(facts) = plan.facts else {
            return PlannedFile::Ready;
        };
        match std::fs::metadata(&plan.path) {
            Ok(metadata) if FileFacts::of(&metadata) == facts => PlannedFile::Ready,
            Ok(_) => PlannedFile::Changed,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => PlannedFile::Changed,
            Err(error) => PlannedFile::Unreadable(error),
        }
    }

    /// The source state a plan decides from, read from committed state, so
    /// the thread that hands out parse work never waits behind the writer's
    /// queue for it; a path has one job at a time and its own writes commit
    /// before its next job is planned, so that read already holds them.
    pub(crate) fn source_snapshot(
        &self,
        source: SourceSelector,
    ) -> Result<polylogue_store::capture_read::SourceSnapshot, EngineError> {
        Ok(self.reads.reader()?.source_snapshot(source)?)
    }

    pub(crate) fn source_read_context(
        &self,
        path: &Path,
        origin: &str,
        route: ReadRoute,
    ) -> SourceReadContext {
        SourceReadContext {
            source: SourceSelector {
                origin: origin.into(),
                machine: self.config.machine.clone(),
                path_bytes: path.as_os_str().as_encoded_bytes().to_vec(),
            },
            route,
            clock: self.config.clock.clone(),
            facts: None,
        }
    }
}
