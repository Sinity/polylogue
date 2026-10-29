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
    pub(crate) fn sampling_context(
        &mut self,
        path: &Path,
        origin: &str,
        reading_version: &str,
    ) -> Result<AcquisitionContext, EngineError> {
        let mut context = self.acquisition_context(path, origin, AcquisitionRoute::NativeCapture);
        let snapshot = self
            .writer
            .source_snapshot(context.source.clone(), reading_version)?;
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
        }
    }
}
