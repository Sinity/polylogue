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
        }
    }
}
impl Engine {
    pub(crate) fn sampling_context(
        &mut self,
        path: &Path,
        origin: &str,
        reading_version: &str,
    ) -> Result<AcquisitionContext, EngineError> {
        let mut context = self.acquisition_context(path, origin, AcquisitionRoute::NativeCapture);
        context.basis = Some(
            self.writer
                .source_snapshot(context.source.clone(), reading_version)?
                .basis,
        );
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
        }
    }
}
