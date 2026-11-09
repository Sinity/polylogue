//! Semantic export callback seals are independent of receiver admission.
use super::*;
use polylogue_model::batch::{ExportCursor, ExportDelivery, ExportDeliverySource};

impl ParseStep {
    pub fn seal_export(
        &mut self,
        ordinal: u64,
        source: ExportDeliverySource,
        previous_prefix: BlobHash,
    ) -> Result<ExportCursor> {
        let delivery =
            ExportDelivery::new(ordinal, source, self.export_body_hash()?, previous_prefix)
                .map_err(invalid_export)?;
        let cursor = delivery.cursor().map_err(invalid_export)?;
        self.delivery = Some(delivery);
        Ok(cursor)
    }

    pub fn validate_export(&self) -> Result<()> {
        if let Some(delivery) = &self.delivery {
            delivery.validate().map_err(invalid_export)?;
            if delivery.body_hash != self.export_body_hash()? {
                return Err(StoreError::Invalid(
                    "export callback changed its sealed semantic body".into(),
                ));
            }
        }
        Ok(())
    }

    fn export_body_hash(&self) -> Result<BlobHash> {
        let mut output = if self.prepared.released {
            self.complete_output()?
        } else {
            self.output.clone()
        };
        output.message_receipts.clear();
        output.session_receipts.clear();
        output.document_receipts.clear();
        output.event_receipts.clear();
        output.link_receipts.clear();
        output.fragment_receipts.clear();
        let body = (
            &self.reading_version,
            output,
            &self.counts,
            self.order_from,
            self.reset,
            self.repeats_whole_conversations,
            matches!(
                &self.progress,
                polylogue_model::batch::StepProgress::NoParserYet { .. }
            ),
            &self.checkpoint,
        );
        let hash = canonical_json::canonical_hash("polylogue.export-delivery.body.v1", &body)
            .map_err(crate::encoding::identity_encoding)?;
        Ok(BlobHash(*hash.as_bytes()))
    }
}

fn invalid_export(error: impl std::fmt::Display) -> StoreError {
    StoreError::Invalid(error.to_string())
}
