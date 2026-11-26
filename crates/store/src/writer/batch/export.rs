//! An export callback's seal covers its body, independent of what the receiver
//! accepts.
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
                .map_err(StoreError::invalid)?;
        let cursor = delivery.cursor().map_err(StoreError::invalid)?;
        self.delivery = Some(delivery);
        Ok(cursor)
    }

    pub fn validate_export(&self) -> Result<()> {
        if let Some(delivery) = &self.delivery {
            delivery.validate().map_err(StoreError::invalid)?;
            if delivery.body_hash != self.export_body_hash()? {
                return Err(StoreError::Invalid(
                    "export callback changed its sealed body".into(),
                ));
            }
        }
        Ok(())
    }

    fn export_body_hash(&self) -> Result<BlobHash> {
        let output = if self.prepared.released {
            self.complete_output()?
        } else {
            self.output.clone()
        };
        let body = (
            &self.parse_version,
            output,
            &self.counts,
            self.order_from,
            self.reset,
            self.repeats_whole_conversations,
            &self.failure,
            &self.resume,
        );
        let hash = canonical_json::canonical_hash("polylogue.export-delivery.body.v1", &body)
            .map_err(crate::encoding::identity_encoding)?;
        Ok(BlobHash(*hash.as_bytes()))
    }
}
