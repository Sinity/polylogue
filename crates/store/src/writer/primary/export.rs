//! The same callback admission fold runs before private semantic publication.
use super::*;
use crate::primary::ExportReadingKey;

impl Inner {
    pub(in crate::writer) fn pending_export_prefix(
        &self,
        key: &ExportReadingKey,
    ) -> Result<Option<crate::wire_shapes::ExportPrefix>> {
        key.validate()?;
        let mut prefix = crate::wire_shapes::load_prefix(
            &self.index,
            &key.input.id,
            key.chain,
            &key.reading_version,
        )?;
        let sinex = self.sinex.as_ref().ok_or_else(|| {
            StoreError::Invalid("pending export control requires its producer custody".into())
        })?;
        sinex.visit_pending_export_readings(key, &mut |captured| {
            if captured.export_key.as_ref() != Some(key)
                || captured.delivery.is_none()
                || captured.reading_version != key.reading_version
            {
                return Err(StoreError::Invalid(
                    "pending export callback changed its actual source binding".into(),
                ));
            }
            let step = captured.clone().into_step();
            crate::wire_shapes::advance_prefix(&mut prefix, key.length, &step)?;
            Ok(())
        })?;
        Ok(prefix)
    }
    pub(super) fn admit_primary_export(
        &self,
        key: &ExportReadingKey,
        step: &ParseStep,
    ) -> Result<bool> {
        let mut prefix = self.pending_export_prefix(key)?;
        crate::wire_shapes::advance_prefix(&mut prefix, key.length, step)
    }
}
