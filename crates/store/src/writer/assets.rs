//! Derived member metadata stays local; immutable bindings require certified publication.
use super::*;
use crate::assets::AssetRow;

impl Inner {
    pub(super) fn install_assets(
        &mut self,
        mut staged: crate::export_items::ExportItems,
    ) -> Result<()> {
        if self.sinex.is_none() {
            return crate::assets::install(&mut self.index, &self.archive, staged);
        }
        self.primary_cache_ready()?;
        let mut local = crate::export_items::ExportItems::new()?;
        let mut ordinal = 0_u64;
        let mut missing = None;
        while let Some(item) = staged.next_item()? {
            let row: AssetRow = serde_json::from_reader(item.reader)?;
            match row {
                AssetRow::Binding { record } => {
                    if !crate::assets::bindings::known(&self.index, &record)? {
                        // Neutral declarations remain in the admitted source and
                        // message facts. A serialized witness cannot replace the
                        // original receiving admission and current owner basis.
                        missing.get_or_insert_with(|| record.context().message.to_string());
                    }
                }
                row => {
                    let bytes = serde_json::to_vec(&row)?;
                    local.push(item.range, ordinal.to_be_bytes().to_vec(), None, &bytes)?;
                    ordinal = ordinal.checked_add(1).ok_or_else(|| {
                        StoreError::Invalid("asset metadata order overflow".into())
                    })?;
                }
            }
        }
        crate::assets::install(&mut self.index, &self.archive, local)?;
        if let Some(message) = missing {
            return Err(StoreError::ReferenceUnavailable {
                reference: format!("attachment-binding:{message}"),
                reason: "canonical receiving admission and owner publication proof are unavailable"
                    .into(),
            });
        }
        Ok(())
    }
}
