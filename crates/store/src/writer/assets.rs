//! Asset metadata stays local; attachment bindings are facts sent to Sinex in
//! primary mode and reach the index through the confirmed stream.
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
        let sinex = Arc::clone(self.sinex.as_ref().expect("primary asset writer"));
        let mut local = crate::export_items::ExportItems::new()?;
        let mut ordinal = 0_u64;
        let mut bindings = Vec::new();
        while let Some(item) = staged.next_item()? {
            let row: AssetRow = serde_json::from_reader(item.reader)?;
            match row {
                AssetRow::Binding { record } => {
                    record.validate()?;
                    if !crate::assets::bindings::known(&self.index, &record)? {
                        bindings.push(crate::primary::ProductWrite {
                            product: crate::primary::Product::AttachmentBinding {
                                record: Box::new(record),
                            },
                            location: None,
                            parents: Vec::new(),
                            machine: None,
                            output_index: bindings.len() as u64,
                            publish: true,
                        });
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
        if !bindings.is_empty() {
            sinex.prepare_products(&bindings)?;
        }
        crate::assets::install(&mut self.index, &self.archive, local)
    }
}
